import asyncio
import logging
import os
import json
from redis.asyncio import Redis
from openai import AsyncOpenAI

from event_bus.schemas import BaseEvent, EventType, EventSeverity
from tools import search_web, TOOLS_MANIFEST  # Импортируем наш поиск

from datetime import datetime


REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Настройки моделей
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
CLOUD_API_URL = os.getenv("CLOUD_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

if "generativelanguage" in CLOUD_API_URL and "openai" not in CLOUD_API_URL:
    CLOUD_API_URL = CLOUD_API_URL.rstrip("/") + "/openai/"

CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5:3b")  # Убедись, что тут qwen2.5:3b
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash") 

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
logging.basicConfig(level=logging.INFO)

local_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama")
cloud_client = AsyncOpenAI(base_url=CLOUD_API_URL, api_key=CLOUD_API_KEY)

async def get_chat_history(chat_id: str) -> list:
    history_key = f"chat_history:{chat_id}"
    history_raw = await redis_client.lrange(history_key, 0, -1)
    if not history_raw:
        return []
    return [json.loads(msg) for msg in history_raw]

async def save_chat_message(chat_id: str, role: str, content: str):
    history_key = f"chat_history:{chat_id}"
    message = {"role": role, "content": content}
    await redis_client.rpush(history_key, json.dumps(message))
    await redis_client.ltrim(history_key, -10, -1)

async def handle_tool_calls(client, model_name, messages, tool_calls):
    """Вспомогательная функция для выполнения инструментов и повторного запроса к модели"""
    # Добавляем ответ модели с требованием вызова инструмента в контекст
    messages.append({
        "role": "assistant",
        "content": "",  # <--- ИСПРАВЛЕНИЕ: Ollama требует, чтобы это поле не было пустым (nil)
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in tool_calls
        ]
    })
    
    for tool_call in tool_calls:
        if tool_call.function.name == "search_web":
            arguments = json.loads(tool_call.function.arguments)
            query = arguments.get("query")
            
            # Вызываем наш реальный Python-скрипт поиска
            search_result = await search_web(query)
            
            # Отправляем результат обратно модели
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": "search_web",
                "content": str(search_result) # На всякий случай жестко приводим к строке
            })
            
    # Повторный запрос к модели, теперь имея на руках результаты поиска
    second_response = await client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    return second_response.choices[0].message.content

async def call_model_router(user_text: str, chat_id: str) -> str:
    current_date = datetime.now().strftime("%d %B %Y (день недели: %A)")

    system_prompt = (
    f"Ты — полезный ИИ-ассистент ядра AIOS. Текущая дата: {current_date}. "
    "У тебя есть доступ к интернету через инструмент search_web. "
    "ВАЖНОЕ ПРАВИЛО ПОИСКА: Если пользователь ищет информацию о мировых событиях, спорте, IT или новостях, "
    "ФОРМУЛИРУЙ ПОИСКОВЫЙ ЗАПРОС (query) НА АНГЛИЙСКОМ ЯЗЫКЕ (например, 'FIFA World Cup 2026 matches July 7'). "
    "Англоязычный поиск выдает более точные данные. "
    "Получив результаты, переведи их и ответь пользователю на чистом русском языке, не выдумывая фактов.")
    
    history = await get_chat_history(chat_id)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_text}]
    
    # Попытка 1: Локальная модель
    try:
        logging.info(f"Using Primary Local Model: {PRIMARY_MODEL}")
        response = await local_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            tools=TOOLS_MANIFEST,  # Передаем описание наших навыков
            timeout=8.0  # Чуть увеличили таймаут для 2-этапной генерации локалки
        )
        
        # Проверяем, хочет ли локальная модель вызвать инструмент
        if response.choices[0].message.tool_calls:
            logging.info(f"[Kernel] Локальная модель запросила вызов инструментов: {response.choices[0].message.tool_calls}")
            ai_reply = await handle_tool_calls(local_client, PRIMARY_MODEL, messages, response.choices[0].message.tool_calls)
        else:
            ai_reply = response.choices[0].message.content
            
        await save_chat_message(chat_id, "user", user_text)
        await save_chat_message(chat_id, "assistant", ai_reply)
        return ai_reply
        
    except Exception as e:
        logging.warning(f"Local model skipped, slow or failed: {e}. Routing to Cloud API...")
        
        # Попытка 2: Облако Gemini
        try:
            logging.info(f"Using Cloud Model: {FALLBACK_MODEL}")
            # Сбрасываем контекст для чистого вызова облака (без следов неудачной локалки)
            cloud_messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_text}]
            
            response = await cloud_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=cloud_messages,
                tools=TOOLS_MANIFEST  # Передаем плагины и в облако тоже!
            )
            
            if response.choices[0].message.tool_calls:
                logging.info(f"[Kernel] Облако запросило вызов инструментов: {response.choices[0].message.tool_calls}")
                ai_reply = await handle_tool_calls(cloud_client, FALLBACK_MODEL, cloud_messages, response.choices[0].message.tool_calls)
            else:
                ai_reply = response.choices[0].message.content
            
            await save_chat_message(chat_id, "user", user_text)
            await save_chat_message(chat_id, "assistant", ai_reply)
            return ai_reply
            
        except Exception as cloud_err:
            logging.error(f"All models failed: {cloud_err}")
            return "Извини, произошла ошибка соединения с ИИ-модулями."

async def process_event(event_id: str, event_json: str):
    try:
        event_dict = json.loads(event_json)
        event = BaseEvent(**event_dict)
        
        if event.type == EventType.USER_MESSAGE_RECEIVED:
            chat_id = event.payload.get("chat_id")
            user_text = event.payload.get("text")
            logging.info(f"Processing message from chat {chat_id}: '{user_text}'")
            
            ai_response = await call_model_router(user_text, str(chat_id))
            
            command_event = BaseEvent(
                trace_id=event.trace_id,
                source="ai_kernel",
                type=EventType.COMMAND_REQUESTED,
                payload={
                    "chat_id": chat_id,
                    "text": ai_response
                }
            )
            await redis_client.xadd("stream:egress", {"data": command_event.model_dump_json()})
            
    except Exception as e:
        logging.error(f"Error processing event {event_id}: {e}")

async def main():
    logging.info("AI Kernel started. Listening to stream:ingress...")
    last_id = '$'
    while True:
        try:
            response = await redis_client.xread({"stream:ingress": last_id}, count=1, block=1000)
            if response:
                stream_name, messages = response[0]
                for msg_id, msg_data in messages:
                    await process_event(msg_id, msg_data["data"])
                    last_id = msg_id
        except Exception as e:
            pass
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
