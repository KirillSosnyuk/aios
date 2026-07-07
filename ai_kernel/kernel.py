import asyncio
import logging
import os
import json
from typing import Optional
from redis.asyncio import Redis
from openai import AsyncOpenAI, RateLimitError

from event_bus.schemas import BaseEvent, EventType, EventSeverity
from tools import search_web, remember_preference, add_memory_note, TOOLS_MANIFEST  # Импортируем наши инструменты
import memory  # Слой памяти/профиля (раздел 14 спецификации)

from datetime import datetime


REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Настройки моделей
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
CLOUD_API_URL = os.getenv("CLOUD_API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")

if "generativelanguage" in CLOUD_API_URL and "openai" not in CLOUD_API_URL:
    CLOUD_API_URL = CLOUD_API_URL.rstrip("/") + "/openai/"

CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "")
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5:3b")  # Дефолт держим в синхроне с PRIMARY_MODEL из .env
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
logging.basicConfig(level=logging.INFO)

# ВАЖНО: у openai-python по умолчанию max_retries=2 — то есть сам SDK молча
# повторяет зависший запрос ещё 2 раза ПОВЕРХ нашего timeout=8.0, и реальное
# ожидание перед фоллбэком на облако растягивается до ~24с вместо 8. Локалке
# ретраить нечего — если она не уложилась в timeout, она не уложится и на
# повторной попытке, поэтому отключаем ретраи именно для local_client.
local_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama", max_retries=0)
cloud_client = AsyncOpenAI(base_url=CLOUD_API_URL, api_key=CLOUD_API_KEY)

# --- Разрешения инструментов (раздел 18 спецификации) -----------------------
# Статическая карта вместо таблицы в БД — на этом этапе проекта инструментов
# мало и они не меняются в рантайме; если это изменится, легко перенести в
# Postgres без изменения вызывающего кода (сигнатура останется той же).
TOOL_PERMISSION_LEVELS = {
    "search_web": 0,           # только чтение, без побочных эффектов
    "remember_preference": 0,  # запись факта о пользователе — безвредно (18.3)
    "add_memory_note": 0,      # запись свободной заметки — тоже безвредно
}
DEFAULT_PERMISSION_LEVEL = 1  # для инструментов, не описанных явно выше

# Интерактивный UX подтверждения (кнопки в Telegram и т.п. для уровня 2+) ещё
# не реализован — его нельзя ни протестировать, ни использовать вслепую.
# Поэтому вместо того, чтобы молча выполнить рискованное действие или упасть,
# такие вызовы блокируются понятным сообщением. Это временное ограничение:
# как только появится первый по-настоящему рискованный skill, здесь появится
# реальный запрос подтверждения пользователя.
MAX_AUTO_PERMISSION_LEVEL = 1

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


async def _dispatch_tool_call(tool_call, user_id: Optional[int]) -> str:
    """Выполняет один вызов инструмента и возвращает текст tool-сообщения.

    Единая точка диспетчеризации: раньше поиск был захардкожен прямо внутри
    handle_tool_calls, и каждый новый инструмент добавлял бы туда ещё одну
    ветку if/else. Теперь handle_tool_calls занимается только раундами
    диалога и проверкой разрешений, а конкретная реализация — здесь.
    """
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        arguments = {}

    if name == "search_web":
        return str(await search_web(arguments.get("query")))
    elif name == "remember_preference":
        return str(await remember_preference(
            user_id,
            arguments.get("category"),
            arguments.get("key"),
            arguments.get("value"),
        ))
    elif name == "add_memory_note":
        return str(await add_memory_note(
            user_id,
            arguments.get("content"),
            arguments.get("category", "episodic"),
        ))
    else:
        return f"Инструмент '{name}' не существует."


async def handle_tool_calls(
    client, model_name, messages, tool_calls,
    user_id: Optional[int] = None,
    trace_id: Optional[str] = None,
    timeout: float = 15.0,
    max_rounds: int = 6,
):
    """Выполняет вызовы инструментов и опрашивает модель дальше, пока она не
    даст обычный текстовый ответ (или не кончится лимит раундов max_rounds).

    Раньше делался только ОДИН раунд, и повторный запрос уходил без tools= —
    из-за этого некоторые модели (например, gpt-oss-120b через Groq), которые
    на дозаправленном контексте иногда хотят вызвать инструмент ещё раз
    (например, доуточнить поиск), получали жёсткий 400 Bad Request
    ('Tool choice is none, but model called a tool') вместо обычного ответа.

    Каждый вызов инструмента теперь проходит проверку уровня разрешения
    (раздел 18 спецификации) и пишется в журнал решений (memory.log_decision,
    раздел 14.8/23), прежде чем реально выполниться.

    max_rounds было поднято с 3 до 6: облачная модель (gpt-oss-120b через Groq)
    на длинных "расскажи о себе" сообщениях иногда вызывает remember_preference
    ПО ОДНОМУ факту за раунд вместо пакета за один раз — при 3 раундах и
    богатом сообщении (десяток фактов) лимит исчерпывался раньше, чем модель
    успевала дать финальный текстовый ответ, и пользователь получал только
    служебное сообщение о неудаче вместо ответа (при этом часть фактов уже
    успевала сохраниться — см. system_prompt в call_model_router, где теперь
    отдельно объяснено, что нужно укладываться в несколько вызовов, а не длинную серию).
    """
    for _ in range(max_rounds):
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
            name = tool_call.function.name
            level = TOOL_PERMISSION_LEVELS.get(name, DEFAULT_PERMISSION_LEVEL)

            if level > MAX_AUTO_PERMISSION_LEVEL:
                content = (
                    f"Инструмент '{name}' требует подтверждения пользователя, "
                    "а интерактивное подтверждение пока не реализовано — "
                    "действие отменено."
                )
                await memory.log_decision(
                    trace_id, user_id, "tool_blocked_permission",
                    model_used=model_name, permission_level=level,
                    details={"tool": name},
                )
            else:
                content = await _dispatch_tool_call(tool_call, user_id)
                await memory.log_decision(
                    trace_id, user_id, "tool_executed",
                    model_used=model_name, permission_level=level,
                    details={"tool": name},
                )

            # Отправляем результат обратно модели. Каждому tool_call нужен
            # ответный "tool"-message с тем же id — иначе следующий запрос к
            # API упадёт из-за рассинхрона истории сообщений.
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": name,
                "content": content,
            })

        # Повторный запрос к модели, теперь имея на руках результаты вызовов.
        # ВАЖНО: tools= передаём и здесь — иначе модель, которая захочет
        # вызвать инструмент ещё раз, упрётся в ошибку вместо обычного ответа.
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_MANIFEST,
            timeout=timeout
        )

        if response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            continue  # модель просит ещё один раунд

        return response.choices[0].message.content

    # Лимит раундов исчерпан, а модель всё ещё пытается вызывать инструменты —
    # не зависаем и не падаем, отдаём понятное сообщение вместо ошибки. Часть
    # вызовов (например, сохранение фактов в память) к этому моменту уже могла
    # успешно выполниться — сама фраза ниже нейтральна и не про поиск конкретно,
    # т.к. до лимита раундов может довести любой инструмент, не только поиск.
    return "Не получилось собрать окончательный ответ за разумное число шагов — попробуй переформулировать запрос или отправить его покороче."

async def call_model_router(
    user_text: str,
    chat_id: str,
    telegram_user_id: Optional[int] = None,
    display_name: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> str:
    current_date = datetime.now().strftime("%d %B %Y (день недели: %A)")

    # Разрешаем внутренний user_id по telegram_id (создаёт пользователя при
    # первом обращении). Рассчитано на несколько устройств/каналов на одного
    # человека в будущем (голос/колонка) — все они будут вести к тому же
    # users.id, поэтому память и разрешения не привязаны к Telegram напрямую.
    user_id = await memory.get_or_create_user(telegram_user_id, display_name)
    profile_blurb = await memory.build_profile_blurb(user_id)

    system_prompt = (
        f"Ты — полезный ИИ-ассистент ядра AIOS. Текущая дата: {current_date}. "
        "У тебя есть доступ к интернету через инструмент search_web, а также два "
        "инструмента памяти: remember_preference — для чётких фактов вида "
        "категория/ключ/значение (профессия, город, диета, часовой пояс, язык), "
        "и add_memory_note — для более развёрнутого контекста, который не "
        "сводится к одной паре (интересы с деталями, привычки, история). "
        "ВАЖНОЕ ПРАВИЛО ПАМЯТИ: если пользователь сообщает о себе МНОГО фактов "
        "за раз (представляется, описывает увлечения, просит что-то запомнить) — "
        "НЕ пытайся сохранить каждую мелкую деталь отдельным вызовом "
        "remember_preference, это слишком много шагов подряд и часть ответа "
        "потеряется. Вместо этого: вызови remember_preference не больше 3-4 раз "
        "для самых чётких структурированных фактов (имя, профессия/статус, "
        "город, язык), а всё остальное (увлечения с деталями, привычки, "
        "контекст) сохрани ОДНИМ вызовом add_memory_note в виде краткого связного "
        "резюме. Суммарно старайся уложиться в 4-5 вызовов инструментов памяти "
        "на одно сообщение пользователя — и обязательно дай обычный текстовый "
        "ответ пользователю после этого, не обрывай диалог одними вызовами. "
        "ВАЖНОЕ ПРАВИЛО ПОИСКА: Если пользователь ищет информацию о мировых событиях, спорте, IT или новостях, "
        "ФОРМУЛИРУЙ ПОИСКОВЫЙ ЗАПРОС (query) НА АНГЛИЙСКОМ ЯЗЫКЕ (например, 'FIFA World Cup 2026 matches July 7'). "
        "Англоязычный поиск выдает более точные данные. "
        "Получив результаты, переведи их и ответь пользователю на чистом русском языке, не выдумывая фактов.\n"
        f"{profile_blurb}"
    )

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
            ai_reply = await handle_tool_calls(
                local_client, PRIMARY_MODEL, messages, response.choices[0].message.tool_calls,
                user_id=user_id, trace_id=trace_id, timeout=8.0,
            )
        else:
            ai_reply = response.choices[0].message.content

        await save_chat_message(chat_id, "user", user_text)
        await save_chat_message(chat_id, "assistant", ai_reply)
        await memory.log_decision(trace_id, user_id, "answered_local", model_used=PRIMARY_MODEL)
        return ai_reply

    except Exception as e:
        logging.warning(f"Local model skipped, slow or failed: {e}. Routing to Cloud API...")

        # Попытка 2: Облачный фоллбэк (провайдер настраивается через .env — см. FALLBACK_MODEL/CLOUD_API_*)
        try:
            logging.info(f"Using Cloud Model: {FALLBACK_MODEL}")
            # Если локалка уже успела вызвать инструмент и получить результат
            # (в messages появилось tool-сообщение), переиспользуем их вместо
            # повторного вызова с нуля — иначе тот же самый запрос улетает
            # (например, во все поисковики) второй раз подряд, и это чистая
            # трата времени.
            already_searched = any(m.get("role") == "tool" for m in messages)
            if already_searched:
                logging.info("[Kernel] Локалка уже вызывала инструмент — переиспользую результат для облака")
                cloud_messages = messages
            else:
                # Сбрасываем контекст для чистого вызова облака (без следов неудачной локалки)
                cloud_messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_text}]

            response = await cloud_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=cloud_messages,
                tools=TOOLS_MANIFEST  # Передаем плагины и в облако тоже!
            )

            if response.choices[0].message.tool_calls:
                logging.info(f"[Kernel] Облако запросило вызов инструментов: {response.choices[0].message.tool_calls}")
                ai_reply = await handle_tool_calls(
                    cloud_client, FALLBACK_MODEL, cloud_messages, response.choices[0].message.tool_calls,
                    user_id=user_id, trace_id=trace_id, timeout=20.0,
                )
            else:
                ai_reply = response.choices[0].message.content

            await save_chat_message(chat_id, "user", user_text)
            await save_chat_message(chat_id, "assistant", ai_reply)
            await memory.log_decision(trace_id, user_id, "answered_cloud_fallback", model_used=FALLBACK_MODEL)
            return ai_reply

        except RateLimitError as cloud_err:
            # Отдельно от прочих ошибок: это не "не достучались", а упёрлись в
            # квоту/лимит запросов облачного провайдера. Ретраить бессмысленно —
            # квота не появится за секунды.
            logging.error(f"Cloud model rate-limited or quota exceeded: {cloud_err}")
            await memory.log_decision(
                trace_id, user_id, "cloud_rate_limited",
                model_used=FALLBACK_MODEL, details={"error": str(cloud_err)},
            )
            return (
                f"ИИ временно недоступен: облачная модель ({FALLBACK_MODEL}) упёрлась в лимит "
                "запросов. Попробуй чуть позже — лимит обычно сбрасывается в течение суток."
            )
        except Exception as cloud_err:
            logging.error(f"All models failed: {cloud_err}")
            await memory.log_decision(
                trace_id, user_id, "all_models_failed",
                details={"error": str(cloud_err)},
            )
            return "Извини, произошла ошибка соединения с ИИ-модулями."

async def keep_local_model_warm():
    """Держит локальную модель постоянно загруженной в Ollama.

    Ollama по умолчанию выгружает модель из памяти после ~5 минут простоя
    (см. вывод `ollama ps`, колонка UNTIL) — а следующий запрос после этого
    платит временем холодного старта (загрузка весов в VRAM) ПОВЕРХ обычной
    генерации, что легко выбивает наш timeout=8.0 даже на маленькой модели
    и на простом сообщении без поиска. Пингуем чаще, чем таймаут простоя,
    чтобы модель не успевала выгружаться в нормальном режиме работы бота.
    """
    while True:
        try:
            await local_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=30.0,  # прогрев в фоне, никто его не ждёт — можно не спешить
            )
            logging.info("[Kernel] Локальная модель прогрета (keep-alive)")
        except Exception as e:
            logging.warning(f"[Kernel] Не удалось прогреть локальную модель: {e}")
        # Пингуем СРАЗУ при старте (закрывает холодный старт сразу после
        # docker compose up), а затем каждые 3 минуты — с запасом до
        # дефолтных 5 минут простоя, после которых Ollama выгружает модель.
        await asyncio.sleep(180)


async def process_event(event_id: str, event_json: str):
    try:
        event_dict = json.loads(event_json)
        event = BaseEvent(**event_dict)

        if event.type == EventType.USER_MESSAGE_RECEIVED:
            chat_id = event.payload.get("chat_id")
            user_text = event.payload.get("text")
            telegram_user_id = event.payload.get("telegram_user_id")
            display_name = event.payload.get("display_name")
            logging.info(f"Processing message from chat {chat_id}: '{user_text}'")

            ai_response = await call_model_router(
                user_text, str(chat_id),
                telegram_user_id=telegram_user_id,
                display_name=display_name,
                trace_id=str(event.trace_id),
            )

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
    await memory.init_pool()  # Ретраит сам, не падает, если Postgres ещё не готов
    asyncio.create_task(keep_local_model_warm())
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
