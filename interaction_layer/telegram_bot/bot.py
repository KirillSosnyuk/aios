import asyncio
import logging
import os
import json
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from redis.asyncio import Redis

# Импортируем наши строгие контракты из общего модуля
from event_bus.schemas import BaseEvent, EventType, EventSeverity

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# Читаем белый список пользователей из .env
ALLOWED_USERS_RAW = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = {int(uid.strip()) for uid in ALLOWED_USERS_RAW.split(",") if uid.strip().isdigit()}

bot = Bot(token=TOKEN)
dp = Dispatcher()
redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

logging.basicConfig(level=logging.INFO)

# --- INGRESS: Отправка сообщений из TG в шину Redis ---
@dp.message()
async def handle_user_message(message: types.Message):
    if not message.text:
        return

    # Проверка белого списка: если он настроен и пользователя там нет, полностью игнорируем
    if ALLOWED_USERS and message.from_user.id not in ALLOWED_USERS:
        logging.warning(f"Unauthorized access attempt from user_id: {message.from_user.id} (@{message.from_user.username})")
        return

    # Формируем строгое событие по спецификации 12.3
    event = BaseEvent(
        source="telegram_bot",
        type=EventType.USER_MESSAGE_RECEIVED,
        payload={
            "chat_id": message.chat.id,
            "text": message.text,
            "username": message.from_user.username
        },
        severity=EventSeverity.INFO
    )
    
    # Сериализуем в JSON и отправляем во входящий поток
    event_json = event.model_dump_json()
    await redis_client.xadd("stream:ingress", {"data": event_json})
    logging.info(f"Sent Event {event.unique_id} to stream:ingress")

# --- EGRESS: Прослушивание команд из шины Redis и отправка в TG ---
async def listen_egress_stream():
    logging.info("Started listening to stream:egress...")
    # Для MVP используем простой механизм чтения, позже заменим на Consumer Groups
    last_id = '$' 
    
    while True:
        try:
            # Ждем новые сообщения в выходном потоке (таймаут 1 сек)
            response = await redis_client.xread({"stream:egress": last_id}, count=1, block=1000)
            if response:
                stream_name, messages = response[0]
                for msg_id, msg_data in messages:
                    event_data = json.loads(msg_data["data"])
                    
                    # Проверяем, что это команда отправки сообщения
                    if event_data.get("type") == EventType.COMMAND_REQUESTED:
                        payload = event_data.get("payload", {})
                        chat_id = payload.get("chat_id")
                        text = payload.get("text")
                        
                        if chat_id and text:
                            await bot.send_message(chat_id=chat_id, text=text)
                            logging.info(f"Delivered message to chat {chat_id}")
                    
                    last_id = msg_id
        except Exception as e:
            logging.error(f"Error in egress listener: {e}")
        await asyncio.sleep(0.1)

async def main():
    # Запускаем параллельно поллинг телеграма и прослушивание шины Redis
    asyncio.create_task(listen_egress_stream())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
