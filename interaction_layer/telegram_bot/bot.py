import asyncio
import logging
import os
import json
import time
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
            "username": message.from_user.username,
            # Внутренний users.id в памяти ядра резолвится по telegram_user_id
            # (см. memory.get_or_create_user) — не по username/chat_id, т.к.
            # они могут меняться или отсутствовать. display_name — только для
            # первичного заполнения профиля, дальше не перезаписывается.
            "telegram_user_id": message.from_user.id,
            "display_name": message.from_user.full_name,
        },
        severity=EventSeverity.INFO
    )
    
    # Сериализуем в JSON и отправляем во входящий поток
    event_json = event.model_dump_json()
    await redis_client.xadd("stream:ingress", {"data": event_json})
    logging.info(f"Sent Event {event.unique_id} to stream:ingress")

TELEGRAM_MAX_LEN = 4096  # Жёсткий лимит Telegram Bot API на одно текстовое сообщение


def split_message(text: str, max_length: int = TELEGRAM_MAX_LEN) -> list:
    """Режет длинный текст на части не длиннее max_length.

    Старается резать по границе абзаца / предложения / пробела, чтобы не
    рвать слова посередине. Нужен потому, что send_message с текстом длиннее
    4096 символов падает с ошибкой 'message is too long' — раньше это просто
    ронял доставку целиком.
    """
    text = text.strip()
    if len(text) <= max_length:
        return [text]

    chunks = []
    while len(text) > max_length:
        window = text[:max_length]
        cut = window.rfind("\n\n")
        if cut == -1:
            cut = window.rfind("\n")
        if cut == -1:
            cut = window.rfind(". ")
            if cut != -1:
                cut += 1
        if cut == -1:
            cut = window.rfind(" ")
        if cut <= 0:
            cut = max_length  # совсем без пробелов — режем жёстко

        chunks.append(text[:cut].strip())
        text = text[cut:].lstrip()

    if text:
        chunks.append(text)
    return chunks


# --- Статусные сообщения ("думаю локально…", "ищу в интернете…" и т.п.) ----
# Раздел 4 спецификации — Explainability: показываем пользователю, что именно
# сейчас происходит, вместо молчания на 2-3 минуты. Одно редактируемое
# сообщение на чат, которое исчезает перед финальным ответом.
#
# Ограничения Telegram Bot API, которые здесь учтены:
#   - editMessageText падает с "message is not modified", если текст не
#     изменился — пропускаем такие правки, а не считаем их ошибкой.
#   - Слишком частые правки одного сообщения ловят 429 (retry_after) — держим
#     минимальный интервал между правками на чат и просто пропускаем правку,
#     если он не выдержан (следующий статус или финальный ответ всё равно
#     скоро придут).
#   - Если сообщение уже удалено/недоступно (например, юзер сам его стёр) —
#     deleteMessage/editMessageText упадут; ловим это и просто сбрасываем
#     состояние, чтобы не застрять.
STATUS_MIN_EDIT_INTERVAL = 1.5  # сек между правками одного статусного сообщения
_status_state = {}  # chat_id -> {"message_id": int, "last_edit": float, "last_text": str}


async def _show_status(chat_id, text: str):
    if not chat_id or not text:
        return
    state = _status_state.get(chat_id)
    now = time.monotonic()

    if state is None:
        try:
            msg = await bot.send_message(chat_id=chat_id, text=text, disable_notification=True)
            _status_state[chat_id] = {"message_id": msg.message_id, "last_edit": now, "last_text": text}
        except Exception as e:
            logging.warning(f"Не удалось отправить статусное сообщение в чат {chat_id}: {e}")
        return

    if text == state["last_text"]:
        return  # тот же текст — editMessageText только упадёт на "not modified"
    if now - state["last_edit"] < STATUS_MIN_EDIT_INTERVAL:
        return  # слишком часто правим — пропускаем, догонит следующий статус

    try:
        await bot.edit_message_text(chat_id=chat_id, message_id=state["message_id"], text=text)
        state["last_edit"] = now
        state["last_text"] = text
    except Exception as e:
        # Сообщение могло исчезнуть (юзер удалил, истёк срок правки и т.п.) —
        # не пытаемся оживить его вечно, просто забываем и начнём заново со
        # следующего статуса.
        logging.warning(f"Не удалось обновить статусное сообщение в чате {chat_id}: {e}")
        _status_state.pop(chat_id, None)


async def _clear_status(chat_id):
    state = _status_state.pop(chat_id, None)
    if not state:
        return
    try:
        await bot.delete_message(chat_id=chat_id, message_id=state["message_id"])
    except Exception as e:
        logging.warning(f"Не удалось удалить статусное сообщение в чате {chat_id}: {e}")


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
                    # ВАЖНО: продвигаем позицию сразу, до обработки записи.
                    # Раньше last_id обновлялся в конце тела цикла, и если
                    # send_message падал (например, 'message is too long'),
                    # позиция в стриме не двигалась — та же самая запись
                    # перечитывалась и падала снова на каждой итерации,
                    # бесконечно, и весь бот переставал доставлять что-либо.
                    last_id = msg_id
                    try:
                        event_data = json.loads(msg_data["data"])
                        event_type = event_data.get("type")
                        payload = event_data.get("payload", {})
                        # str(...) — защитная нормализация типа. Раньше STATUS_UPDATE
                        # и COMMAND_REQUESTED иногда несли chat_id разных типов
                        # (str из ядра для статусов, "голый" int для финального
                        # ответа) — _status_state хранится по chat_id как ключу
                        # словаря, а int(123) и str("123") в Python это разные
                        # ключи. Из-за этого статус прошлого запроса не находился
                        # для удаления и следующий запрос редактировал чужое
                        # сообщение. Приводим к строке здесь ещё раз на всякий
                        # случай, независимо от того, что именно прислало ядро.
                        raw_chat_id = payload.get("chat_id")
                        chat_id = str(raw_chat_id) if raw_chat_id is not None else None

                        if event_type == EventType.STATUS_UPDATE:
                            await _show_status(chat_id, payload.get("text"))

                        elif event_type == EventType.COMMAND_REQUESTED:
                            text = payload.get("text")
                            if chat_id and text:
                                # Статусное сообщение исчезает, и на его месте
                                # (точнее — следующим сообщением) появляется ответ.
                                await _clear_status(chat_id)
                                for chunk in split_message(text):
                                    await bot.send_message(chat_id=chat_id, text=chunk)
                                logging.info(f"Delivered message to chat {chat_id}")
                    except Exception as msg_err:
                        logging.error(f"Failed to deliver event {msg_id}: {msg_err}")
        except Exception as e:
            logging.error(f"Error in egress listener: {e}")
            await asyncio.sleep(2.0)  # не спамить лог тем же сбоем 10 раз в секунду
            continue
        await asyncio.sleep(0.1)

async def main():
    # Запускаем параллельно поллинг телеграма и прослушивание шины Redis
    asyncio.create_task(listen_egress_stream())
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
