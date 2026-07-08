import asyncio
import logging
import os
import json
import re
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
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen3:8b")  # Дефолт держим в синхроне с PRIMARY_MODEL из .env
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash")

redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
logging.basicConfig(level=logging.INFO)

# ВАЖНО: у openai-python по умолчанию max_retries=2 — то есть сам SDK молча
# повторяет зависший запрос ещё 2 раза ПОВЕРХ нашего timeout=8.0, и реальное
# ожидание перед фоллбэком на облако растягивается до ~24с вместо 8. Локалке
# ретраить нечего — если она не уложилась в timeout, она не уложится и на
# повторной попытке, поэтому отключаем ретраи именно для local_client.
# То же самое соображение теперь и для cloud_client: на бесплатном тарифе
# Groq под реальной нагрузкой (несколько раундов инструментов подряд) сервер
# стал отвечать 429 с Retry-After 46-54 секунды. Стандартный ретрай SDK молча
# ждёт это время и повторяет — пользователь получает не "ошибку", а минуты
# полной тишины, что для чат/голосового интерфейса хуже понятного отказа.
# Теперь 429 сразу поднимается как RateLimitError и ловится в
# _run_cloud_fallback с понятным сообщением, вместо слепого ожидания.
local_client = AsyncOpenAI(base_url=OLLAMA_URL, api_key="ollama", max_retries=0)
cloud_client = AsyncOpenAI(base_url=CLOUD_API_URL, api_key=CLOUD_API_KEY, max_retries=0)

# Потолок на генерацию облака за один вызов. КРИТИЧНО для бесплатного Groq:
# там лимит TPM=8000 токенов/мин, а Groq считает «запрошено» = токены промпта +
# ЗАРЕЗЕРВИРОВАННЫЙ max_completion. Без явного max_tokens gpt-oss-120b резервирует
# огромный бюджет вывода — в реальном инциденте 2026-07-08 запрос оценился в
# 18924 токена (413 Payload Too Large) при промпте всего в пару тысяч. Явный
# небольшой max_tokens убирает эту «резервацию» и удерживает запрос под лимитом.
CLOUD_MAX_TOKENS = 1024

# --- Разрешения инструментов (раздел 18 спецификации) -----------------------
# Статическая карта вместо таблицы в БД — на этом этапе проекта инструментов
# мало и они не меняются в рантайме; если это изменится, легко перенести в
# Postgres без изменения вызывающего кода (сигнатура останется той же).
TOOL_PERMISSION_LEVELS = {
    "search_web": 0,           # только чтение, без побочных эффектов
    "remember_preference": 0,  # запись факта о пользователе — безвредно (18.3)
    "add_memory_note": 0,      # запись свободной заметки — тоже безвредно
}
# 2026-07-08: было 1 (= авто-разрешено, т.к. MAX_AUTO_PERMISSION_LEVEL тоже
# 1). Это значит, что ЛЮБОЙ новый инструмент, который забыли явно добавить в
# TOOL_PERMISSION_LEVELS, по умолчанию молча авто-выполнялся бы — fail-open
# вместо fail-closed. Сейчас это не эксплуатируется (все 3 существующих
# инструмента явно перечислены выше), но это неверная защита по умолчанию на
# будущее: новая, потенциально опасная возможность должна по умолчанию
# требовать подтверждения, пока её явно не пометят безопасной, а не наоборот.
DEFAULT_PERMISSION_LEVEL = 2  # для инструментов, не описанных явно выше — блокируется (см. MAX_AUTO_PERMISSION_LEVEL)

# 2026-07-08: жёсткий потолок на число обращений к search_web ЗА ОДИН ход
# пользователя (через все раунды handle_tool_calls вместе, не за раунд).
# Раньше это регулировалось только текстовой инструкцией в system_prompt
# ("НЕ переформулируй запрос бесконечно") — на практике облачная модель
# (gpt-oss-120b) её не всегда слушается: в реальном случае она
# переформулировала запрос про рост Галкина и искала заново 4 раза подряд,
# хотя оба поисковых движка (Google, Brave) каждый раз возвращали пусто/429.
# Кумулятивный расход токенов на эти бесплодные раунды исчерпал TPM-лимит
# Groq ещё ДО финального ответа — пользователь не получил вообще ничего,
# даже честного "не нашёл". Тот же урок, что и с remember_preference раньше:
# полагаться на то, что модель послушает промпт-инструкцию про лимит попыток,
# недостаточно — теперь это соблюдается механически: после
# MAX_SEARCH_CALLS_PER_TURN вызовов search_web инструмент физически
# убирается из списка доступных на следующий раунд (а не просто "просим её
# так не делать"), и модель вынуждена ответить тем, что уже есть.
MAX_SEARCH_CALLS_PER_TURN = 2

# Интерактивный UX подтверждения (кнопки в Telegram и т.п. для уровня 2+) ещё
# не реализован — его нельзя ни протестировать, ни использовать вслепую.
# Поэтому вместо того, чтобы молча выполнить рискованное действие или упасть,
# такие вызовы блокируются понятным сообщением. Это временное ограничение:
# как только появится первый по-настоящему рискованный skill, здесь появится
# реальный запрос подтверждения пользователя.
MAX_AUTO_PERMISSION_LEVEL = 1

# Жёсткий потолок на обработку ОДНОГО сообщения целиком (раздел 13 —
# устойчивость AI Kernel). main() — единственный последовательный consumer
# stream:ingress: пока process_event() не завершится, ядро физически не
# может взять следующее сообщение — ни от этого пользователя, ни от других
# двух. Если call_model_router где-то зависнет БЕЗ исключения (например,
# 2026-07-08: pool.acquire() у asyncpg без timeout мог ждать освобождения
# соединения вечно — try/except его не ловит, потому что это не исключение,
# а обычное ожидание) — раньше это останавливало бота целиком без единой
# строчки в логах. Теперь такой сценарий гарантированно обрывается по
# таймауту, и ядро возвращается к очереди, а не подвисает навсегда.
# 180 секунд (2026-07-08: поднято со 150). Легитимный сценарий сам по себе
# может занять до ~145с (LOCAL_TIMEOUT=25 + до 6 раундов инструментов по
# timeout=20 каждый на облаке = 25+120), а 150 оставляли только 5с запаса на
# всё остальное (обращения к памяти, публикацию статусов, сетевые накладные
# расходы) — слишком туго: легитимный медленный запрос рисковал попасть под
# тот же таймаут, что и настоящее зависание. 180 даёт разумный запас, не
# превращаясь в бесконечность.
PROCESS_HARD_DEADLINE = 180.0

# Локальная модель — быстрый путь. Ограничиваем длину её ответа: во-первых,
# это напрямую сокращает время генерации (меньше токенов — меньше времени),
# во-вторых — не даёт ей писать развёрнутые эссе на простые бытовые вопросы
# (например, 4 рецепта вместо одного совета). Облако намеренно НЕ ограничено —
# когда до него доходит очередь, пользователь уже готов подождать чуть дольше
# ради более полного ответа.
# 2026-07-08: поднято с 400 до 600. Раньше thinking-режим Qwen3 подавлялся
# только через chat_template_kwargs, который OpenAI-совместимый эндпоинт
# Ollama, как выяснилось, вообще игнорирует (см. reasoning_effort ниже и
# комментарий в _extract_final_reply) — то есть thinking, скорее всего, был
# активен всё это время и ел токены ИЗ ЭТОГО ЖЕ бюджета раньше, чем модель
# доходила до видимого ответа (отсюда и обрыв ответа посреди слова). Теперь,
# когда reasoning_effort="none" должен реально отключать thinking, 600 —
# это запас на случай, если он всё-таки где-то просочится, а не компенсация
# основной причины. 600 токенов при ~55 ток/с на этом железе — это ~11с,
# всё ещё далеко от LOCAL_TIMEOUT=25.
LOCAL_MAX_TOKENS = 600
# ВРЕМЕННО поднято с 8 до 25 секунд (2026-07-08): облачный фоллбэк (бесплатный
# тариф Groq) сейчас сам по себе периодически недоступен на 46-54 секунды
# (Retry-After под нагрузкой) — при таком раскладе быстрый таймаут локалки
# больше не гарантирует быстрый ответ вообще, а просто быстрее пересылает
# запрос туда, где он всё равно зависнет надолго. Пока нет платного тарифа
# (сознательное решение — проект без бюджета), выгоднее дать локалке реальный
# шанс дотянуть до ответа, чем экономить секунды перед фоллбэком, который сам
# не быстрее. Как только будут реальные цифры токенов/сек с этого железа
# (см. `ollama run qwen3:8b --verbose`), таймаут стоит пересчитать осмысленно,
# а не подбирать вслепую ещё раз.
LOCAL_TIMEOUT = 25.0
LOCAL_BREVITY_SUFFIX = (
    "\n\n/no_think\n"  # Явный текстовый переключатель Qwen3 — дублирует ниже extra_body
    "ТЫ СЕЙЧАС РАБОТАЕШЬ КАК БЫСТРАЯ ЛОКАЛЬНАЯ МОДЕЛЬ: отвечай МАКСИМАЛЬНО "
    "КОРОТКО — 2-4 предложения, одна конкретная рекомендация вместо перечисления "
    "вариантов, без пошаговых рецептов, инструкций и длинных списков. Если тема "
    "того заслуживает — дай сжатую суть и предложи спросить подробнее, если "
    "понадобится."
)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_TRUNCATION_FALLBACK_MSG = (
    "Не успел сформулировать полный ответ за отведённое время — спроси ещё "
    "раз, желательно покороче."
)


def _extract_final_reply(response, context_label: str = "") -> str:
    """Единственная точка превращения сырого ответа API в текст для пользователя.

    2026-07-08, ревизия 2: изначально причиной обрыва ответа посреди слова
    ("...занимался б" в истории с Галкиным) считался просочившийся в content
    <think>-блок. Проверка по документации Ollama (docs.ollama.com/api/
    openai-compatibility, github.com/ollama/ollama/issues/14820) показала,
    что это было верно лишь отчасти: OpenAI-совместимый эндпоинт Ollama
    ВООБЩЕ НЕ смотрит на chat_template_kwargs.enable_thinking (это параметр
    нативного шаблона Qwen3, который на этом эндпоинте молча игнорируется) —
    официальный переключатель thinking там reasoning_effort="none" (см. его
    в call_model_router). То есть все предыдущие попытки подавить thinking,
    скорее всего, вообще не работали, и обрыв ответа мог случаться и БЕЗ
    каких-либо <think>-тегов в content — просто потому что невидимые
    reasoning-токены съедали LOCAL_MAX_TOKENS до того, как модель успевала
    записать видимый ответ.

    Поэтому проверка здесь НЕ полагается на наличие <think>-тегов как на
    единственный сигнал обрыва — авторитетный сигнал это finish_reason
    ("length" = обрезано по max_tokens, вне зависимости от того, что именно
    успело попасть в content). <think>-теги вырезаются отдельно как чистая
    защитная зачистка на случай, если thinking всё же просочится в content
    целиком на каком-то будущем/другом провайдере.
    """
    choice = response.choices[0]
    finish_reason = choice.finish_reason
    raw = choice.message.content

    truncated_inside_think = bool(raw) and "<think>" in raw and "</think>" not in raw
    cleaned = _THINK_TAG_RE.sub("", raw).strip() if raw else raw

    if finish_reason == "length" or truncated_inside_think or not cleaned:
        logging.warning(
            f"[Kernel] Ответ обрезан или пуст ({context_label}, "
            f"finish_reason={finish_reason}) — сырой текст: {raw!r}"
        )
        return _TRUNCATION_FALLBACK_MSG

    return cleaned


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


async def _publish_status(chat_id, trace_id: Optional[str], text: str):
    """Публикует лёгкое статусное событие (раздел 4 — Explainability): что
    именно сейчас делает ядро — думает локально, подключает облако, ищет,
    запоминает и т.п. Interaction layer (bot.py) показывает это как одно
    редактируемое сообщение, которое исчезает перед финальным ответом.

    Никогда не бросает исключение наружу — статус вспомогательный, не должен
    иметь возможность сорвать доставку настоящего ответа.
    """
    if not chat_id:
        return
    try:
        # trace_id у BaseEvent — обязательный UUID с default_factory=uuid4;
        # передаём его, только если реально есть (иначе пусть Pydantic сам
        # сгенерирует новый, а не падает на None).
        kwargs = {"source": "ai_kernel", "type": EventType.STATUS_UPDATE, "payload": {"chat_id": chat_id, "text": text}}
        if trace_id:
            kwargs["trace_id"] = trace_id
        status_event = BaseEvent(**kwargs)
        await redis_client.xadd("stream:egress", {"data": status_event.model_dump_json()})
    except Exception as e:
        logging.warning(f"[Kernel] Не удалось опубликовать статус: {e}")


def _describe_tool_call(tool_call) -> str:
    """Короткое человекочитаемое описание вызова инструмента для статуса."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except (json.JSONDecodeError, TypeError):
        args = {}

    if name == "search_web":
        query = args.get("query")
        return f"Ищу в интернете: {query}" if query else "Ищу в интернете…"
    elif name == "remember_preference":
        n = len(args.get("facts") or [])
        return f"Запоминаю {n} факт(ов) о тебе…" if n else "Запоминаю факты о тебе…"
    elif name == "add_memory_note":
        return "Сохраняю заметку о тебе…"
    else:
        return f"Выполняю: {name}…"


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
        return str(await remember_preference(user_id, arguments.get("facts")))
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
    chat_id=None,
    timeout: float = 15.0,
    max_rounds: int = 6,
    create_extra: Optional[dict] = None,
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
    раздел 14.8/23), прежде чем реально выполниться. Перед каждым раундом
    публикуется статус (chat_id) с кратким описанием того, что вот-вот
    произойдёт — один статус на раунд, а не на отдельный вызов, чтобы не
    заваливать Telegram частыми правками одного сообщения.

    max_rounds было поднято с 3 до 6: облачная модель (gpt-oss-120b через Groq)
    на длинных "расскажи о себе" сообщениях иногда вызывает remember_preference
    ПО ОДНОМУ факту за раунд вместо пакета за один раз — при 3 раундах и
    богатом сообщении (десяток фактов) лимит исчерпывался раньше, чем модель
    успевала дать финальный текстовый ответ. remember_preference теперь сам
    принимает список фактов за один вызов (см. tools.py), что должно резко
    сократить число раундов на практике — max_rounds=6 остаётся как запас.

    2026-07-08: добавлен отдельный, более узкий счётчик именно для
    search_web (см. MAX_SEARCH_CALLS_PER_TURN) — max_rounds сам по себе не
    спасал от того, что модель тратила ВСЕ раунды подряд на переформулировку
    одного и того же безуспешного поиска, ничего не давая взамен, и
    упиралась в TPM-лимит облака раньше, чем добиралась до финального ответа.
    """
    create_extra = create_extra or {}  # доп. параметры create(): max_tokens и т.п.
    search_calls_used = 0

    for _ in range(max_rounds):
        if chat_id:
            summary = "; ".join(_describe_tool_call(tc) for tc in tool_calls)
            await _publish_status(chat_id, trace_id, summary)

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
            if name == "search_web":
                search_calls_used += 1
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
        #
        # 2026-07-08, финальное решение по «Tool choice is none»: когда лимит
        # поисковых попыток исчерпан, НЕЛЬЗЯ ни оставлять search_web в tools=,
        # ни пытаться «запретить» его через tool_choice="none" или пустой
        # список — Groq/gpt-oss в ответ жёстко отдаёт 400 ('Tool choice is
        # none, but model called a tool' / 'tool not in request.tools'),
        # потому что модель по инерции своей же истории всё равно генерирует
        # вызов инструмента, а сервер это отвергает. Единственный надёжный
        # способ — ВООБЩЕ не передавать параметр tools: без схемы инструментов
        # модели нечего вызывать и нечему конфликтовать, и она отвечает
        # обычным текстом. Плюс один раз явно сообщаем ей, что поиск исчерпан.
        # (Прежние «ревизии 1-3» с available_tools/tool_choice="none" удалены —
        # именно они и были источником самого 400, а не защитой от него.)
        search_capped = search_calls_used >= MAX_SEARCH_CALLS_PER_TURN

        if search_capped:
            messages.append({
                "role": "system",
                "content": (
                    "Достигнут лимит попыток веб-поиска в этом запросе. "
                    "Больше НЕ вызывай инструменты. Ответь пользователю тем, "
                    "что уже известно из результатов выше, или честно скажи, "
                    "что не удалось найти актуальную информацию."
                ),
            })
            try:
                # tools НАМЕРЕННО не передаём (см. комментарий выше) — это и
                # снимает 400 'Tool choice is none, but model called a tool'.
                final = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    timeout=timeout,
                    **create_extra,
                )
            except Exception as api_err:
                logging.warning(f"[Kernel] Финальный запрос без инструментов не удался: {api_err}")
                return (
                    "Не получилось найти свежую информацию по этому вопросу "
                    "после нескольких попыток поиска — попробуй переформулировать "
                    "запрос или спросить позже."
                )
            return _extract_final_reply(final, context_label=f"tools-capped:{model_name}")

        # Лимит не исчерпан — обычный раунд, tools= передаём (иначе модель,
        # которая захочет вызвать инструмент ещё раз, упрётся в ошибку).
        response = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=TOOLS_MANIFEST,
            timeout=timeout,
            **create_extra,
        )

        if response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            continue  # модель просит ещё один раунд

        return _extract_final_reply(response, context_label=f"tools:{model_name}")

    # Лимит раундов исчерпан, а модель всё ещё пытается вызывать инструменты —
    # не зависаем и не падаем, отдаём понятное сообщение вместо ошибки. Часть
    # вызовов (например, сохранение фактов в память) к этому моменту уже могла
    # успешно выполниться — сама фраза ниже нейтральна и не про поиск конкретно,
    # т.к. до лимита раундов может довести любой инструмент, не только поиск.
    return "Не получилось собрать окончательный ответ за разумное число шагов — попробуй переформулировать запрос или отправить его покороче."


async def _run_cloud_fallback(
    cloud_messages: list,
    user_text: str,
    chat_id: str,
    user_id: Optional[int],
    trace_id: Optional[str],
    reuse_tool_calls=None,
):
    """Единая точка облачного пути. Вызывается в двух случаях: (1) локальная
    модель вообще не ответила (таймаут/ошибка), (2) локальная модель ответила
    и запросила инструмент, но саму оркестровку (выполнение + возможные
    доп. раунды + связный финальный ответ) сразу отдаём облаку, не давая
    локалке тянуть это самой.

    2026-07-08 (после включения надёжного Tavily): оркестровку инструментов
    теперь СНАЧАЛА делает локальная модель на GPU (см. call_model_router) — это
    Local First (§4.2) и §17 (если локаль справляется, облако не трогаем). Сюда,
    в облако, попадаем в двух случаях: (1) локальная модель вообще не ответила
    (таймаут/ошибка ещё до инструментов); (2) локальная оркестровка инструментов
    упала и её эскалировали. Прежний режим «любой инструмент сразу в облако»
    убран: он гонял облако даже на простом поиске и утыкался в TPM-лимит Groq.

    reuse_tool_calls: если передан — решение "что вызвать" уже принято локальной
    моделью, и облако не решает ту же задачу с нуля. В облако передаётся
    канонический `messages` ([system, история, user]) — отдельный от
    local_messages список, НЕ затронутый локальными tool-раундами, поэтому
    рассинхрона истории нет.
    """
    decision_label = "answered_hybrid_tool_handoff" if reuse_tool_calls else "answered_cloud_fallback"
    try:
        logging.info(f"Using Cloud Model: {FALLBACK_MODEL}")
        await _publish_status(chat_id, trace_id, f"Подключаю облачную модель ({FALLBACK_MODEL})…")

        if reuse_tool_calls:
            ai_reply = await handle_tool_calls(
                cloud_client, FALLBACK_MODEL, cloud_messages, reuse_tool_calls,
                user_id=user_id, trace_id=trace_id, chat_id=chat_id, timeout=20.0,
                create_extra={"max_tokens": CLOUD_MAX_TOKENS},
            )
        else:
            response = await cloud_client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=cloud_messages,
                tools=TOOLS_MANIFEST,  # Передаем плагины и в облако тоже!
                max_tokens=CLOUD_MAX_TOKENS,
            )

            if response.choices[0].message.tool_calls:
                logging.info(f"[Kernel] Облако запросило вызов инструментов: {response.choices[0].message.tool_calls}")
                ai_reply = await handle_tool_calls(
                    cloud_client, FALLBACK_MODEL, cloud_messages, response.choices[0].message.tool_calls,
                    user_id=user_id, trace_id=trace_id, chat_id=chat_id, timeout=20.0,
                    create_extra={"max_tokens": CLOUD_MAX_TOKENS},
                )
            else:
                ai_reply = _extract_final_reply(response, context_label=f"cloud:{FALLBACK_MODEL}")

        await save_chat_message(chat_id, "user", user_text)
        await save_chat_message(chat_id, "assistant", ai_reply)
        await memory.log_decision(trace_id, user_id, decision_label, model_used=FALLBACK_MODEL)
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


async def call_model_router(
    user_text: str,
    chat_id: str,
    telegram_user_id: Optional[int] = None,
    display_name: Optional[str] = None,
    trace_id: Optional[str] = None,
) -> str:
    current_date = datetime.now().strftime("%d %B %Y (день недели: %A)")

    await _publish_status(chat_id, trace_id, "Обрабатываю запрос…")

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
        "ВАЖНОЕ ПРАВИЛО ПАМЯТИ: remember_preference принимает СПИСОК фактов — "
        "если пользователь сообщает о себе МНОГО фактов за раз (представляется, "
        "описывает увлечения, просит что-то запомнить), сохрани самые чёткие "
        "структурированные факты (имя, профессия/статус, город, язык) ОДНИМ "
        "вызовом remember_preference, перечислив их все в facts, а НЕ "
        "отдельными вызовами по одному факту. Всё остальное (увлечения с "
        "деталями, привычки, контекст), что не сводится к паре "
        "категория/ключ/значение, сохрани ОДНИМ вызовом add_memory_note в виде "
        "краткого связного резюме. На одно сообщение обычно достаточно 1 "
        "вызова remember_preference + 1 вызова add_memory_note — и обязательно "
        "дай обычный текстовый ответ пользователю после этого, не обрывай "
        "диалог одними вызовами инструментов. "
        "ВАЖНОЕ ПРАВИЛО ПОИСКА: Если пользователь ищет информацию о мировых событиях, спорте, IT или новостях, "
        "ФОРМУЛИРУЙ ПОИСКОВЫЙ ЗАПРОС (query) НА АНГЛИЙСКОМ ЯЗЫКЕ (например, 'FIFA World Cup 2026 matches July 7'). "
        "Англоязычный поиск выдает более точные данные. "
        "Получив результаты, переведи их и ответь пользователю на чистом русском языке, не выдумывая фактов. "
        "Если после 1-2 попыток поиск не дал результата или не отвечает на "
        "вопрос — НЕ переформулируй запрос бесконечно и не ищи заново по кругу. "
        "Прямо скажи, что не нашёл актуальной информации, и ответь тем, что "
        "знаешь, или предложи уточнить вопрос. "
        "ВАЖНОЕ ПРАВИЛО ТОЧНОСТИ: если вопрос касается конкретных фактов о "
        "реальных людях, организациях или событиях (биография, профессия, "
        "рост/возраст/статистика, кто чем занимается, сравнения людей между "
        "собой) — НЕ угадывай и не выдумывай правдоподобно звучащие детали. "
        "Если не уверен на 100% — вызови search_web, чтобы проверить, а не "
        "отвечай уверенным тоном непроверенные подробности. Если поиск "
        "недоступен или не помог — честно скажи, что не уверен в деталях, "
        "вместо того чтобы выдавать вымысел за факт. "
        "ВАЖНОЕ ПРАВИЛО ФОРМАТА: это чат в Telegram (в будущем — ещё и голосовой "
        "интерфейс), а не документ. НЕ используй markdown-разметку: никаких **, "
        "##-заголовков, таблиц с |, вложенных списков. Пиши обычным связным "
        "текстом, как будто говоришь вслух — короткими абзацами. Несколько "
        "вариантов перечисляй прямо в тексте («во-первых… во-вторых…» или через "
        "запятую), а не списком или таблицей.\n"
        f"{profile_blurb}"
    )

    history = await get_chat_history(chat_id)
    messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_text}]
    # Локальная модель получает тот же system_prompt плюс отдельное указание
    # быть краткой (см. LOCAL_BREVITY_SUFFIX) — это НЕ просачивается в облако:
    # если дело дойдёт до _run_cloud_fallback, туда уходит канонический
    # `messages` без суффикса, а не local_messages.
    local_messages = [{"role": "system", "content": system_prompt + LOCAL_BREVITY_SUFFIX}] + history + [{"role": "user", "content": user_text}]

    # Попытка 1: Локальная модель — быстрый путь для простых ответов без инструментов.
    try:
        logging.info(f"Using Primary Local Model: {PRIMARY_MODEL}")
        await _publish_status(chat_id, trace_id, f"Думаю (локальная модель {PRIMARY_MODEL})…")
        response = await local_client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=local_messages,
            tools=TOOLS_MANIFEST,  # Передаем описание наших навыков
            timeout=LOCAL_TIMEOUT,
            max_tokens=LOCAL_MAX_TOKENS,
            # Qwen3 — гибридная "thinking"-модель: по умолчанию перед ЛЮБЫМ
            # ответом (даже простым решением "вызвать ли инструмент") генерит
            # видимую цепочку рассуждений, которая ничего не даёт для такой
            # простой задачи, но заметно удлиняет ответ и ест бюджет
            # LOCAL_MAX_TOKENS ещё до видимого ответа.
            # 2026-07-08, ревизия 2: chat_template_kwargs.enable_thinking — это
            # параметр НАТИВНОГО шаблона Qwen3 (как его понимают vLLM/
            # transformers), а OpenAI-совместимый эндпоинт Ollama на него
            # вообще не смотрит — то есть эта строка, скорее всего, никогда не
            # работала. Официально задокументированный способ (проверено по
            # исходникам Ollama: github.com/ollama/ollama/issues/14820 и
            # docs.ollama.com/api/openai-compatibility) — поле reasoning_effort:
            # "none". Оставляем оба варианта одновременно: reasoning_effort как
            # реально рабочий механизм, chat_template_kwargs — как безвредный
            # запасной на случай других провайдеров, которые всё же его учитывают.
            extra_body={
                "reasoning_effort": "none",
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )

        if response.choices[0].message.tool_calls:
            local_tool_calls = response.choices[0].message.tool_calls
            # 2026-07-08 (после включения надёжного Tavily): раньше ЛЮБОЙ вызов
            # инструмента немедленно уходил в облако — из-за этого даже простой
            # поиск утыкался в TPM-лимит бесплатного Groq (реальный инцидент:
            # 413 на 18924 токенах), а локальная GPU-модель, у которой лимитов
            # по токенам нет, простаивала. Теперь оркестровку (выполнить
            # инструмент + связно ответить) СНАЧАЛА делает локальная модель —
            # это и есть Local First (§4.2) и §17 (если локаль справляется,
            # облако не трогаем). Прежняя причина отдавать всё облаку —
            # «локалка не тянет многораундовые инструменты» — во многом была
            # следствием сломанного поиска (бесконечные пустые ретраи →
            # таймаут); с рабочим поиском модель получает результат с первой
            # попытки. Облако остаётся честным фолбэком: туда уходим ТОЛЬКО
            # если локальная оркестровка упала (таймаут/сбой).
            logging.info(
                f"[Kernel] Локальная модель запросила инструмент(ы) "
                f"{[tc.function.name for tc in local_tool_calls]} — выполняю ЛОКАЛЬНО "
                f"(облако только при сбое)"
            )
            try:
                ai_reply = await handle_tool_calls(
                    local_client, PRIMARY_MODEL, local_messages, local_tool_calls,
                    user_id=user_id, trace_id=trace_id, chat_id=chat_id,
                    timeout=LOCAL_TIMEOUT,
                    create_extra={
                        "max_tokens": LOCAL_MAX_TOKENS,
                        "extra_body": {"reasoning_effort": "none"},
                    },
                )
                await save_chat_message(chat_id, "user", user_text)
                await save_chat_message(chat_id, "assistant", ai_reply)
                await memory.log_decision(trace_id, user_id, "answered_local_tools", model_used=PRIMARY_MODEL)
                return ai_reply
            except Exception as local_tool_err:
                logging.warning(
                    f"[Kernel] Локальная оркестровка инструментов не удалась "
                    f"({local_tool_err}) — эскалирую в облако ({FALLBACK_MODEL})"
                )
                return await _run_cloud_fallback(
                    messages, user_text, chat_id, user_id, trace_id,
                    reuse_tool_calls=local_tool_calls,
                )

        ai_reply = _extract_final_reply(response, context_label=f"local:{PRIMARY_MODEL}")
        await save_chat_message(chat_id, "user", user_text)
        await save_chat_message(chat_id, "assistant", ai_reply)
        await memory.log_decision(trace_id, user_id, "answered_local", model_used=PRIMARY_MODEL)
        return ai_reply

    except Exception as e:
        logging.warning(f"Local model skipped, slow or failed: {e}. Routing to Cloud API...")
        return await _run_cloud_fallback(messages, user_text, chat_id, user_id, trace_id)

async def keep_local_model_warm():
    """Держит локальную модель постоянно загруженной в Ollama.

    Ollama по умолчанию выгружает модель из памяти после ~5 минут простоя
    (см. вывод `ollama ps`, колонка UNTIL) — а следующий запрос после этого
    платит временем холодного старта (загрузка весов в VRAM) ПОВЕРХ обычной
    генерации, что легко выбивает даже LOCAL_TIMEOUT на маленькой модели и
    на простом сообщении без поиска. Пингуем чаще, чем таймаут простоя,
    чтобы модель не успевала выгружаться в нормальном режиме работы бота.
    """
    while True:
        try:
            await local_client.chat.completions.create(
                model=PRIMARY_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                timeout=30.0,  # прогрев в фоне, никто его не ждёт — можно не спешить
                extra_body={"reasoning_effort": "none"},  # не тратим GPU-время на thinking ради пустого пинга
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

            try:
                ai_response = await asyncio.wait_for(
                    call_model_router(
                        user_text, str(chat_id),
                        telegram_user_id=telegram_user_id,
                        display_name=display_name,
                        trace_id=str(event.trace_id),
                    ),
                    timeout=PROCESS_HARD_DEADLINE,
                )
            except asyncio.TimeoutError:
                logging.error(
                    f"[Kernel] Обработка сообщения из чата {chat_id} превысила "
                    f"{PROCESS_HARD_DEADLINE}с — прерываю принудительно, чтобы не "
                    "заблокировать ядро для остальных сообщений"
                )
                await memory.log_decision(
                    str(event.trace_id), None, "hard_deadline_exceeded",
                    details={"timeout": PROCESS_HARD_DEADLINE, "chat_id": chat_id},
                )
                ai_response = (
                    "Извини, обработка запроса зависла и была прервана "
                    "(превышен предельный таймаут). Попробуй ещё раз."
                )

            command_event = BaseEvent(
                trace_id=event.trace_id,
                source="ai_kernel",
                type=EventType.COMMAND_REQUESTED,
                payload={
                    # ВАЖНО: str(chat_id) — тот же тип, что уходит в STATUS_UPDATE
                    # события (call_model_router получает chat_id уже строкой).
                    # Раньше здесь был "голый" chat_id (int), а в статусах —
                    # строка; interaction layer держит их в одном dict по
                    # chat_id как ключу, и int("123") != str("123") в Python —
                    # из-за этого статусное сообщение прошлого запроса никогда
                    # не находилось для очистки и следующий запрос по ошибке
                    # редактировал чужое, "зависшее" сообщение.
                    "chat_id": str(chat_id),
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
            # Раньше здесь было пустое `pass` — если Redis моргнёт (перезапуск
            # контейнера, сетевой сбой, ещё не поднялся при старте), ядро
            # молча уходило в бесконечный быстрый цикл ничего-не-делания БЕЗ
            # единой строчки в логах: процесс выглядит живым (docker ps
            # покажет running), а по факту сообщения не обрабатываются, и
            # никакого сигнала об этом нет — тот же класс проблемы, что и
            # молчаливо зависавший pool.acquire() (см. POOL_ACQUIRE_TIMEOUT в
            # memory.py), только теперь на уровне всего event-loop'а ядра.
            # Логируем и делаем паузу подольше, чтобы не заспамить лог одной
            # и той же ошибкой 10 раз в секунду, пока Redis не поднимется.
            logging.error(f"[Kernel] Ошибка в главном цикле чтения stream:ingress: {e}")
            await asyncio.sleep(2.0)
            continue
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())
