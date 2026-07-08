import asyncio
import logging
import os
import random
import time
from typing import Dict, List, Optional

from memory import set_preference as _set_preference, add_memory_entry as _add_memory_entry

# --- Импорт поискового движка ---------------------------------------------
# Пакет `duckduckgo_search` заморожен в июле 2025 и переименован в `ddgs`.
# Старое имя больше не получает фиксы анти-бот защиты DuckDuckGo и со временем
# ломается. `ddgs` — это метапоиск: он агрегирует выдачу Google, Bing,
# DuckDuckGo, Brave, Yahoo, Yandex и др. с автоматическим fallback между ними,
# что само по себе куда устойчивее к rate-limit, чем старый DuckDuckGo-only клиент.
try:
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
except ImportError:  # обратная совместимость, если стоит только старый пакет
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import (  # type: ignore
        DuckDuckGoSearchException as DDGSException,
        RatelimitException,
        TimeoutException,
    )

logger = logging.getLogger("aios.tools.search")

# --- Настройки -------------------------------------------------------------
MAX_RESULTS = 5
# Было 3: внутренний ретрай здесь умножается на внешний (модель сама может
# переформулировать запрос и вызвать search_web ещё раз в следующем раунде
# handle_tool_calls) — 3×несколько раундов на практике выливалось в добрый
# десяток запросов к Google/Brave за один вопрос пользователя и укладывало
# Brave в rate limit. 2 попытки внутри — достаточно для реальных сетевых
# сбоев, а не для маскировки того, что движок in principle не находит ответ.
MAX_RETRIES = 2
HTTP_TIMEOUT = 10  # сек на один HTTP-запрос движка

# backend="auto" перебирает до 5-6 движков за один вызов (wikipedia, grokipedia,
# google, yahoo, brave, startpage, ...), включая нестабильные (startpage часто
# отдаёт капчу, grokipedia иногда 502) — это и есть основная причина долгого
# поиска. Явно ограничиваем небольшим набором быстрых и обычно надёжных
# движков; при необходимости меняется через .env без правки кода.
# 2026-07-08: добавлен duckduckgo третьим движком. В реальном инциденте
# (вопрос о росте Галкина) Google 4/4 раза вернул 200, но ddgs не смог
# распарсить из ответа результаты ("No results found" — похоже на то, что
# Google отдал непривычную для парсера структуру страницы), а Brave все 8/8
# раз словил 429 (исчерпан лимит запросов). Раз оба сконфигурированных
# движка одновременно неработоспособны — это не разовая случайность, а
# показатель того, что двух движков мало для устойчивости; независимая третья
# инфраструктура снижает шанс одновременного отказа всех сразу. Это не
# устраняет саму природу проблемы (неофициальный скрейпинг вместо платного
# API всегда будет периодически спотыкаться о капчи/рейт-лимиты/смену
# вёрстки) — это смягчение, а не полное решение.
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "google,brave,duckduckgo")


def _run_search_sync(query: str) -> List[Dict[str, str]]:
    """Блокирующий поиск с ретраями и экспоненциальным backoff.

    Выполняется в отдельном потоке (см. search_web), поэтому здесь можно
    безопасно использовать time.sleep, не блокируя event loop ядра.
    Если все попытки провалились — пробрасывает последнее исключение наверх.
    """
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with DDGS(timeout=HTTP_TIMEOUT) as ddgs:
                results = list(
                    ddgs.text(query, max_results=MAX_RESULTS, backend=SEARCH_BACKEND)
                )
            if results:
                return results
            logger.info("[search] Пустая выдача (попытка %s/%s)", attempt, MAX_RETRIES)
        except RatelimitException as e:
            last_error = e
            wait = 2 ** attempt + random.uniform(0, 1)  # ~2..9 сек + джиттер
            logger.warning(
                "[search] Rate limit (попытка %s/%s). Пауза %.1f с.",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except (TimeoutException, DDGSException) as e:
            last_error = e
            logger.warning(
                "[search] Ошибка движка (попытка %s/%s): %s", attempt, MAX_RETRIES, e
            )
            time.sleep(1)
        except Exception as e:  # сеть, DNS и прочее непредвиденное
            last_error = e
            logger.error(
                "[search] Непредвиденная ошибка (попытка %s/%s): %s",
                attempt, MAX_RETRIES, e,
            )
            time.sleep(1)

    if last_error is not None:
        raise last_error
    return []


def _format_results(results: List[Dict[str, str]]) -> str:
    """Приводит выдачу к компактному тексту для передачи модели."""
    blocks = []
    for i, r in enumerate(results, start=1):
        title = r.get("title") or "Без заголовка"
        link = r.get("href") or r.get("link") or "—"
        body = r.get("body") or "Нет описания"
        blocks.append(f"[{i}] {title}\nИсточник: {link}\n{body}")
    return "\n---\n".join(blocks)


async def search_web(query: str) -> str:
    """Асинхронная обёртка веб-поиска для ядра AIOS.

    Возвращает готовый текст с результатами или понятное сообщение об ошибке.
    Никогда не возвращает None — контракт с kernel.handle_tool_calls сохранён.
    """
    if not query or not query.strip():
        return "Пустой поисковый запрос."

    logger.info("[search] Запрос: %r", query)
    loop = asyncio.get_running_loop()

    try:
        # DDGS синхронный и блокирующий — уводим его в пул потоков, чтобы
        # event loop ядра продолжал обслуживать другие события во время поиска.
        results = await loop.run_in_executor(None, _run_search_sync, query)
    except RatelimitException:
        return (
            "Поиск временно недоступен: поисковые движки ограничили частоту "
            "запросов (rate limit). Попробуй ещё раз через минуту."
        )
    except Exception as e:
        logger.error("[search] Поиск не выполнен: %s", e)
        return f"Не удалось выполнить поиск: {e}"

    if not results:
        return "По этому запросу ничего не найдено. Уточни формулировку."

    return _format_results(results)


async def remember_preference(user_id: Optional[int], facts: Optional[list]) -> str:
    """Инструмент для модели: сохранить ОДИН ИЛИ НЕСКОЛЬКО фактов/предпочтений
    о пользователе в Postgres ЗА ОДИН ВЫЗОВ (раздел 14.4 спецификации —
    Preference Memory), чтобы они были доступны в будущих разговорах.

    facts — список объектов {category, key, value}. Раньше инструмент принимал
    ровно один факт за вызов, и на длинных "расскажи о себе" сообщениях модели
    (особенно облачная, gpt-oss-120b через Groq) вызывали его ПО ОДНОМУ факту
    за раунд — при десятке фактов это упиралось в лимит раундов handle_tool_calls
    и в лимиты Groq раньше, чем модель успевала дать финальный ответ (см.
    критический разбор от 2026-07-08). Теперь форма вызова ровно одна — список,
    даже для одного факта — так у модели нет соблазна звать инструмент много
    раз подряд вместо одного пакетного вызова.

    Уровень разрешения — 0 (см. TOOL_PERMISSION_LEVELS в kernel.py): по
    спецификации (18.3) "записать заметку" — безвредное автоматическое
    действие, подтверждение не требуется.
    """
    if not user_id:
        return "Не удалось сохранить: пользователь не определён."
    if not facts or not isinstance(facts, list):
        return "Не удалось сохранить: нужен непустой список facts (category/key/value)."

    saved, errors = [], []
    for fact in facts:
        fact = fact or {}
        category, key, value = fact.get("category"), fact.get("key"), fact.get("value")
        if not category or not key or not value:
            errors.append(str(fact))
            continue
        # Модель с некорректно сформированным вызовом инструмента иногда
        # присылает не строку (например, число для value) — колонки в
        # Postgres TEXT, а asyncpg (в отличие от psycopg2) строго проверяет
        # соответствие типов и уронит весь вызов исключением на нестроковом
        # аргументе вместо понятного сообщения об ошибке.
        category, key, value = str(category), str(key), str(value)
        try:
            await _set_preference(user_id, category, key, value)
            saved.append(f"{category}/{key} = {value}")
        except Exception as e:
            logger.error("[remember] Не удалось сохранить %s/%s: %s", category, key, e)
            errors.append(f"{category}/{key} ({e})")

    if saved:
        logger.info("[remember] Сохранено для user_id=%s (%d шт.): %s", user_id, len(saved), "; ".join(saved))

    parts = []
    if saved:
        parts.append(f"Запомнил {len(saved)}: " + "; ".join(saved) + ".")
    if errors:
        parts.append(f"Не удалось сохранить {len(errors)}: " + "; ".join(errors) + ".")
    return " ".join(parts) if parts else "Не удалось сохранить: пустой список фактов."


async def add_memory_note(user_id: Optional[int], content: Optional[str], category: str = "episodic") -> str:
    """Инструмент для модели: сохранить свободную заметку о пользователе,
    которая плохо сводится к одной паре категория/ключ/значение (раздел 14.2
    спецификации — Episodic Memory).

    Дополняет remember_preference: тот — для чётких фактов вида ключ/значение,
    этот — для более развёрнутого контекста (интересы с деталями, привычки,
    история), который иначе терялся бы, если модель не смогла (или не
    попыталась) разложить его на пары ключ-значение. На практике маленькая
    локальная модель при длинном рассказе о себе часто сохраняет 1-2 самых
    очевидных факта через remember_preference и пропускает остальное — этот
    инструмент даёт ей более простой путь не потерять то, что не раскладывается
    аккуратно на categoty/key/value.

    Уровень разрешения — 0, как и у remember_preference: безвредная запись,
    подтверждение не требуется (см. TOOL_PERMISSION_LEVELS в kernel.py).
    """
    if not user_id:
        return "Не удалось сохранить: пользователь не определён."
    if not isinstance(content, str):
        # См. аналогичную защиту в remember_preference — модель иногда
        # присылает не строку, а .strip() ниже на не-строке уронит весь
        # раунд инструментов исключением вместо понятного ответа.
        content = str(content) if content is not None else ""
    if not content or not content.strip():
        return "Не удалось сохранить: пустая заметка."

    try:
        await _add_memory_entry(user_id, content.strip(), category=category or "episodic", source="model_noted")
        logger.info("[remember] Заметка сохранена для user_id=%s (%s): %s", user_id, category, content[:80])
        return "Заметка сохранена."
    except Exception as e:
        logger.error("[remember] Не удалось сохранить заметку: %s", e)
        return f"Не удалось сохранить заметку: {e}"


# --- JSON-описание инструментов для модели (стандарт OpenAI / Ollama) -------
TOOLS_MANIFEST = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Поиск свежей информации в интернете: новости, результаты "
                "событий, погода, факты, в которых ты не уверен на 100%. "
                "Запрос (query) формулируй кратко и по делу. Для мировых "
                "событий, спорта и IT точнее искать запросом на английском."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос, например: "
                            "'FIFA World Cup 2026 fixtures July 7'."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_preference",
            "description": (
                "Сохрани один или несколько чётких фактов/предпочтений о "
                "пользователе на будущее — профессия, город, диета, часовой "
                "пояс, язык, любимые бренды, аллергии и т.п. ВАЖНО: если "
                "фактов несколько (например, пользователь представляется и "
                "называет сразу много деталей о себе) — передай их ВСЕ СРАЗУ "
                "одним вызовом, перечислив в facts, а НЕ отдельными вызовами "
                "по одному факту. Один вызов с 5 фактами быстрее и надёжнее "
                "пяти вызовов по одному."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "description": "Список фактов для сохранения за один вызов (даже для одного факта — список из одного элемента).",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "description": "Категория факта, например 'food', 'communication_style', 'schedule', 'health'.",
                                },
                                "key": {
                                    "type": "string",
                                    "description": "Короткий ключ факта, например 'diet' или 'timezone'.",
                                },
                                "value": {
                                    "type": "string",
                                    "description": "Значение, например 'вегетарианец' или 'Europe/Moscow'.",
                                },
                            },
                            "required": ["category", "key", "value"],
                        },
                    },
                },
                "required": ["facts"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_memory_note",
            "description": (
                "Сохрани свободную заметку о пользователе, которая плохо "
                "сводится к одной паре категория/ключ/значение — например, "
                "увлечение с деталями, привычку, контекст из разговора. "
                "Используй ВМЕСТЕ с remember_preference: если пользователь "
                "делится сразу несколькими фактами о себе (например, "
                "представляется или описывает свои интересы), вызови "
                "remember_preference ОДИН раз со всеми чёткими фактами "
                "(профессия, город, язык и т.п.) И add_memory_note для "
                "остального контекста, который не хочется терять."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": (
                            "Текст заметки на русском, например: 'Смотрит "
                            "киберспорт CS2, tier 1 турниры (PGL, BLAST, ESL), "
                            "топ-10 команд, вместе с женой.'"
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": "Необязательная категория заметки, например 'interests', 'habits', 'background'. По умолчанию 'episodic'.",
                    },
                },
                "required": ["content"],
            },
        },
    },
]