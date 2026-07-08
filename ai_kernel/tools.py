import asyncio
import logging
import os
import random
import time
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from memory import set_preference as _set_preference, add_memory_entry as _add_memory_entry

# --- Поисковые движки ------------------------------------------------------
# ddgs остаётся как БЕСПЛАТНЫЙ фолбэк (не требует ключа). Пакет
# `duckduckgo_search` заморожен в июле 2025 и переименован в `ddgs`; старое имя
# больше не получает фиксы анти-бот защиты и со временем ломается. Но сам ddgs —
# это неофициальный скрейпинг выдачи Google/Brave/DDG, и он В ПРИНЦИПЕ
# периодически спотыкается о капчи (202), rate-limit (429) и смену вёрстки
# (200, но «No results found»). Именно это давало «поиск работает через раз».
# Поэтому основным движком теперь является официальный HTTP-API (Tavily,
# см. _search_tavily), а ddgs — только аварийный запасной путь.
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
MAX_RETRIES = 2          # попыток внутри ddgs-движка (реальные сетевые сбои)
HTTP_TIMEOUT = 10        # сек на один HTTP-запрос движка

# Провайдер поиска — абстракция (спецификация §4.8 Replaceable, §21 unified API):
# движок переключается через .env БЕЗ правки кода. Значения:
#   "auto"   — Tavily, если задан TAVILY_API_KEY, иначе ddgs (по умолчанию);
#   "tavily" — Tavily основным + ddgs как аварийный фолбэк;
#   "ddgs"   — только старый скрейпинг-движок (совсем без внешнего ключа).
# Добавить Brave/SearXNG позже — это ещё одна функция-провайдер и одна строка в
# _provider_chain(), не трогая ни kernel.py, ни контракт search_web().
SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()

# Tavily — поисковый API, спроектированный под LLM-агентов: отдаёт готовые
# ранжированные сниппеты (а не сырой HTML, который нужно парсить), и делает это
# по официальному эндпоинту с ключом — без капч и произвольной смены вёрстки.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_URL = "https://api.tavily.com/search"

# Движки для ddgs-фолбэка. backend="auto" перебирал бы до 5-6 движков за вызов,
# включая нестабильные (startpage — капча, grokipedia — 502) — это и была одна
# из причин долгого/пустого поиска. Держим короткий список; меняется через .env.
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "google,brave,duckduckgo")


class SearchRateLimited(Exception):
    """Единый для всех провайдеров тип «нас ограничили по частоте» (429).

    Нужен, чтобы search_web мог отличить rate-limit (есть смысл сообщить
    «попробуй через минуту») от обычной ошибки движка и единообразно уйти в
    следующий провайдер цепочки, не зная деталей конкретного SDK.
    """


# --- Провайдер: Tavily (основной) -----------------------------------------
def _parse_tavily(payload: dict) -> List[Dict[str, str]]:
    """Нормализует ответ Tavily к общему виду {title, href, body}.

    Вынесено отдельно от HTTP-вызова, чтобы парсинг можно было проверить
    юнит-тестом без сети (в песочнице внешней сети нет).
    """
    out: List[Dict[str, str]] = []
    for r in (payload or {}).get("results", []) or []:
        out.append({
            "title": r.get("title") or "Без заголовка",
            "href": r.get("url") or "—",
            "body": r.get("content") or "Нет описания",
        })
    return out


async def _search_tavily(query: str) -> List[Dict[str, str]]:
    """Поиск через Tavily API (официальный HTTP-эндпоинт, не скрейпинг).

    httpx уже в зависимостях (его тянет openai) — отдельный SDK не нужен.
    Бросает SearchRateLimited на 429 и обычное исключение на прочих ошибках;
    решение «падать или в фолбэк» принимает search_web.
    """
    if not TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY не задан — Tavily недоступен")

    body = {
        "query": query,
        "max_results": MAX_RESULTS,
        "search_depth": "basic",
        "include_answer": False,
    }
    headers = {"Authorization": f"Bearer {TAVILY_API_KEY}"}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(TAVILY_URL, json=body, headers=headers)

    if resp.status_code == 429:
        raise SearchRateLimited("Tavily 429")
    resp.raise_for_status()
    return _parse_tavily(resp.json())


# --- Провайдер: ddgs (аварийный фолбэк) -----------------------------------
def _run_ddgs_sync(query: str) -> List[Dict[str, str]]:
    """Блокирующий поиск ddgs с ретраями и экспоненциальным backoff.

    Выполняется в пуле потоков (см. _search_ddgs), поэтому time.sleep здесь
    не блокирует event loop ядра. Если все попытки провалились — пробрасывает
    последнее исключение наверх (RatelimitException конвертируется в
    SearchRateLimited уже в _search_ddgs).
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
            logger.info("[search] ddgs: пустая выдача (попытка %s/%s)", attempt, MAX_RETRIES)
        except RatelimitException as e:
            last_error = e
            wait = 2 ** attempt + random.uniform(0, 1)  # ~2..9 сек + джиттер
            logger.warning(
                "[search] ddgs rate limit (попытка %s/%s). Пауза %.1f с.",
                attempt, MAX_RETRIES, wait,
            )
            time.sleep(wait)
        except (TimeoutException, DDGSException) as e:
            last_error = e
            logger.warning("[search] ddgs ошибка (попытка %s/%s): %s", attempt, MAX_RETRIES, e)
            time.sleep(1)
        except Exception as e:  # сеть, DNS и прочее непредвиденное
            last_error = e
            logger.error("[search] ddgs непредвиденная ошибка (попытка %s/%s): %s", attempt, MAX_RETRIES, e)
            time.sleep(1)

    if last_error is not None:
        raise last_error
    return []


async def _search_ddgs(query: str) -> List[Dict[str, str]]:
    """Асинхронная обёртка ddgs: уводит блокирующий вызов в пул потоков и
    нормализует выдачу к общему виду {title, href, body}."""
    loop = asyncio.get_running_loop()
    try:
        raw = await loop.run_in_executor(None, _run_ddgs_sync, query)
    except RatelimitException as e:
        raise SearchRateLimited(str(e))
    return [
        {
            "title": r.get("title") or "Без заголовка",
            "href": r.get("href") or r.get("link") or "—",
            "body": r.get("body") or "Нет описания",
        }
        for r in raw
    ]


# --- Выбор цепочки провайдеров --------------------------------------------
def _provider_chain() -> List[Tuple[str, Callable]]:
    """Возвращает упорядоченный список (имя, функция-провайдер).

    Первый провайдер — основной, остальные — фолбэки, к которым search_web
    переходит по очереди при ошибке/пустой выдаче предыдущего.
    """
    if SEARCH_PROVIDER == "ddgs":
        return [("ddgs", _search_ddgs)]
    if SEARCH_PROVIDER == "tavily":
        return [("tavily", _search_tavily), ("ddgs", _search_ddgs)]
    # "auto" (по умолчанию): Tavily, если есть ключ, иначе только ddgs —
    # система остаётся рабочей ещё ДО того, как пользователь заведёт ключ.
    if TAVILY_API_KEY:
        return [("tavily", _search_tavily), ("ddgs", _search_ddgs)]
    return [("ddgs", _search_ddgs)]


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
    """Асинхронный веб-поиск ядра AIOS с цепочкой провайдеров.

    Идёт по _provider_chain() по очереди: первый провайдер, вернувший
    непустой результат, выигрывает; ошибка/пусто — переход к следующему.
    Возвращает готовый текст с результатами или понятное сообщение об ошибке.
    Никогда не возвращает None — контракт с kernel.handle_tool_calls сохранён.
    """
    if not query or not query.strip():
        return "Пустой поисковый запрос."

    query = query.strip()
    logger.info("[search] Запрос: %r", query)

    chain = _provider_chain()
    rate_limited = False
    last_error: Optional[Exception] = None

    for name, fn in chain:
        try:
            results = await fn(query)
        except SearchRateLimited as e:
            rate_limited = True
            last_error = e
            logger.warning("[search] %s: rate limit — перехожу к следующему движку", name)
            continue
        except Exception as e:
            last_error = e
            logger.warning("[search] %s: ошибка (%s) — перехожу к следующему движку", name, e)
            continue

        if results:
            logger.info("[search] %s: %d результат(ов)", name, len(results))
            return _format_results(results)
        logger.info("[search] %s: пустая выдача", name)

    if rate_limited:
        return (
            "Поиск временно недоступен: движки ограничили частоту запросов "
            "(rate limit). Попробуй ещё раз через минуту."
        )
    if last_error is not None:
        logger.error("[search] Поиск не выполнен (последняя ошибка: %s)", last_error)
    return "По этому запросу ничего не найдено. Уточни формулировку."


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
