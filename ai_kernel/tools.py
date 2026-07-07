import asyncio
import logging
import random
import time
from typing import Dict, List, Optional

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
MAX_RETRIES = 3
HTTP_TIMEOUT = 10  # сек на один HTTP-запрос движка


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
                # backend="auto" — метапоиск по всем движкам в случайном порядке
                results = list(
                    ddgs.text(query, max_results=MAX_RESULTS, backend="auto")
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


# --- JSON-описание инструмента для модели (стандарт OpenAI / Ollama) --------
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
    }
]