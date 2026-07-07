import asyncio
import logging
from duckduckgo_search import DDGS

logging.basicConfig(level=logging.INFO)

async def search_web(query: str) -> str:
    """
    Выполняет анонимный поиск в интернете через DuckDuckGo и возвращает краткие результаты.
    """
    logging.info(f"[Tool] Запуск веб-поиска для запроса: '{query}'")
    try:
        loop = asyncio.get_running_loop()
        def sync_search():
            with DDGS() as ddgs:
                # Попробуем сначала оригинальный запрос
                results = list(ddgs.text(query, max_results=5))
                
                # Если запрос слишком точный и ничего не найдено, пробуем "календарь/расписание"
                if not results and "2026" in query:
                    logging.info("[Tool] Точный поиск пуст. Пробуем общий календарь ЧМ-2026...")
                    results = list(ddgs.text("Чемпионат мира по футболу 2026 расписание таблица матчей", max_results=5))
                
                if not results:
                    return "Ничего не найдено по этому запросу."
                
                formatted_results = []
                for r in results:
                    title = r.get('title', 'Без заголовка')
                    link = r.get('href', r.get('link', 'Ссылка отсутствует'))
                    body = r.get('body', 'Нет описания')
                    
                    formatted_results.append(
                        f"Заголовок: {title}\n"
                        f"Ссылка: {link}\n"
                        f"Контекст: {body}\n"
                    )
                return "\n---\n".join(formatted_results)
        
    except Exception as e:
        logging.error(f"[Tool] Ошибка при поиске в интернете: {e}")
        return f"Не удалось выполнить поиск из-за ошибки: {e}"

# JSON-описание инструмента для нейросети (стандарт OpenAI/Ollama)
TOOLS_MANIFEST = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Используй этот инструмент, когда пользователю нужна свежая, актуальная информация из интернета, "
                "новости, результаты событий, погода или факты, в которых ты не уверен на 100%. "
                "Запрос (query) должен быть четким, на языке пользователя, оптимизированным для поисковика."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (например, 'последние коллекции российских брендов косметики 2026').",
                    }
                },
                "required": ["query"],
            },
        },
    }
]
