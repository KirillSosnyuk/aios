"""
Слой памяти AIOS (раздел 14 спецификации): постоянное хранилище профиля
пользователя, предпочтений, заметок и журнала решений в Postgres
(раздел 22 — "Хранилище сущностей Модели Мира", которое было поднято в
docker-compose с самого начала проекта, но до сих пор ничем не использовалось).

Рассчитан на несколько пользователей с самого начала (сейчас 3 доверенных
telegram-аккаунта из ALLOWED_USERS, см. .env) и на то, что интерфейс не
всегда будет только текстовым в Telegram (планируются голос/колонка) —
поэтому личность пользователя (`users`) идентифицируется по telegram_id
только как ОДИН из способов связаться с ним, а все остальные таблицы
ссылаются на внутренний users.id, а не напрямую на telegram_id. Когда
появится второй канал (голос), для него будет достаточно добавить свою
колонку/таблицу связи с users.id, не трогая preferences/memory_entries/audit.
"""
import asyncio
import json
import logging
import os
from typing import Any, Optional

import asyncpg

logger = logging.getLogger("aios.memory")

DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER", "aios_admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "aios_secure_pass")
DB_NAME = os.getenv("DB_NAME", "aios_world_model")

_pool: Optional[asyncpg.Pool] = None

# Идентификаторы (SERIAL/id) вместо UUID для простоты на этом этапе проекта;
# trace_id храним как TEXT, а не как Postgres UUID — чтобы не зависеть от
# того, передали ли его строкой или объектом uuid.UUID на стороне вызова.
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    telegram_id BIGINT UNIQUE NOT NULL,
    display_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS preferences (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, category, key)
);

CREATE TABLE IF NOT EXISTS memory_entries (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    category TEXT,
    source TEXT,
    confidence REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_records (
    id SERIAL PRIMARY KEY,
    trace_id TEXT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    decision TEXT NOT NULL,
    model_used TEXT,
    permission_level SMALLINT,
    details JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_preferences_user ON preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_entries_user ON memory_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_records_user ON audit_records(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_records_trace ON audit_records(trace_id);
"""


async def init_pool(retries: int = 5, delay: float = 2.0) -> None:
    """Создаёт пул соединений и накатывает схему (идемпотентно — CREATE TABLE
    IF NOT EXISTS). Вызывается один раз при старте kernel.py.

    Ретраит подключение несколько раз: при `docker compose up` Postgres может
    ещё не принимать соединения, когда ai_kernel уже стартовал (depends_on
    в Compose гарантирует только порядок запуска контейнеров, не готовность).
    Если Postgres так и не поднялся — ядро должно продолжить работать без
    памяти, а не падать целиком.
    """
    global _pool
    if _pool is not None:
        return

    for attempt in range(1, retries + 1):
        try:
            _pool = await asyncpg.create_pool(
                host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
                database=DB_NAME, min_size=1, max_size=5,
            )
            async with _pool.acquire() as conn:
                await conn.execute(SCHEMA)
            logger.info("[Memory] Подключено к Postgres, схема готова")
            return
        except Exception as e:
            logger.warning(f"[Memory] Postgres пока недоступен (попытка {attempt}/{retries}): {e}")
            _pool = None
            await asyncio.sleep(delay)

    logger.error("[Memory] Не удалось подключиться к Postgres — ядро работает без памяти")


def _ready() -> bool:
    return _pool is not None


async def get_or_create_user(telegram_id: Optional[int], display_name: Optional[str] = None) -> Optional[int]:
    """Возвращает внутренний users.id для telegram_id, создавая запись при
    первом обращении и подтягивая имя, если раньше его не было.

    Возвращает None, если telegram_id не передан или БД недоступна —
    остальной код должен уметь работать и без памяти (деградация, не падение).
    """
    if not _ready() or not telegram_id:
        return None
    try:
        async with _pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (telegram_id, display_name)
                VALUES ($1, $2)
                ON CONFLICT (telegram_id) DO UPDATE
                    SET display_name = COALESCE(EXCLUDED.display_name, users.display_name)
                RETURNING id
                """,
                telegram_id, display_name,
            )
            return row["id"]
    except Exception as e:
        logger.warning(f"[Memory] get_or_create_user не удался: {e}")
        return None


async def set_preference(user_id: Optional[int], category: str, key: str, value: str) -> None:
    if not _ready() or not user_id:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO preferences (user_id, category, key, value)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, category, key)
                    DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                user_id, category, key, value,
            )
    except Exception as e:
        logger.warning(f"[Memory] set_preference не удался: {e}")


async def add_memory_entry(user_id: Optional[int], content: str, category: str = "episodic", source: str = "user_stated") -> None:
    if not _ready() or not user_id:
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memory_entries (user_id, content, category, source) VALUES ($1, $2, $3, $4)",
                user_id, content, category, source,
            )
    except Exception as e:
        logger.warning(f"[Memory] add_memory_entry не удался: {e}")


async def build_profile_blurb(user_id: Optional[int], max_memories: int = 8) -> str:
    """Собирает короткую сводку о пользователе для system_prompt: имя,
    предпочтения, последние заметки. Пустая строка, если памяти ещё нет или
    БД недоступна — промпт не должен раздуваться пустыми заголовками."""
    if not _ready() or not user_id:
        return ""

    try:
        async with _pool.acquire() as conn:
            user_row = await conn.fetchrow("SELECT display_name FROM users WHERE id = $1", user_id)
            prefs = await conn.fetch(
                "SELECT category, key, value FROM preferences WHERE user_id = $1 ORDER BY updated_at DESC LIMIT 20",
                user_id,
            )
            memories = await conn.fetch(
                "SELECT content FROM memory_entries WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
                user_id, max_memories,
            )
    except Exception as e:
        logger.warning(f"[Memory] build_profile_blurb не удался: {e}")
        return ""

    parts = []
    if user_row and user_row["display_name"]:
        parts.append(f"Имя пользователя: {user_row['display_name']}.")
    if prefs:
        pref_text = "; ".join(f"{p['category']}/{p['key']}: {p['value']}" for p in prefs)
        parts.append(f"Известные предпочтения: {pref_text}.")
    if memories:
        mem_text = "; ".join(m["content"] for m in memories)
        parts.append(f"Заметки о пользователе из прошлых разговоров: {mem_text}.")

    if not parts:
        return ""
    return "Память о пользователе (используй как контекст, но не выдумывай сверх этого): " + " ".join(parts) + "\n"


async def log_decision(
    trace_id: Optional[str],
    user_id: Optional[int],
    decision: str,
    model_used: Optional[str] = None,
    permission_level: Optional[int] = None,
    details: Optional[dict] = None,
) -> None:
    """Пишет запись в audit_records (раздел 14.8/23 — журнал решений).

    Никогда не бросает исключение наружу: аудит — вспомогательная функция,
    и не должен быть способен уронить обработку сообщения пользователя.
    """
    if not _ready():
        return
    try:
        async with _pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO audit_records (trace_id, user_id, decision, model_used, permission_level, details)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                """,
                trace_id, user_id, decision, model_used, permission_level,
                json.dumps(details or {}),
            )
    except Exception as e:
        logger.warning(f"[Memory] log_decision не удался: {e}")
