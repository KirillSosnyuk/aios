from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict
import time

class EventSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

class EventType(str, Enum):
    # Раздел 12.2: Типы событий для MVP
    USER_MESSAGE_RECEIVED = "UserMessageReceived"
    COMMAND_REQUESTED = "CommandRequested"
    SYSTEM_TICK = "SystemTick"
    ERROR_OCCURRED = "ErrorOccurred"
    # Лёгкое статусное событие для "живого" индикатора хода обработки в
    # интерфейсе (раздел 4 — Explainability): "думаю локально", "ищу в
    # интернете" и т.п. Публикуется в тот же stream:egress, что и
    # COMMAND_REQUESTED — interaction layer различает их по полю type.
    STATUS_UPDATE = "StatusUpdate"

class BaseEvent(BaseModel):
    # Включаем поддержку работы с типами, если потребуется
    model_config = ConfigDict(use_enum_values=True)

    unique_id: UUID = Field(default_factory=uuid4, description="unique id события")
    trace_id: UUID = Field(default_factory=uuid4, description="trace id для отслеживания цепочки решений")
    timestamp: float = Field(default_factory=time.time, description="timestamp генерации")
    
    source: str = Field(..., description="Источник (например: 'telegram_bot', 'ai_kernel')")
    type: EventType = Field(..., description="Тип события")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Полезная нагрузка")
    
    severity: EventSeverity = Field(default=EventSeverity.INFO, description="Уровень критичности")
    confidence: float = Field(default=1.0, description="Уверенность системы в событии (от 0.0...1.0)")
    references: Optional[Dict[str, Any]] = Field(default=None, description="Ссылки на сущности доменной модели")
