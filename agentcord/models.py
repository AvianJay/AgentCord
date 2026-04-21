from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Provider(StrEnum):
    POLLINATIONS = "pollinations"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    CUSTOM = "custom"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"


@dataclass(slots=True)
class UserModelConfig:
    provider: Provider = Provider.POLLINATIONS
    model: str = "openai"
    api_key: str = ""


@dataclass(slots=True)
class AIUsage:
    input_tokens: int
    output_tokens: int
    cost: float
    model_rate: float


@dataclass(slots=True)
class AIResponse:
    content: str
    usage: AIUsage
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRecord:
    id: int
    title: str
    status: TaskStatus
    related_files: list[str]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
