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
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass(slots=True)
class UserModelConfig:
    provider: Provider = Provider.POLLINATIONS
    model: str = "openai"
    api_key: str = ""


@dataclass(slots=True)
class UserPterodactylConfig:
    base_url: str = ""
    api_key: str = ""


@dataclass(slots=True)
class ConversationMessage:
    role: str
    content: str


@dataclass(slots=True)
class AgentTaskItem:
    title: str
    status: str = "pending"


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
    model: str = ""
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PollinationsModelInfo:
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    context_length: int | None = None
    paid_only: bool = False
    tools: bool = False


@dataclass(slots=True)
class ProviderModelInfo:
    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    context_length: int | None = None


@dataclass(slots=True)
class TaskRecord:
    id: int
    user_id: int
    title: str
    status: TaskStatus
    related_files: list[str]
    summary: str = ""
    plan: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    messages: list[ConversationMessage] = field(default_factory=list)
    task_items: list[AgentTaskItem] = field(default_factory=list)
    model: str = ""
    context_length: int | None = None
    compression_count: int = 0
    created_at: int = 0
    updated_at: int = 0


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
