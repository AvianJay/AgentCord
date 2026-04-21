from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from agentcord.models import Provider

DEFAULT_MODEL_RATES: dict[str, float] = {
    "pollinations:*": 0.02,
    "pollinations:openai": 0.02,
    "pollinations:qwen-coder": 0.025,
    "pollinations:gemini-search": 0.03,
    "openai:*": 0.08,
    "anthropic:*": 0.08,
    "google:*": 0.06,
    "xai:*": 0.07,
    "custom:*": 0.1,
}


@dataclass(slots=True)
class Settings:
    discord_token: str
    discord_application_id: int | None
    bot_owner_id: int | None
    data_dir: Path
    workspace_limit_bytes: int
    default_credits: float
    default_pollinations_model: str
    pollinations_api_key: str
    custom_provider_base_url: str
    agent_max_iterations: int
    agent_max_actions_per_iteration: int
    credit_reserve_output_tokens: int
    model_rates: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        rates = dict(DEFAULT_MODEL_RATES)
        raw_rates = os.getenv("AGENTCORD_MODEL_RATES_JSON", "").strip()
        if raw_rates:
            rates.update(json.loads(raw_rates))

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            discord_application_id=_optional_int(os.getenv("DISCORD_APPLICATION_ID")),
            bot_owner_id=_optional_int(os.getenv("BOT_OWNER_ID")),
            data_dir=Path(os.getenv("AGENTCORD_DATA_DIR", "data")).resolve(),
            workspace_limit_bytes=int(os.getenv("AGENTCORD_WORKSPACE_LIMIT_BYTES", str(5 * 1024 * 1024))),
            default_credits=float(os.getenv("AGENTCORD_DEFAULT_CREDITS", "100")),
            default_pollinations_model=os.getenv("AGENTCORD_DEFAULT_MODEL", "openai").strip() or "openai",
            pollinations_api_key=os.getenv("POLLINATIONS_API_KEY", "").strip(),
            custom_provider_base_url=os.getenv("AGENTCORD_CUSTOM_PROVIDER_BASE_URL", "").strip(),
            agent_max_iterations=max(1, int(os.getenv("AGENTCORD_AGENT_MAX_ITERATIONS", "6"))),
            agent_max_actions_per_iteration=max(1, int(os.getenv("AGENTCORD_AGENT_MAX_ACTIONS", "8"))),
            credit_reserve_output_tokens=max(128, int(os.getenv("AGENTCORD_CREDIT_RESERVE_OUTPUT_TOKENS", "1024"))),
            model_rates=rates,
        )

    def get_model_rate(self, provider: Provider, model: str) -> float:
        exact_key = f"{provider.value}:{model}"
        if exact_key in self.model_rates:
            return self.model_rates[exact_key]
        return self.model_rates.get(f"{provider.value}:*", 0.05)


def _optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)
