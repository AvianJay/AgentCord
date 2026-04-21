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
    discord_log_webhook: str
    data_dir: Path
    workspace_limit_bytes: int
    default_credits: float
    default_pollinations_model: str
    pollinations_api_key: str
    custom_provider_base_url: str
    proxy_url: str
    proxy_username: str
    proxy_password: str
    agent_max_iterations: int
    agent_max_actions_per_iteration: int
    credit_reserve_output_tokens: int
    model_rates: dict[str, float] = field(default_factory=dict)
    proxy_headers: dict[str, str] = field(default_factory=dict)
    proxy_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "Settings":
        rates = dict(DEFAULT_MODEL_RATES)
        raw_rates = os.getenv("AGENTCORD_MODEL_RATES_JSON", "").strip()
        if raw_rates:
            rates.update(json.loads(raw_rates))

        proxy_env = {
            key: value.strip()
            for key, value in os.environ.items()
            if key.startswith("PROXY") and value.strip()
        }
        proxy_headers: dict[str, str] = {}
        raw_proxy_headers = proxy_env.get("PROXY_HEADERS_JSON", "")
        if raw_proxy_headers:
            parsed_headers = json.loads(raw_proxy_headers)
            if not isinstance(parsed_headers, dict):
                raise ValueError("PROXY_HEADERS_JSON 必須是 JSON 物件。")
            proxy_headers = {
                str(key): str(value)
                for key, value in parsed_headers.items()
            }

        proxy_url = proxy_env.get("PROXY_URL", "").strip()
        if not proxy_url:
            proxy_host = proxy_env.get("PROXY_HOST", "").strip()
            if proxy_host:
                proxy_scheme = proxy_env.get("PROXY_SCHEME", "http").strip() or "http"
                proxy_port = proxy_env.get("PROXY_PORT", "").strip()
                proxy_url = f"{proxy_scheme}://{proxy_host}"
                if proxy_port:
                    proxy_url = f"{proxy_url}:{proxy_port}"

        return cls(
            discord_token=os.getenv("DISCORD_TOKEN", "").strip(),
            discord_application_id=_optional_int(os.getenv("DISCORD_APPLICATION_ID")),
            bot_owner_id=_optional_int(os.getenv("BOT_OWNER_ID")),
            discord_log_webhook=os.getenv("DISCORD_LOG_WEBHOOK", "").strip(),
            data_dir=Path(os.getenv("AGENTCORD_DATA_DIR", "data")).resolve(),
            workspace_limit_bytes=int(os.getenv("AGENTCORD_WORKSPACE_LIMIT_BYTES", str(5 * 1024 * 1024))),
            default_credits=float(os.getenv("AGENTCORD_DEFAULT_CREDITS", "100")),
            default_pollinations_model=os.getenv("AGENTCORD_DEFAULT_MODEL", "openai").strip() or "openai",
            pollinations_api_key=os.getenv("POLLINATIONS_API_KEY", "").strip(),
            custom_provider_base_url=os.getenv("AGENTCORD_CUSTOM_PROVIDER_BASE_URL", "").strip(),
            proxy_url=proxy_url,
            proxy_username=proxy_env.get("PROXY_USERNAME", proxy_env.get("PROXY_USER", "")).strip(),
            proxy_password=proxy_env.get("PROXY_PASSWORD", proxy_env.get("PROXY_PASS", "")).strip(),
            agent_max_iterations=max(1, int(os.getenv("AGENTCORD_AGENT_MAX_ITERATIONS", "6"))),
            agent_max_actions_per_iteration=max(1, int(os.getenv("AGENTCORD_AGENT_MAX_ACTIONS", "8"))),
            credit_reserve_output_tokens=max(128, int(os.getenv("AGENTCORD_CREDIT_RESERVE_OUTPUT_TOKENS", "1024"))),
            model_rates=rates,
            proxy_headers=proxy_headers,
            proxy_env=proxy_env,
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
