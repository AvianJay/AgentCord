from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

import aiohttp

from agentcord.config import Settings
from agentcord.models import AIResponse, AIUsage, Provider, UserModelConfig, estimate_tokens


class AIProvider(ABC):
    def __init__(self, session: aiohttp.ClientSession, settings: Settings, config: UserModelConfig) -> None:
        self.session = session
        self.settings = settings
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        raise NotImplementedError

    def _build_usage(self, messages: list[dict[str, str]], content: str) -> AIUsage:
        input_tokens = estimate_tokens("\n".join(message.get("content", "") for message in messages))
        output_tokens = estimate_tokens(content)
        rate = self.settings.get_model_rate(self.config.provider, self.config.model)
        cost = (input_tokens + output_tokens) * rate
        return AIUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model_rate=rate,
        )


class PollinationsProvider(AIProvider):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key or self.settings.pollinations_api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key or self.settings.pollinations_api_key}"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        async with self.session.post(
            "https://gen.pollinations.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = data["choices"][0]["message"]["content"]
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)


class OpenAICompatibleProvider(AIProvider):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        settings: Settings,
        config: UserModelConfig,
        base_url: str,
    ) -> None:
        super().__init__(session, settings, config)
        self.base_url = base_url.rstrip("/")

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = data["choices"][0]["message"]["content"]
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)


class AnthropicProvider(AIProvider):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        payload = {
            "model": self.config.model,
            "max_tokens": kwargs.get("max_tokens", 1600),
            "system": "\n\n".join(system_parts),
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
                if message["role"] != "system"
            ],
        }
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        async with self.session.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = "".join(block["text"] for block in data.get("content", []) if block.get("type") == "text")
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)


class GoogleProvider(AIProvider):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        parts = [
            {"role": "model" if message["role"] == "assistant" else "user", "parts": [{"text": message["content"]}]}
            for message in messages
        ]
        async with self.session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={self.config.api_key}",
            json={"contents": parts},
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)


def create_provider(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserModelConfig,
) -> AIProvider:
    if config.provider is Provider.POLLINATIONS:
        return PollinationsProvider(session, settings, config)
    if config.provider is Provider.OPENAI:
        return OpenAICompatibleProvider(session, settings, config, "https://api.openai.com/v1")
    if config.provider is Provider.XAI:
        return OpenAICompatibleProvider(session, settings, config, "https://api.x.ai/v1")
    if config.provider is Provider.CUSTOM:
        if not settings.custom_provider_base_url:
            raise ValueError("自訂供應商必須設定 AGENTCORD_CUSTOM_PROVIDER_BASE_URL。")
        return OpenAICompatibleProvider(session, settings, config, settings.custom_provider_base_url)
    if config.provider is Provider.ANTHROPIC:
        return AnthropicProvider(session, settings, config)
    if config.provider is Provider.GOOGLE:
        return GoogleProvider(session, settings, config)
    raise ValueError(f"不支援的供應商：{config.provider}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError("模型輸出不包含 JSON 物件。")
    return json.loads(cleaned[start : end + 1])
