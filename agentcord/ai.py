from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

import aiohttp

from agentcord.config import Settings
from agentcord.models import AIResponse, AIUsage, PollinationsModelInfo, Provider, UserModelConfig, estimate_tokens

_POLLINATIONS_MODELS_CACHE_TTL = 900.0
_pollinations_models_cache: tuple[float, list[PollinationsModelInfo], dict[str, PollinationsModelInfo]] | None = None


class AIProvider(ABC):
    def __init__(self, session: aiohttp.ClientSession, settings: Settings, config: UserModelConfig) -> None:
        self.session = session
        self.settings = settings
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        raise NotImplementedError

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], Awaitable[None] | None] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        response = await self.generate(messages, **kwargs)
        if on_delta and response.content:
            maybe_result = on_delta(response.content)
            if maybe_result is not None:
                await maybe_result
        return response

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

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], Awaitable[None] | None] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key or self.settings.pollinations_api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key or self.settings.pollinations_api_key}"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": True,
        }
        async with self.session.post(
            "https://gen.pollinations.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            if response.content_type == "application/json":
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                if on_delta and content:
                    maybe_result = on_delta(content)
                    if maybe_result is not None:
                        await maybe_result
                return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)

            content_parts: list[str] = []
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_line = line[5:].strip()
                if payload_line == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_line)
                except json.JSONDecodeError:
                    continue
                delta = _extract_stream_text(chunk)
                if not delta:
                    continue
                content_parts.append(delta)
                if on_delta:
                    maybe_result = on_delta(delta)
                    if maybe_result is not None:
                        await maybe_result

        content = "".join(content_parts)
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response={"stream": True})


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

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], Awaitable[None] | None] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": True,
        }
        async with self.session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as response:
            response.raise_for_status()
            if response.content_type == "application/json":
                data = await response.json()
                content = data["choices"][0]["message"]["content"]
                if on_delta and content:
                    maybe_result = on_delta(content)
                    if maybe_result is not None:
                        await maybe_result
                return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response=data)

            content_parts: list[str] = []
            async for raw_line in response.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                payload_line = line[5:].strip()
                if payload_line == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_line)
                except json.JSONDecodeError:
                    continue
                delta = _extract_stream_text(chunk)
                if not delta:
                    continue
                content_parts.append(delta)
                if on_delta:
                    maybe_result = on_delta(delta)
                    if maybe_result is not None:
                        await maybe_result

        content = "".join(content_parts)
        return AIResponse(content=content, usage=self._build_usage(messages, content), raw_response={"stream": True})


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


async def fetch_pollinations_models(
    session: aiohttp.ClientSession,
    settings: Settings,
    *,
    force_refresh: bool = False,
) -> list[PollinationsModelInfo]:
    global _pollinations_models_cache

    if not force_refresh and _pollinations_models_cache is not None:
        cached_at, cached_models, _ = _pollinations_models_cache
        if time.time() - cached_at < _POLLINATIONS_MODELS_CACHE_TTL:
            return cached_models

    headers = {"Content-Type": "application/json"}
    if settings.pollinations_api_key:
        headers["Authorization"] = f"Bearer {settings.pollinations_api_key}"
    async with session.get(
        "https://gen.pollinations.ai/text/models",
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=45),
    ) as response:
        response.raise_for_status()
        payload = await response.json()

    models: list[PollinationsModelInfo] = []
    lookup: dict[str, PollinationsModelInfo] = {}
    for item in payload:
        if not isinstance(item, dict) or not str(item.get("name", "")).strip():
            continue
        model = PollinationsModelInfo(
            name=str(item.get("name", "")).strip(),
            aliases=[str(alias).strip() for alias in item.get("aliases", []) if str(alias).strip()],
            description=str(item.get("description", "")).strip(),
            context_length=int(item["context_length"]) if item.get("context_length") is not None else None,
            paid_only=bool(item.get("paid_only", False)),
            tools=bool(item.get("tools", False)),
        )
        models.append(model)
        lookup[model.name] = model
        for alias in model.aliases:
            lookup.setdefault(alias, model)

    models.sort(key=lambda item: item.name)
    _pollinations_models_cache = (time.time(), models, lookup)
    return models


async def resolve_pollinations_model(
    session: aiohttp.ClientSession,
    settings: Settings,
    model_name: str,
) -> PollinationsModelInfo | None:
    models = await fetch_pollinations_models(session, settings)
    if _pollinations_models_cache is None:
        return None
    _, _, lookup = _pollinations_models_cache
    return lookup.get(model_name)


def _extract_stream_text(chunk: dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    delta = first_choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    text = first_choice.get("text")
    if isinstance(text, str):
        return text
    return ""
