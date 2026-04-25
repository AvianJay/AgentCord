from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from urllib.parse import urlsplit
from urllib.parse import parse_qsl, urlencode, urlunsplit
from typing import Any, Awaitable, Callable

import aiohttp

from agentcord.config import Settings
from agentcord.models import AIResponse, AIUsage, PollinationsModelInfo, Provider, ProviderModelInfo, UserModelConfig, estimate_tokens
from agentcord.proxy import build_proxy_request_kwargs, open_proxy_aware_session

_POLLINATIONS_MODELS_CACHE_TTL = 900.0
_pollinations_models_cache: tuple[float, list[PollinationsModelInfo], dict[str, PollinationsModelInfo]] | None = None
_PROVIDER_MODELS_CACHE_TTL = 900.0
_provider_models_cache: dict[
    tuple[str, str, str],
    tuple[float, list[ProviderModelInfo], dict[str, ProviderModelInfo]],
] = {}
_OPENAI_COMPATIBLE_BASE_URLS = {
    Provider.OPENAI: "https://api.openai.com/v1",
    Provider.XAI: "https://api.x.ai/v1",
    Provider.POE: "https://api.poe.com/v1",
    Provider.OPENROUTER: "https://openrouter.ai/api/v1",
    Provider.DEEPSEEK: "https://api.deepseek.com",
    Provider.NVIDIA_NIM: "https://integrate.api.nvidia.com/v1",
}
_OPENAI_COMPATIBLE_PROVIDERS = set(_OPENAI_COMPATIBLE_BASE_URLS) | {Provider.CUSTOM}
_THINKING_TAG_RE = re.compile(r"<(?P<tag>think|thinking)>[\s\S]*?</(?P=tag)>", re.IGNORECASE)
_THINKING_FENCE_RE = re.compile(r"```(?:think|thinking|reasoning|thoughts?)\s*[\s\S]*?```", re.IGNORECASE)
_THINKING_LABEL_RE = re.compile(r"^\s*(?:thinking|reasoning|thought\s*process)\s*(?::|：|\.\.\.)", re.IGNORECASE)
_THINKING_LINE_RE = re.compile(r"^\s*(?:thinking|reasoning|thought\s*process)\s*(?::|：|\.\.\.)?\s*$", re.IGNORECASE)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_BEARER_TOKEN_RE = re.compile(r"(?i)\b(Bearer\s+)([^\s,;]+)")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(api[_-]?key|apikey|access[_-]?token|token|authorization)\b(\s*[=:]\s*|%3[Dd])([^&\s,'\"}]+)"
)
_SENSITIVE_QUERY_KEYS = {"key", "api_key", "apikey", "token", "access_token", "authorization", "auth"}


class ModelJSONParseError(ValueError):
    def __init__(self, output_preview: str = "") -> None:
        self.output_preview = sanitize_sensitive_text(output_preview.strip()) or "(空白輸出)"
        super().__init__("模型輸出不包含合法 JSON 物件。")

    def __str__(self) -> str:
        return f"{super().__str__()} 輸出預覽：{self.output_preview}"


def _resolve_response_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("model", "model_name", "modelVersion", "resolved_model"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def sanitize_ai_response_content(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    cleaned = _THINKING_TAG_RE.sub("", cleaned)
    cleaned = _THINKING_FENCE_RE.sub("", cleaned)

    lines = cleaned.splitlines()
    while lines and (not lines[0].strip() or _THINKING_LINE_RE.match(lines[0])):
        lines.pop(0)
    cleaned = "\n".join(lines).strip()
    if not cleaned:
        return text.strip()

    paragraphs = re.split(r"\n\s*\n", cleaned, maxsplit=1)
    if len(paragraphs) == 2 and _THINKING_LABEL_RE.match(paragraphs[0].strip()):
        cleaned = paragraphs[1].strip()

    return cleaned or text.strip()


def sanitize_sensitive_text(text: str) -> str:
    if not text:
        return text

    def _sanitize_url_match(match: re.Match[str]) -> str:
        url = match.group(0)
        try:
            parsed = urlsplit(url)
        except Exception:
            return url
        if not parsed.query:
            return url
        pairs = parse_qsl(parsed.query, keep_blank_values=True)
        sanitized_pairs = [
            (key, "***" if key.lower() in _SENSITIVE_QUERY_KEYS else value)
            for key, value in pairs
        ]
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urlencode(sanitized_pairs, doseq=True), parsed.fragment))

    redacted = _URL_RE.sub(_sanitize_url_match, text)
    redacted = _BEARER_TOKEN_RE.sub(r"\1***", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}***", redacted)
    return redacted


def build_model_output_preview(text: str, *, limit: int = 280) -> str:
    cleaned = sanitize_sensitive_text(sanitize_ai_response_content(text).strip() or text.strip())
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def format_exception_message(error: BaseException) -> str:
    base_message = sanitize_sensitive_text(str(error).strip()) or type(error).__name__
    preview = sanitize_sensitive_text(str(getattr(error, "output_preview", "")).strip())
    if preview and "輸出預覽：" not in base_message:
        return f"{base_message} 輸出預覽：{preview}"
    return base_message


def _request_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=180, connect=30, sock_connect=30, sock_read=180)


def _stream_timeout() -> aiohttp.ClientTimeout:
    # Streaming responses can legitimately take several minutes; only fail when the socket stalls.
    return aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30, sock_read=300)


def _model_list_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=45, connect=15, sock_connect=15, sock_read=45)


def _cache_secret(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _normalize_model_name(provider: Provider, value: Any) -> str:
    name = str(value or "").strip()
    if provider is Provider.GOOGLE and name.startswith("models/"):
        return name.split("/", 1)[1].strip()
    return name


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        int_value = int(value)
        return int_value if int_value > 0 else None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.isdigit():
            int_value = int(cleaned)
            return int_value if int_value > 0 else None
    return None


def _extract_context_length(payload: Any) -> int | None:
    if isinstance(payload, list):
        for item in payload:
            context_length = _extract_context_length(item)
            if context_length is not None:
                return context_length
        return None
    if not isinstance(payload, dict):
        return None

    for key in (
        "context_length",
        "contextLength",
        "max_context_length",
        "maxContextLength",
        "context_window",
        "contextWindow",
        "input_token_limit",
        "inputTokenLimit",
        "max_input_tokens",
        "maxInputTokens",
        "token_limit",
        "tokenLimit",
    ):
        context_length = _coerce_positive_int(payload.get(key))
        if context_length is not None:
            return context_length

    for key in ("limits", "metadata", "capabilities", "details", "info"):
        nested = payload.get(key)
        context_length = _extract_context_length(nested)
        if context_length is not None:
            return context_length
    return None


def _extract_description(payload: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _provider_cache_key(provider: Provider, api_key: str, *, base_url: str = "") -> tuple[str, str, str]:
    return (provider.value, _cache_secret(api_key.strip()), base_url.rstrip("/"))


def _parse_custom_provider_api_key(value: str) -> tuple[str, str] | None:
    raw = value.strip()
    if not raw:
        return None
    for index in range(len(raw) - 1, -1, -1):
        if raw[index] != ":":
            continue
        base_url = raw[:index].strip().rstrip("/")
        api_key = raw[index + 1 :].strip()
        if not base_url or not api_key:
            continue
        parsed = urlsplit(base_url)
        if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
            return base_url, api_key
    return None


def _resolve_custom_provider_credentials(settings: Settings, value: str) -> tuple[str, str]:
    parsed = _parse_custom_provider_api_key(value)
    if parsed is not None:
        return parsed
    legacy_base_url = settings.custom_provider_base_url.strip().rstrip("/")
    if legacy_base_url and value.strip():
        return legacy_base_url, value.strip()
    raise ValueError(
        "自訂供應商的 api_key 請填 {apiurl}:{apikey}，例如 https://api.example.com/v1:sk-xxx。"
    )


def _resolve_openai_compatible_endpoint(
    settings: Settings,
    provider: Provider,
    api_key: str,
) -> tuple[str, str, bool]:
    raw_api_key = api_key.strip()
    base_url = _OPENAI_COMPATIBLE_BASE_URLS.get(provider)
    if base_url is not None:
        return base_url, raw_api_key, False
    if provider is Provider.CUSTOM:
        base_url, resolved_api_key = _resolve_custom_provider_credentials(settings, raw_api_key)
        return base_url, resolved_api_key, True
    raise ValueError(f"不支援的供應商：{provider}")


def _cache_provider_models(
    provider: Provider,
    api_key: str,
    models: list[ProviderModelInfo],
    *,
    base_url: str = "",
) -> list[ProviderModelInfo]:
    lookup: dict[str, ProviderModelInfo] = {}
    for model in models:
        lookup.setdefault(model.name, model)
        for alias in model.aliases:
            lookup.setdefault(alias, model)
    _provider_models_cache[_provider_cache_key(provider, api_key, base_url=base_url)] = (
        time.time(),
        models,
        lookup,
    )
    return models


def _get_cached_provider_models(
    provider: Provider,
    api_key: str,
    *,
    base_url: str = "",
) -> tuple[list[ProviderModelInfo], dict[str, ProviderModelInfo]] | None:
    cached = _provider_models_cache.get(_provider_cache_key(provider, api_key, base_url=base_url))
    if cached is None:
        return None
    cached_at, models, lookup = cached
    if time.time() - cached_at >= _PROVIDER_MODELS_CACHE_TTL:
        return None
    return models, lookup


def _build_provider_model_info(
    provider: Provider,
    payload: dict[str, Any],
    *,
    name_keys: tuple[str, ...],
    description_keys: tuple[str, ...],
    extra_aliases: list[str] | None = None,
) -> ProviderModelInfo | None:
    raw_name = ""
    for key in name_keys:
        raw_name = str(payload.get(key) or "").strip()
        if raw_name:
            break
    if not raw_name:
        return None

    name = _normalize_model_name(provider, raw_name)
    aliases = [alias for alias in extra_aliases or [] if alias and alias != name]
    if raw_name != name:
        aliases.append(raw_name)

    return ProviderModelInfo(
        name=name,
        aliases=sorted(set(aliases)),
        description=_extract_description(payload, *description_keys),
        context_length=_extract_context_length(payload),
    )


def _sort_and_dedupe_models(models: list[ProviderModelInfo]) -> list[ProviderModelInfo]:
    deduped: dict[str, ProviderModelInfo] = {}
    for model in models:
        existing = deduped.get(model.name)
        if existing is None:
            deduped[model.name] = model
            continue
        if existing.context_length is None and model.context_length is not None:
            existing.context_length = model.context_length
        if not existing.description and model.description:
            existing.description = model.description
        merged_aliases = set(existing.aliases)
        merged_aliases.update(model.aliases)
        existing.aliases = sorted(merged_aliases)
    return sorted(deduped.values(), key=lambda item: item.name)


def _parse_openai_compatible_models(payload: Any, provider: Provider) -> list[ProviderModelInfo]:
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items = [item for item in payload["data"] if isinstance(item, dict)]
    elif isinstance(payload, list):
        items = [item for item in payload if isinstance(item, dict)]

    models: list[ProviderModelInfo] = []
    for item in items:
        model = _build_provider_model_info(
            provider,
            item,
            name_keys=("id", "name"),
            description_keys=("description", "display_name", "owned_by"),
        )
        if model is not None:
            models.append(model)
    return _sort_and_dedupe_models(models)


def _parse_anthropic_models(payload: Any) -> list[ProviderModelInfo]:
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        items = [item for item in payload["data"] if isinstance(item, dict)]

    models: list[ProviderModelInfo] = []
    for item in items:
        display_name = str(item.get("display_name") or item.get("displayName") or "").strip()
        model = _build_provider_model_info(
            Provider.ANTHROPIC,
            item,
            name_keys=("id", "name"),
            description_keys=("description", "display_name", "displayName"),
            extra_aliases=[display_name] if display_name else None,
        )
        if model is not None:
            models.append(model)
    return _sort_and_dedupe_models(models)


def _parse_google_models(payload: Any) -> list[ProviderModelInfo]:
    items: list[dict[str, Any]] = []
    if isinstance(payload, dict) and isinstance(payload.get("models"), list):
        items = [item for item in payload["models"] if isinstance(item, dict)]

    models: list[ProviderModelInfo] = []
    for item in items:
        supported_methods = item.get("supportedGenerationMethods")
        if isinstance(supported_methods, list) and supported_methods:
            supported = {str(method).strip() for method in supported_methods if str(method).strip()}
            if "generateContent" not in supported and "streamGenerateContent" not in supported:
                continue
        display_name = str(item.get("displayName") or item.get("display_name") or "").strip()
        model = _build_provider_model_info(
            Provider.GOOGLE,
            item,
            name_keys=("name",),
            description_keys=("description", "displayName", "display_name"),
            extra_aliases=[display_name] if display_name else None,
        )
        if model is not None:
            models.append(model)
    return _sort_and_dedupe_models(models)


async def _fetch_openai_compatible_models(
    session: aiohttp.ClientSession,
    settings: Settings,
    provider: Provider,
    api_key: str,
    base_url: str,
    *,
    require_proxy: bool = False,
) -> list[ProviderModelInfo]:
    request_kwargs = build_proxy_request_kwargs(settings, require_proxy=require_proxy) if require_proxy else {}
    async with open_proxy_aware_session(session, settings, require_proxy=require_proxy) as request_session:
        async with request_session.get(
            f"{base_url.rstrip('/')}/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=_model_list_timeout(),
            **request_kwargs,
        ) as response:
            response.raise_for_status()
            payload = await response.json()
    return _parse_openai_compatible_models(payload, provider)


async def _fetch_anthropic_models(session: aiohttp.ClientSession, api_key: str) -> list[ProviderModelInfo]:
    async with session.get(
        "https://api.anthropic.com/v1/models",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=_model_list_timeout(),
    ) as response:
        response.raise_for_status()
        payload = await response.json()
    return _parse_anthropic_models(payload)


async def _fetch_google_models(session: aiohttp.ClientSession, api_key: str) -> list[ProviderModelInfo]:
    payloads: list[dict[str, Any]] = []
    page_token = ""
    for _ in range(10):
        params = {"key": api_key}
        if page_token:
            params["pageToken"] = page_token
        async with session.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            params=params,
            timeout=_model_list_timeout(),
        ) as response:
            response.raise_for_status()
            payload = await response.json()
        if isinstance(payload, dict):
            payloads.append(payload)
            page_token = str(payload.get("nextPageToken") or "").strip()
        else:
            page_token = ""
        if not page_token:
            break

    models: list[ProviderModelInfo] = []
    for payload in payloads:
        models.extend(_parse_google_models(payload))
    return _sort_and_dedupe_models(models)


async def fetch_provider_models(
    session: aiohttp.ClientSession,
    settings: Settings,
    provider: Provider,
    api_key: str,
    *,
    force_refresh: bool = False,
) -> list[ProviderModelInfo]:
    if provider is Provider.POLLINATIONS:
        return [
            ProviderModelInfo(
                name=model.name,
                aliases=list(model.aliases),
                description=model.description,
                context_length=model.context_length,
            )
            for model in await fetch_pollinations_models(session, settings, force_refresh=force_refresh)
        ]

    api_key = api_key.strip()
    if not api_key:
        return []

    base_url = ""
    resolved_api_key = api_key
    require_proxy = False
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        base_url, resolved_api_key, require_proxy = _resolve_openai_compatible_endpoint(settings, provider, api_key)

    if not force_refresh:
        cached = _get_cached_provider_models(provider, api_key, base_url=base_url)
        if cached is not None:
            models, _ = cached
            return models

    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        models = await _fetch_openai_compatible_models(
            session,
            settings,
            provider,
            resolved_api_key,
            base_url,
            require_proxy=require_proxy,
        )
    elif provider is Provider.ANTHROPIC:
        models = await _fetch_anthropic_models(session, api_key)
    elif provider is Provider.GOOGLE:
        models = await _fetch_google_models(session, api_key)
    else:
        raise ValueError(f"不支援的供應商：{provider}")

    return _cache_provider_models(provider, api_key, models, base_url=base_url)


async def resolve_provider_model(
    session: aiohttp.ClientSession,
    settings: Settings,
    provider: Provider,
    api_key: str,
    model_name: str,
) -> ProviderModelInfo | None:
    if provider is Provider.POLLINATIONS:
        model = await resolve_pollinations_model(session, settings, model_name)
        if model is None:
            return None
        return ProviderModelInfo(
            name=model.name,
            aliases=list(model.aliases),
            description=model.description,
            context_length=model.context_length,
        )

    api_key = api_key.strip()
    if not api_key:
        return None

    base_url = ""
    if provider in _OPENAI_COMPATIBLE_PROVIDERS:
        base_url, _, _ = _resolve_openai_compatible_endpoint(settings, provider, api_key)

    models = await fetch_provider_models(session, settings, provider, api_key)
    cached = _get_cached_provider_models(provider, api_key, base_url=base_url)
    if cached is None:
        lookup = {model.name: model for model in models}
        for model in models:
            for alias in model.aliases:
                lookup.setdefault(alias, model)
    else:
        _, lookup = cached

    normalized_name = _normalize_model_name(provider, model_name)
    return lookup.get(normalized_name) or lookup.get(model_name.strip())


async def resolve_model_info(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserModelConfig,
) -> ProviderModelInfo | None:
    return await resolve_provider_model(
        session,
        settings,
        config.provider,
        config.api_key,
        config.model,
    )


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
            timeout=_request_timeout(),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = sanitize_ai_response_content(data["choices"][0]["message"]["content"])
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=_resolve_response_model(data) or self.config.model,
            raw_response=data,
        )

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
            timeout=_stream_timeout(),
        ) as response:
            response.raise_for_status()
            if response.content_type == "application/json":
                data = await response.json()
                content = sanitize_ai_response_content(data["choices"][0]["message"]["content"])
                if on_delta and content:
                    maybe_result = on_delta(content)
                    if maybe_result is not None:
                        await maybe_result
                return AIResponse(
                    content=content,
                    usage=self._build_usage(messages, content),
                    model=_resolve_response_model(data) or self.config.model,
                    raw_response=data,
                )

            content_parts: list[str] = []
            response_model = ""
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
                response_model = _resolve_response_model(chunk) or response_model
                delta = _extract_stream_text(chunk)
                if not delta:
                    continue
                content_parts.append(delta)
                if on_delta:
                    maybe_result = on_delta(delta)
                    if maybe_result is not None:
                        await maybe_result

        content = sanitize_ai_response_content("".join(content_parts))
        resolved_model = response_model or self.config.model
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=resolved_model,
            raw_response={"stream": True, "model": resolved_model},
        )


class OpenAICompatibleProvider(AIProvider):
    def __init__(
        self,
        session: aiohttp.ClientSession,
        settings: Settings,
        config: UserModelConfig,
        base_url: str,
        *,
        api_key: str | None = None,
        require_proxy: bool = False,
    ) -> None:
        super().__init__(session, settings, config)
        self.base_url = base_url.rstrip("/")
        self.request_api_key = (api_key if api_key is not None else config.api_key).strip()
        self.require_proxy = require_proxy

    def _proxy_request_kwargs(self) -> dict[str, Any]:
        if not self.require_proxy:
            return {}
        return build_proxy_request_kwargs(self.settings, require_proxy=True)

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        headers = {
            "Authorization": f"Bearer {self.request_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
        }
        async with open_proxy_aware_session(self.session, self.settings, require_proxy=self.require_proxy) as request_session:
            async with request_session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_request_timeout(),
                **self._proxy_request_kwargs(),
            ) as response:
                response.raise_for_status()
                data = await response.json()
        content = sanitize_ai_response_content(data["choices"][0]["message"]["content"])
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=_resolve_response_model(data) or self.config.model,
            raw_response=data,
        )

    async def stream_generate(
        self,
        messages: list[dict[str, str]],
        on_delta: Callable[[str], Awaitable[None] | None] | None = None,
        **kwargs: Any,
    ) -> AIResponse:
        headers = {
            "Authorization": f"Bearer {self.request_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "stream": True,
        }
        async with open_proxy_aware_session(self.session, self.settings, require_proxy=self.require_proxy) as request_session:
            async with request_session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=_stream_timeout(),
                **self._proxy_request_kwargs(),
            ) as response:
                response.raise_for_status()
                if response.content_type == "application/json":
                    data = await response.json()
                    content = sanitize_ai_response_content(data["choices"][0]["message"]["content"])
                    if on_delta and content:
                        maybe_result = on_delta(content)
                        if maybe_result is not None:
                            await maybe_result
                    return AIResponse(
                        content=content,
                        usage=self._build_usage(messages, content),
                        model=_resolve_response_model(data) or self.config.model,
                        raw_response=data,
                    )

                content_parts: list[str] = []
                response_model = ""
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
                    response_model = _resolve_response_model(chunk) or response_model
                    delta = _extract_stream_text(chunk)
                    if not delta:
                        continue
                    content_parts.append(delta)
                    if on_delta:
                        maybe_result = on_delta(delta)
                        if maybe_result is not None:
                            await maybe_result

        content = sanitize_ai_response_content("".join(content_parts))
        resolved_model = response_model or self.config.model
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=resolved_model,
            raw_response={"stream": True, "model": resolved_model},
        )


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
            timeout=_request_timeout(),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = sanitize_ai_response_content(
            "".join(block["text"] for block in data.get("content", []) if block.get("type") == "text")
        )
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=_resolve_response_model(data) or self.config.model,
            raw_response=data,
        )


class GoogleProvider(AIProvider):
    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> AIResponse:
        parts = [
            {"role": "model" if message["role"] == "assistant" else "user", "parts": [{"text": message["content"]}]}
            for message in messages
        ]
        async with self.session.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.config.model}:generateContent?key={self.config.api_key}",
            json={"contents": parts},
            timeout=_request_timeout(),
        ) as response:
            response.raise_for_status()
            data = await response.json()
        content = sanitize_ai_response_content(data["candidates"][0]["content"]["parts"][0]["text"])
        return AIResponse(
            content=content,
            usage=self._build_usage(messages, content),
            model=_resolve_response_model(data) or self.config.model,
            raw_response=data,
        )


def create_provider(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserModelConfig,
) -> AIProvider:
    if config.provider is Provider.POLLINATIONS:
        return PollinationsProvider(session, settings, config)
    if config.provider in _OPENAI_COMPATIBLE_PROVIDERS:
        base_url, resolved_api_key, require_proxy = _resolve_openai_compatible_endpoint(
            settings,
            config.provider,
            config.api_key,
        )
        return OpenAICompatibleProvider(
            session,
            settings,
            config,
            base_url,
            api_key=resolved_api_key,
            require_proxy=require_proxy,
        )
    if config.provider is Provider.ANTHROPIC:
        return AnthropicProvider(session, settings, config)
    if config.provider is Provider.GOOGLE:
        return GoogleProvider(session, settings, config)
    raise ValueError(f"不支援的供應商：{config.provider}")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = sanitize_ai_response_content(text)
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].lstrip()

    decoder = json.JSONDecoder()
    search_start = 0
    while True:
        start = cleaned.find("{", search_start)
        if start == -1:
            break
        try:
            parsed, _ = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            search_start = start + 1
            continue
        if isinstance(parsed, dict):
            return parsed
        search_start = start + 1
    raise ModelJSONParseError(build_model_output_preview(cleaned or text))


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
