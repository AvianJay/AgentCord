from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from urllib.parse import quote, urlsplit, urlunsplit

import aiohttp

from agentcord.config import Settings

try:
    from aiohttp_socks import ProxyConnector
except ImportError:  # pragma: no cover - handled at runtime when dependency is missing
    ProxyConnector = None


_SOCKS_PROXY_SCHEMES = {"socks4", "socks4a", "socks5", "socks5h"}


class ProxyConfigurationError(ValueError):
    pass


def get_proxy_url(settings: Settings, *, require_proxy: bool = False) -> str:
    proxy_url = settings.proxy_url.strip()
    if proxy_url:
        return proxy_url
    if require_proxy:
        raise ProxyConfigurationError(
            "此操作僅允許透過 proxy 請求；請先設定 PROXY_URL，或提供 PROXY_HOST/PROXY_PORT。"
        )
    return ""


def is_socks_proxy_url(proxy_url: str) -> bool:
    return urlsplit(proxy_url).scheme.lower() in _SOCKS_PROXY_SCHEMES


def build_proxy_request_kwargs(settings: Settings, *, require_proxy: bool = False) -> dict[str, Any]:
    proxy_url = get_proxy_url(settings, require_proxy=require_proxy)
    if not proxy_url or is_socks_proxy_url(proxy_url):
        return {}
    request_kwargs: dict[str, Any] = {"proxy": proxy_url}
    if settings.proxy_username:
        request_kwargs["proxy_auth"] = aiohttp.BasicAuth(
            settings.proxy_username,
            settings.proxy_password,
        )
    if settings.proxy_headers:
        request_kwargs["proxy_headers"] = settings.proxy_headers
    return request_kwargs


@asynccontextmanager
async def open_proxy_aware_session(
    base_session: aiohttp.ClientSession,
    settings: Settings,
    *,
    require_proxy: bool = False,
) -> AsyncIterator[aiohttp.ClientSession]:
    proxy_url = get_proxy_url(settings, require_proxy=require_proxy)
    if not proxy_url or not is_socks_proxy_url(proxy_url):
        yield base_session
        return
    if ProxyConnector is None:
        raise ProxyConfigurationError(
            "目前設定的是 SOCKS proxy，但尚未安裝 aiohttp-socks；請安裝依賴或改用 http/https proxy。"
        )
    connector = ProxyConnector.from_url(_build_proxy_url_with_auth(proxy_url, settings))
    async with aiohttp.ClientSession(connector=connector, trust_env=False) as proxy_session:
        yield proxy_session


def _build_proxy_url_with_auth(proxy_url: str, settings: Settings) -> str:
    parsed = urlsplit(proxy_url)
    if parsed.username or not settings.proxy_username:
        return proxy_url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    userinfo = quote(settings.proxy_username, safe="")
    if settings.proxy_password:
        userinfo = f"{userinfo}:{quote(settings.proxy_password, safe='')}"
    netloc = f"{userinfo}@{host}"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))