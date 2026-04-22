from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from agentcord.config import Settings
from agentcord.models import UserPterodactylConfig
from agentcord.proxy import ProxyConfigurationError, build_proxy_request_kwargs, open_proxy_aware_session

_CLIENT_ACCEPT_HEADER = "Application/vnd.pterodactyl.v1+json"
_ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


class PterodactylError(ValueError):
    pass


@dataclass(slots=True)
class PterodactylResponse:
    status: int
    data: Any
    text: str


def build_required_proxy_request_kwargs(settings: Settings) -> dict[str, Any]:
    try:
        return build_proxy_request_kwargs(settings, require_proxy=True)
    except ProxyConfigurationError as exc:
        raise PterodactylError(str(exc)) from exc


def normalize_pterodactyl_base_url(base_url: str) -> str:
    raw = base_url.strip()
    if not raw:
        raise PterodactylError("Pterodactyl API 網址不能是空的。")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PterodactylError("Pterodactyl API 網址格式無效，請輸入 http 或 https 網址。")

    path = parsed.path.rstrip("/")
    marker = "/api/client"
    marker_index = path.find(marker)
    if marker_index >= 0:
        normalized_path = path[: marker_index + len(marker)]
    elif path.endswith("/api"):
        normalized_path = f"{path}/client"
    elif path:
        normalized_path = f"{path}/api/client"
    else:
        normalized_path = marker

    return urlunsplit((parsed.scheme, parsed.netloc, normalized_path, "", ""))


def build_pterodactyl_client_url(base_url: str, path: str) -> str:
    relative_path = (path or "").strip()
    if relative_path.startswith(("http://", "https://")):
        raise PterodactylError("Pterodactyl path 必須是相對於 /api/client 的路徑。")
    if relative_path in {"", ".", "/"}:
        return base_url.rstrip("/")
    if relative_path.startswith("/api/client"):
        relative_path = relative_path[len("/api/client") :]
    return f"{base_url.rstrip('/')}/{relative_path.lstrip('/')}"


def normalize_pterodactyl_server_path(path: str) -> str:
    cleaned = str(path or "").strip().replace("\\", "/")
    if not cleaned or cleaned == "/":
        return "/"
    pure = PurePosixPath(cleaned)
    if pure.is_absolute():
        parts = [part for part in pure.parts if part not in ("", "/")]
    else:
        parts = [part for part in pure.parts if part not in ("", ".")]
    if ".." in parts:
        raise PterodactylError("Pterodactyl server path 不允許包含 ..。")
    if not parts:
        return "/"
    return "/" + PurePosixPath(*parts).as_posix()


def join_pterodactyl_server_path(base_path: str, relative_path: str) -> str:
    base = normalize_pterodactyl_server_path(base_path)
    relative = normalize_pterodactyl_server_path(relative_path)
    if base == "/":
        return relative
    if relative == "/":
        return base
    return f"{base.rstrip('/')}/{relative.lstrip('/')}"


def _format_pterodactyl_error(status: int, response_text: str) -> str:
    lowered_text = response_text.lower()
    if "cloudflare" in lowered_text and ("you have been blocked" in lowered_text or "attention required" in lowered_text):
        ray_match = re.search(r"Cloudflare Ray ID:\s*<strong[^>]*>([^<]+)</strong>", response_text, flags=re.IGNORECASE)
        ray_id = ray_match.group(1).strip() if ray_match else ""
        suffix = f" Cloudflare Ray ID: {ray_id}" if ray_id else ""
        return (
            f"Pterodactyl API 請求失敗（{status}）：目標站點的 Cloudflare 已封鎖目前 proxy 的出口 IP。"
            "請更換 proxy／出口 IP，或請站方放行這個來源。"
            f"{suffix}"
        )

    details: list[str] = []
    payload: Any = None
    if response_text.strip():
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            payload = None

    if isinstance(payload, dict):
        errors = payload.get("errors")
        if isinstance(errors, list):
            for item in errors[:3]:
                if not isinstance(item, dict):
                    continue
                detail = str(item.get("detail") or item.get("code") or "").strip()
                if detail:
                    details.append(detail)
        if not details:
            detail = str(payload.get("message") or payload.get("error") or "").strip()
            if detail:
                details.append(detail)

    if not details and response_text.strip():
        details.append(response_text.strip())

    detail_text = "；".join(details) if details else "未知錯誤"
    return f"Pterodactyl API 請求失敗（{status}）：{detail_text}"


def _decode_response_data(response: aiohttp.ClientResponse, response_text: str, expect: str) -> Any:
    if response.status == 204 or not response_text.strip():
        return None
    if expect == "text":
        return response_text

    content_type = response.headers.get("Content-Type", "").lower()
    should_parse_json = expect == "json" or "json" in content_type or response_text.lstrip().startswith(("{", "["))
    if not should_parse_json:
        return response_text

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        if expect == "json" or "json" in content_type:
            raise PterodactylError("Pterodactyl 回傳了無法解析的 JSON。") from exc
        return response_text


def _redact_proxy_url(proxy_url: str) -> str:
    parsed = urlsplit(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return "已設定的 proxy"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}"


def _format_pterodactyl_network_error(exc: BaseException, settings: Settings, request_url: str) -> str:
    target_host = urlsplit(request_url).netloc or request_url
    if settings.proxy_url:
        proxy_label = _redact_proxy_url(settings.proxy_url)
        if isinstance(exc, aiohttp.ServerDisconnectedError):
            return (
                "Pterodactyl 網路請求失敗：proxy 在建立連線時提前中斷。"
                f"目標：{target_host}；proxy：{proxy_label}。"
                "這通常不是 API key 被拒絕，而是 proxy 設定、proxy 驗證，或 proxy 不允許連到該 HTTPS 主機。"
            )
        if isinstance(exc, asyncio.TimeoutError):
            return (
                "Pterodactyl 網路請求逾時：透過 proxy 連線到目標主機時未在時間內完成。"
                f"目標：{target_host}；proxy：{proxy_label}。"
                "請檢查 proxy 是否可連外、是否允許 CONNECT 到該主機，以及 DNS 是否可解析。"
            )
        return (
            "Pterodactyl 網路請求失敗：無法透過 proxy 完成連線。"
            f"目標：{target_host}；proxy：{proxy_label}；錯誤：{type(exc).__name__}: {exc}"
        )
    if isinstance(exc, asyncio.TimeoutError):
        return f"Pterodactyl 網路請求逾時：連線到 {target_host} 時未在時間內完成。"
    return f"Pterodactyl 網路請求失敗：連線到 {target_host} 時發生 {type(exc).__name__}: {exc}"


async def request_pterodactyl_client_api(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: Any = None,
    raw_body: str | bytes | None = None,
    content_type: str | None = None,
    expect: str = "auto",
) -> PterodactylResponse:
    normalized_method = method.strip().upper()
    if normalized_method not in _ALLOWED_METHODS:
        raise PterodactylError("Pterodactyl method 只能是 GET、POST、PUT、PATCH 或 DELETE。")
    normalized_expect = expect.strip().lower() or "auto"
    if normalized_expect not in {"auto", "json", "text"}:
        raise PterodactylError("Pterodactyl expect 只能是 auto、json 或 text。")
    if body is not None and raw_body is not None:
        raise PterodactylError("Pterodactyl request 不能同時提供 body 與 raw_body。")
    normalized_api_key = config.api_key.strip()
    if not normalized_api_key:
        raise PterodactylError("你尚未設定 Pterodactyl API key，請先使用 /set-pterodactyl。")

    base_url = normalize_pterodactyl_base_url(config.base_url)
    url = build_pterodactyl_client_url(base_url, path)
    headers = {
        "Authorization": f"Bearer {normalized_api_key}",
        "Accept": _CLIENT_ACCEPT_HEADER,
    }
    proxy_request_kwargs = build_required_proxy_request_kwargs(settings)
    request_kwargs: dict[str, Any] = {}
    if raw_body is not None:
        request_kwargs["data"] = raw_body.encode("utf-8") if isinstance(raw_body, str) else raw_body
        headers["Content-Type"] = content_type or "text/plain; charset=utf-8"
    elif body is not None:
        request_kwargs["json"] = body
        headers["Content-Type"] = content_type or "application/json"
    params = {str(key): value for key, value in (query or {}).items()}

    try:
        async with open_proxy_aware_session(session, settings, require_proxy=True) as request_session:
            async with request_session.request(
                normalized_method,
                url,
                params=params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=45),
                **proxy_request_kwargs,
                **request_kwargs,
            ) as response:
                response_text = await response.text()
                if response.status >= 400:
                    raise PterodactylError(_format_pterodactyl_error(response.status, response_text))
                return PterodactylResponse(
                    status=response.status,
                    data=_decode_response_data(response, response_text, normalized_expect),
                    text=response_text,
                )
    except ProxyConfigurationError as exc:
        raise PterodactylError(str(exc)) from exc
    except PterodactylError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise PterodactylError(_format_pterodactyl_network_error(exc, settings, url)) from exc


async def fetch_pterodactyl_account(
    session: aiohttp.ClientSession,
    settings: Settings,
    base_url: str,
    api_key: str,
) -> tuple[UserPterodactylConfig, dict[str, Any]]:
    normalized_config = UserPterodactylConfig(
        base_url=normalize_pterodactyl_base_url(base_url),
        api_key=api_key.strip(),
    )
    response = await request_pterodactyl_client_api(
        session,
        settings,
        normalized_config,
        "GET",
        "/account",
        expect="json",
    )
    if not isinstance(response.data, dict):
        raise PterodactylError("Pterodactyl 帳號驗證回應格式無效。")
    return normalized_config, response.data


async def get_pterodactyl_startup(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
) -> dict[str, Any]:
    response = await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "GET",
        f"/servers/{server}/startup",
        expect="json",
    )
    if not isinstance(response.data, dict):
        raise PterodactylError("Pterodactyl startup 回應格式無效。")
    return response.data


async def update_pterodactyl_startup_variable(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    key: str,
    value: str,
) -> dict[str, Any]:
    response = await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "PUT",
        f"/servers/{server}/startup/variable",
        body={"key": key, "value": value},
        expect="json",
    )
    if not isinstance(response.data, dict):
        raise PterodactylError("Pterodactyl startup variable 回應格式無效。")
    return response.data


async def set_pterodactyl_power_state(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    signal: str,
) -> None:
    await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "POST",
        f"/servers/{server}/power",
        body={"signal": signal},
        expect="auto",
    )


async def send_pterodactyl_console_command(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    command: str,
) -> None:
    await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "POST",
        f"/servers/{server}/command",
        body={"command": command},
        expect="auto",
    )


async def read_pterodactyl_server_file(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    path: str,
) -> str:
    response = await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "GET",
        f"/servers/{server}/files/contents",
        query={"file": normalize_pterodactyl_server_path(path)},
        expect="text",
    )
    return str(response.data or "")


async def write_pterodactyl_server_file(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    path: str,
    content: str,
) -> None:
    await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "POST",
        f"/servers/{server}/files/write",
        query={"file": normalize_pterodactyl_server_path(path)},
        raw_body=content,
        content_type="text/plain; charset=utf-8",
        expect="auto",
    )


async def create_pterodactyl_server_folder(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    path: str,
) -> None:
    normalized_path = normalize_pterodactyl_server_path(path)
    if normalized_path == "/":
        return
    pure = PurePosixPath(normalized_path)
    root = str(pure.parent).replace("\\", "/")
    name = pure.name
    try:
        await request_pterodactyl_client_api(
            session,
            settings,
            config,
            "POST",
            f"/servers/{server}/files/create-folder",
            body={"root": root or "/", "name": name},
            expect="auto",
        )
    except PterodactylError as exc:
        lowered = str(exc).lower()
        if "exist" in lowered or "already" in lowered:
            return
        raise


async def list_pterodactyl_server_directory(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    path: str,
) -> list[dict[str, Any]]:
    normalized_path = normalize_pterodactyl_server_path(path)
    response = await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "GET",
        f"/servers/{server}/files/list",
        query={"directory": normalized_path},
        expect="json",
    )

    raw_items: Any = None
    if isinstance(response.data, dict):
        if isinstance(response.data.get("data"), list):
            raw_items = response.data.get("data")
        elif isinstance(response.data.get("files"), list):
            raw_items = response.data.get("files")
    elif isinstance(response.data, list):
        raw_items = response.data
    if not isinstance(raw_items, list):
        raise PterodactylError("Pterodactyl 檔案列表回應格式無效。")

    entries: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        attributes = item.get("attributes") if isinstance(item.get("attributes"), dict) else item
        if not isinstance(attributes, dict):
            continue
        name = str(attributes.get("name") or attributes.get("filename") or "").strip()
        if not name:
            continue
        raw_is_file = attributes.get("is_file")
        if isinstance(raw_is_file, bool):
            kind = "file" if raw_is_file else "folder"
        else:
            mimetype = str(attributes.get("mimetype") or "").strip().lower()
            kind = "folder" if mimetype in {"inode/directory", "directory"} else "file"
        size_value = attributes.get("size", 0)
        size = size_value if isinstance(size_value, int) and size_value >= 0 else 0
        entries.append(
            {
                "name": name,
                "kind": kind,
                "size": size,
                "mimetype": str(attributes.get("mimetype") or "").strip(),
            }
        )
    return entries


async def collect_pterodactyl_server_files(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    path: str,
) -> dict[str, Any]:
    normalized_path = normalize_pterodactyl_server_path(path)
    files: list[dict[str, Any]] = []

    async def visit(directory_path: str, prefix: str) -> None:
        entries = await list_pterodactyl_server_directory(session, settings, config, server, directory_path)
        for entry in entries:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            remote_item_path = join_pterodactyl_server_path(directory_path, name)
            relative_path = name if not prefix else f"{prefix}/{name}"
            if entry.get("kind") == "folder":
                await visit(remote_item_path, relative_path)
                continue
            files.append(
                {
                    "remote_path": remote_item_path,
                    "relative_path": relative_path,
                    "size": int(entry.get("size") or 0),
                    "mimetype": str(entry.get("mimetype") or "").strip(),
                }
            )

    try:
        await visit(normalized_path, "")
        return {
            "source_path": normalized_path,
            "source_kind": "folder",
            "files": files,
        }
    except PterodactylError as exc:
        if normalized_path == "/":
            raise
        try:
            content = await read_pterodactyl_server_file(session, settings, config, server, normalized_path)
        except PterodactylError:
            raise exc
        file_name = PurePosixPath(normalized_path).name or "file"
        return {
            "source_path": normalized_path,
            "source_kind": "file",
            "files": [
                {
                    "remote_path": normalized_path,
                    "relative_path": file_name,
                    "size": len(content.encode("utf-8")),
                    "mimetype": "text/plain",
                }
            ],
        }


async def get_pterodactyl_websocket_credentials(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
) -> tuple[str, str]:
    response = await request_pterodactyl_client_api(
        session,
        settings,
        config,
        "GET",
        f"/servers/{server}/websocket",
        expect="json",
    )
    if not isinstance(response.data, dict):
        raise PterodactylError("Pterodactyl websocket token 回應格式無效。")
    data = response.data.get("data")
    if not isinstance(data, dict):
        raise PterodactylError("Pterodactyl websocket token 回應缺少 data。")
    token = str(data.get("token") or "").strip()
    socket_url = str(data.get("socket") or "").strip()
    if not token or not socket_url:
        raise PterodactylError("Pterodactyl websocket token 回應缺少 token 或 socket。")
    return token, socket_url


async def read_pterodactyl_console(
    session: aiohttp.ClientSession,
    settings: Settings,
    config: UserPterodactylConfig,
    server: str,
    *,
    wait_seconds: float = 5.0,
    max_lines: int = 80,
) -> dict[str, Any]:
    if wait_seconds <= 0 or wait_seconds > 30:
        raise PterodactylError("read_console 的 wait_seconds 必須介於 0 到 30 秒之間。")
    if max_lines <= 0 or max_lines > 400:
        raise PterodactylError("read_console 的 max_lines 必須介於 1 到 400 之間。")

    token, socket_url = await get_pterodactyl_websocket_credentials(session, settings, config, server)
    base = urlsplit(normalize_pterodactyl_base_url(config.base_url))
    origin = urlunsplit((base.scheme, base.netloc, "", "", ""))
    proxy_request_kwargs = build_required_proxy_request_kwargs(settings)
    console_lines: list[str] = []
    statuses: list[str] = []
    stats: list[dict[str, Any] | str] = []
    daemon_messages: list[str] = []
    jwt_errors: list[str] = []
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds

    try:
        async with open_proxy_aware_session(session, settings, require_proxy=True) as request_session:
            async with request_session.ws_connect(
                socket_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Origin": origin,
                },
                heartbeat=20,
                timeout=aiohttp.ClientTimeout(total=max(10, wait_seconds + 5)),
                **proxy_request_kwargs,
            ) as websocket:
                await websocket.send_json({"event": "auth", "args": [token]})
                while True:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        message = await asyncio.wait_for(websocket.receive(), timeout=min(remaining, 2.0))
                    except asyncio.TimeoutError:
                        continue

                    if message.type in {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                        break
                    if message.type is aiohttp.WSMsgType.ERROR:
                        raise PterodactylError(f"Pterodactyl websocket 錯誤：{websocket.exception()}")
                    if message.type is not aiohttp.WSMsgType.TEXT:
                        continue

                    try:
                        payload = json.loads(message.data)
                    except json.JSONDecodeError:
                        daemon_messages.append(str(message.data))
                        daemon_messages[:] = daemon_messages[-20:]
                        continue

                    event = str(payload.get("event") or "").strip().lower()
                    args = payload.get("args") if isinstance(payload.get("args"), list) else []
                    if event == "console output" and args:
                        raw_output = str(args[0])
                        lines = raw_output.splitlines() or [raw_output]
                        for line in lines:
                            normalized_line = line.rstrip("\r")
                            if normalized_line:
                                console_lines.append(normalized_line)
                        console_lines[:] = console_lines[-max_lines:]
                        continue
                    if event == "status" and args:
                        statuses.append(str(args[0]))
                        statuses[:] = statuses[-20:]
                        continue
                    if event == "stats" and args:
                        raw_stats = args[0]
                        if isinstance(raw_stats, str):
                            try:
                                stats_payload: dict[str, Any] | str = json.loads(raw_stats)
                            except json.JSONDecodeError:
                                stats_payload = raw_stats
                        else:
                            stats_payload = raw_stats if isinstance(raw_stats, dict) else str(raw_stats)
                        stats.append(stats_payload)
                        stats[:] = stats[-5:]
                        continue
                    if event == "daemon message" and args:
                        daemon_messages.append(str(args[0]))
                        daemon_messages[:] = daemon_messages[-20:]
                        continue
                    if event == "jwt error" and args:
                        jwt_errors.append(str(args[0]))
                        jwt_errors[:] = jwt_errors[-10:]
    except ProxyConfigurationError as exc:
        raise PterodactylError(str(exc)) from exc
    except PterodactylError:
        raise
    except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
        raise PterodactylError(_format_pterodactyl_network_error(exc, settings, socket_url)) from exc

    return {
        "console": console_lines,
        "status": statuses[-1] if statuses else "",
        "statuses": statuses,
        "stats": stats,
        "daemon_messages": daemon_messages,
        "jwt_errors": jwt_errors,
    }
