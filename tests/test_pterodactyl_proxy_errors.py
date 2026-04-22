from __future__ import annotations

import asyncio
import unittest
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import patch

from agentcord.config import Settings
from agentcord.models import UserPterodactylConfig
from agentcord.pterodactyl import PterodactylError, read_pterodactyl_console


class _FailingWebsocketContext:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def __aenter__(self):
        raise self._error

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeSession:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    def ws_connect(self, *args, **kwargs):
        return _FailingWebsocketContext(self._error)


class PterodactylProxyErrorTests(unittest.IsolatedAsyncioTestCase):
    async def test_read_console_wraps_incomplete_read_from_proxy_handshake(self) -> None:
        settings = Settings(
            discord_token="",
            discord_application_id=None,
            bot_owner_id=None,
            discord_log_webhook="",
            data_dir=Path("data"),
            workspace_limit_bytes=1024,
            default_credits=100,
            default_pollinations_model="openai",
            pollinations_api_key="",
            custom_provider_base_url="",
            proxy_url="socks5://proxy.local:1080",
            proxy_username="",
            proxy_password="",
            agent_max_iterations=4,
            agent_max_actions_per_iteration=4,
            credit_reserve_output_tokens=1024,
        )
        config = UserPterodactylConfig(base_url="https://panel.example.com", api_key="token")

        @asynccontextmanager
        async def fake_open_proxy_aware_session(base_session, incoming_settings, *, require_proxy=False):
            del base_session, incoming_settings, require_proxy
            yield _FakeSession(asyncio.IncompleteReadError(partial=b"", expected=3))

        async def fake_get_websocket_credentials(session, incoming_settings, incoming_config, server):
            del session, incoming_settings, incoming_config, server
            return "jwt-token", "wss://panel.example.com/api/client/servers/test/ws"

        with (
            patch("agentcord.pterodactyl.open_proxy_aware_session", fake_open_proxy_aware_session),
            patch("agentcord.pterodactyl.get_pterodactyl_websocket_credentials", fake_get_websocket_credentials),
        ):
            with self.assertRaisesRegex(PterodactylError, "SOCKS 握手"):
                await read_pterodactyl_console(
                    None,
                    settings,
                    config,
                    "test",
                    wait_seconds=1,
                    max_lines=10,
                )


if __name__ == "__main__":
    unittest.main()