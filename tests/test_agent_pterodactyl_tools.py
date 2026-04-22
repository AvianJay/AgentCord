from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agentcord.agent import CodingAgent
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import UserPterodactylConfig
from agentcord.workspace import WorkspaceManager


class AgentPterodactylToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.settings = Settings(
            discord_token="",
            discord_application_id=None,
            bot_owner_id=None,
            discord_log_webhook="",
            data_dir=root / "data",
            workspace_limit_bytes=1024,
            default_credits=100,
            default_pollinations_model="openai",
            pollinations_api_key="",
            custom_provider_base_url="",
            proxy_url="",
            proxy_username="",
            proxy_password="",
            agent_max_iterations=4,
            agent_max_actions_per_iteration=4,
            credit_reserve_output_tokens=1024,
        )
        self.db = Database(root / "agentcord.db", default_credits=100)
        self.addCleanup(self.db.close)
        self.workspace = WorkspaceManager(root / "workspaces", limit_bytes=1024)
        self.user_id = 123
        self.db.set_pterodactyl_config(
            self.user_id,
            UserPterodactylConfig(base_url="https://panel.example.com", api_key="token"),
        )
        self.agent = CodingAgent(self.settings, self.db, self.workspace, None)

    async def test_list_servers_tool_handler_returns_server_list(self) -> None:
        servers = [
            {
                "identifier": "abc123",
                "uuid": "uuid-1",
                "name": "Lobby",
                "description": "",
                "node": "",
                "is_owner": True,
                "is_suspended": False,
                "is_installing": False,
                "current_state": "running",
            }
        ]

        with patch("agentcord.agent.list_pterodactyl_servers", new=AsyncMock(return_value=servers)) as mocked:
            result, touched_files, current_task_items = await self.agent._tool_pterodactyl_list_servers(
                self.user_id,
                {"tool": "pterodactyl_list_servers"},
                [],
                None,
            )

        self.assertEqual(result, {"count": 1, "result": servers})
        self.assertEqual(touched_files, [])
        self.assertEqual(current_task_items, [])
        mocked.assert_awaited_once()

    async def test_read_startup_tool_handler_passes_server_identifier(self) -> None:
        startup = {"meta": {"startup_command": "java -jar server.jar"}}

        with patch("agentcord.agent.get_pterodactyl_startup", new=AsyncMock(return_value=startup)) as mocked:
            result, touched_files, current_task_items = await self.agent._tool_pterodactyl_read_startup(
                self.user_id,
                {"tool": "pterodactyl_read_startup", "server": "srv-1"},
                [],
                None,
            )

        self.assertEqual(result, {"server": "srv-1", "result": startup})
        self.assertEqual(touched_files, [])
        self.assertEqual(current_task_items, [])
        mocked.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()