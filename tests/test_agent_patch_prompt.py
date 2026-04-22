from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.agent import CodingAgent
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.workspace import WorkspaceManager


class AgentPatchPromptTests(unittest.TestCase):
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
        self.agent = CodingAgent(self.settings, self.db, self.workspace, None)

    def test_apply_patch_prompt_includes_required_unified_diff_headers(self) -> None:
        prompt = self.agent._build_agent_system_prompt()

        self.assertIn("--- 舊路徑", prompt)
        self.assertIn("+++ 新路徑", prompt)
        self.assertIn("@@ hunk header", prompt)
        self.assertIn("不要輸出 *** Begin Patch", prompt)
        self.assertIn("合法範例", prompt)
        self.assertIn("--- src/controllers/posts.ts", prompt)
        self.assertIn("+++ src/controllers/posts.ts", prompt)


if __name__ == "__main__":
    unittest.main()