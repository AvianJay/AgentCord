from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.agent import CodingAgent, _CURRENT_TASK_ID
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.workspace import WorkspaceManager


class AgentReviewTrackingTests(unittest.IsolatedAsyncioTestCase):
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
            workspace_limit_bytes=4096,
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
        self.workspace = WorkspaceManager(root / "workspaces", limit_bytes=4096)
        self.agent = CodingAgent(self.settings, self.db, self.workspace, None)
        self.user_id = 123
        self.task_id = 9

    async def test_write_file_tracks_original_version_for_review(self) -> None:
        self.workspace.write_file(self.user_id, "src/app.ts", "const value = 1;\n")
        token = _CURRENT_TASK_ID.set(self.task_id)
        self.addCleanup(_CURRENT_TASK_ID.reset, token)

        await self.agent._tool_write_file(
            self.user_id,
            {"tool": "write_file", "path": "src/app.ts", "content": "const value = 2;\n"},
            [],
            None,
        )

        changes = self.workspace.list_task_file_changes(self.user_id, self.task_id)
        self.assertEqual([item["path"] for item in changes], ["src/app.ts"])
        diff = self.workspace.get_task_file_change_diff(self.user_id, self.task_id, "src/app.ts")
        self.assertIn("-const value = 1;", diff["diff"])
        self.assertIn("+const value = 2;", diff["diff"])

    async def test_apply_patch_tracks_changed_file_for_review(self) -> None:
        self.workspace.write_file(self.user_id, "src/posts.ts", "export const count = 1;\n")
        token = _CURRENT_TASK_ID.set(self.task_id)
        self.addCleanup(_CURRENT_TASK_ID.reset, token)

        await self.agent._tool_apply_patch(
            self.user_id,
            {
                "tool": "apply_patch",
                "diff": "--- src/posts.ts\n+++ src/posts.ts\n@@ -1 +1 @@\n-export const count = 1;\n+export const count = 2;\n",
            },
            [],
            None,
        )

        changes = self.workspace.list_task_file_changes(self.user_id, self.task_id)
        self.assertEqual([item["path"] for item in changes], ["src/posts.ts"])
        self.assertEqual(self.workspace.read_file(self.user_id, "src/posts.ts"), "export const count = 2;\n")

    async def test_restore_file_reverts_tracked_change_and_clears_review_entry(self) -> None:
        self.workspace.write_file(self.user_id, "src/app.ts", "const value = 1;\n")
        token = _CURRENT_TASK_ID.set(self.task_id)
        self.addCleanup(_CURRENT_TASK_ID.reset, token)

        await self.agent._tool_write_file(
            self.user_id,
            {"tool": "write_file", "path": "src/app.ts", "content": "const value = 2;\n"},
            [],
            None,
        )

        result, touched_files, _ = await self.agent._tool_restore_file(
            self.user_id,
            {"tool": "restore_file", "path": "src/app.ts"},
            [],
            None,
        )

        self.assertEqual(result, {"path": "src/app.ts", "result": "restored"})
        self.assertEqual(touched_files, ["src/app.ts"])
        self.assertEqual(self.workspace.read_file(self.user_id, "src/app.ts"), "const value = 1;\n")
        self.assertEqual(self.workspace.list_task_file_changes(self.user_id, self.task_id), [])

    async def test_restore_deleted_new_python_file_skips_compile_validation(self) -> None:
        token = _CURRENT_TASK_ID.set(self.task_id)
        self.addCleanup(_CURRENT_TASK_ID.reset, token)

        await self.agent._tool_write_file(
            self.user_id,
            {"tool": "write_file", "path": "src/new_file.py", "content": "print('ok')\n"},
            [],
            None,
        )

        await self.agent._tool_restore_file(
            self.user_id,
            {"tool": "restore_file", "path": "src/new_file.py"},
            [],
            None,
        )

        self.assertFalse(self.workspace.file_exists(self.user_id, "src/new_file.py"))
        self.assertEqual(self.agent._validate_changed_python_files(self.user_id, ["src/new_file.py"]), [])


if __name__ == "__main__":
    unittest.main()