from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agentcord.bot import WorkspaceExplorerView
from agentcord.config import Settings
from agentcord.workspace import WorkspaceManager


class _FakeBot:
    def __init__(self, root: Path) -> None:
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
        self.workspace = WorkspaceManager(root / "workspaces", self.settings.workspace_limit_bytes)


class WorkspaceExplorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.bot = _FakeBot(root)
        self.user = SimpleNamespace(id=123)
        self.bot.workspace.create_folder(self.user.id, "src")
        self.bot.workspace.write_file(self.user.id, "src/app.py", "print('ok')\n")

    def test_open_selected_folder_moves_into_directory(self) -> None:
        view = WorkspaceExplorerView(self.bot, self.user)
        view.selected_path = "src"

        entry = view.open_selected()

        self.assertEqual(entry.kind, "folder")
        self.assertEqual(view.current_path, "src")
        self.assertEqual([item.path for item in view.entries], ["src/app.py"])

    def test_write_file_from_current_path_selects_written_file(self) -> None:
        view = WorkspaceExplorerView(self.bot, self.user, start_path="src")

        written_path = view.write_file("notes.txt", "hello")

        self.assertEqual(written_path, "src/notes.txt")
        self.assertEqual(view.current_path, "src")
        self.assertEqual(view.selected_path, "src/notes.txt")
        self.assertEqual(self.bot.workspace.read_file(self.user.id, "src/notes.txt"), "hello")

    def test_delete_selected_file_removes_it_from_workspace(self) -> None:
        view = WorkspaceExplorerView(self.bot, self.user, start_path="src")
        view.selected_path = "src/app.py"

        removed_path = view.delete_selected()

        self.assertEqual(removed_path, "src/app.py")
        self.assertFalse(self.bot.workspace.file_exists(self.user.id, "src/app.py"))
        self.assertIsNone(view.selected_path)
        self.assertEqual(view.entries, [])


if __name__ == "__main__":
    unittest.main()