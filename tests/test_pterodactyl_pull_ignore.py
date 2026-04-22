from __future__ import annotations

import unittest
from unittest.mock import patch

from agentcord import pterodactyl


class PterodactylPullIgnoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_collect_remote_files_hard_skips_default_directories_before_recursing(self) -> None:
        visited: list[str] = []

        async def fake_list(session, settings, config, server, directory_path):
            visited.append(directory_path)
            if directory_path == "/":
                return [
                    {"name": "node_modules", "kind": "folder", "size": 0, "mimetype": "inode/directory"},
                    {"name": "__pycache__", "kind": "folder", "size": 0, "mimetype": "inode/directory"},
                    {"name": "src", "kind": "folder", "size": 0, "mimetype": "inode/directory"},
                ]
            if directory_path == "/src":
                return [
                    {"name": "index.js", "kind": "file", "size": 20, "mimetype": "text/javascript"},
                ]
            raise AssertionError(f"unexpected recursion into {directory_path}")

        with patch.object(pterodactyl, "list_pterodactyl_server_directory", side_effect=fake_list):
            manifest = await pterodactyl.collect_pterodactyl_server_files(
                None,
                None,
                None,
                "srv",
                "/",
            )

        self.assertEqual(visited, ["/", "/src"])
        self.assertEqual([item["relative_path"] for item in manifest["files"]], ["src/index.js"])
        self.assertEqual(
            manifest["skipped"],
            [
                {"path": "/node_modules", "reason": "ignored"},
                {"path": "/__pycache__", "reason": "ignored"},
            ],
        )

    async def test_collect_remote_files_skips_ignored_directories_before_recursing(self) -> None:
        visited: list[str] = []

        async def fake_list(session, settings, config, server, directory_path):
            visited.append(directory_path)
            if directory_path == "/":
                return [
                    {"name": "node_modules", "kind": "folder", "size": 0, "mimetype": "inode/directory"},
                    {"name": "src", "kind": "folder", "size": 0, "mimetype": "inode/directory"},
                    {"name": "package.json", "kind": "file", "size": 10, "mimetype": "application/json"},
                ]
            if directory_path == "/src":
                return [
                    {"name": "index.js", "kind": "file", "size": 20, "mimetype": "text/javascript"},
                ]
            raise AssertionError(f"unexpected recursion into {directory_path}")

        with patch.object(pterodactyl, "list_pterodactyl_server_directory", side_effect=fake_list):
            manifest = await pterodactyl.collect_pterodactyl_server_files(
                None,
                None,
                None,
                "srv",
                "/",
                ignore_patterns=["node_modules"],
            )

        self.assertEqual(visited, ["/", "/src"])
        self.assertEqual(
            [item["relative_path"] for item in manifest["files"]],
            ["src/index.js", "package.json"],
        )
        self.assertEqual(manifest["skipped"], [{"path": "/node_modules", "reason": "ignored"}])


if __name__ == "__main__":
    unittest.main()