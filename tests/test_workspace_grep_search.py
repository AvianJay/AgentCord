from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.workspace import WorkspaceManager


class WorkspaceGrepSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workspace = WorkspaceManager(Path(self._tempdir.name), limit_bytes=4096)
        self.user_id = 123

    def test_grep_search_finds_plain_text_matches_with_line_numbers(self) -> None:
        self.workspace.write_file(self.user_id, "src/app.ts", "const value = 1;\nreturn value;\n")
        self.workspace.write_file(self.user_id, "src/util.ts", "export const VALUE = 2;\n")

        result = self.workspace.grep_search(self.user_id, "value")

        self.assertEqual(
            result["matches"],
            [
                {"path": "src/app.ts", "line": 1, "text": "const value = 1;"},
                {"path": "src/app.ts", "line": 2, "text": "return value;"},
                {"path": "src/util.ts", "line": 1, "text": "export const VALUE = 2;"},
            ],
        )
        self.assertFalse(result["truncated"])

    def test_grep_search_supports_regex_and_ignores_agentcord_storage(self) -> None:
        self.workspace.write_file(self.user_id, "src/app.ts", "const userId = 1;\nconst postId = 2;\n")
        self.workspace.write_file(self.user_id, ".agentcord/task-9/manifest.json", '{"userId": true}\n')

        result = self.workspace.grep_search(self.user_id, r"\b[a-z]+Id\b", is_regex=True, max_results=1)

        self.assertEqual(result["matches"], [{"path": "src/app.ts", "line": 1, "text": "const userId = 1;"}])
        self.assertTrue(result["truncated"])


if __name__ == "__main__":
    unittest.main()