from __future__ import annotations

import unittest
from types import SimpleNamespace

from agentcord.live_agent import AgentTaskReviewPopupView


class _FakeReviewSession:
    def __init__(self) -> None:
        self.user = SimpleNamespace(id=123)
        self._changes = [
            {"path": "src/controllers/posts.ts"},
            {"path": "src/routes/posts.ts"},
        ]

    def list_pending_file_changes(self) -> list[dict[str, str]]:
        return list(self._changes)

    def get_pending_file_change_diff(self, path: str) -> dict[str, str]:
        return {
            "path": path,
            "status": "modified",
            "diff": f"--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n-old\n+new\n",
        }

    def is_busy(self) -> bool:
        return False

    def _shorten(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."


class LiveAgentReviewViewTests(unittest.TestCase):
    def test_review_popup_uses_file_select_options(self) -> None:
        view = AgentTaskReviewPopupView(_FakeReviewSession())

        self.assertEqual([option.label for option in view._file_select.options], [
            "src/controllers/posts.ts",
            "src/routes/posts.ts",
        ])
        self.assertEqual(view._file_select.options[0].value, "path:0")
        self.assertTrue(view._file_select.options[0].default)
        self.assertFalse(view._file_select.disabled)

    def test_render_message_follows_selected_file_index(self) -> None:
        view = AgentTaskReviewPopupView(_FakeReviewSession())
        view._sync_paths(preferred_path="src/routes/posts.ts")
        view._sync_buttons()

        message = view.render_message()

        self.assertIn("待確認變更 2/2", message)
        self.assertIn("檔案：src/routes/posts.ts", message)


if __name__ == "__main__":
    unittest.main()