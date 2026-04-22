from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.workspace import WorkspaceManager


class WorkspaceTaskReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workspace = WorkspaceManager(Path(self._tempdir.name), limit_bytes=4096)
        self.user_id = 123
        self.task_id = 9

    def test_stage_change_creates_snapshot_and_revert_restores_original(self) -> None:
        self.workspace.write_file(self.user_id, "src/app.ts", "const value = 1;\n")
        original_total = self.workspace.total_size(self.user_id)

        staged = self.workspace.stage_task_file_changes(self.user_id, self.task_id, ["src/app.ts"])
        self.workspace.write_file(self.user_id, "src/app.ts", "const value = 2;\n")

        self.assertEqual(staged, ["src/app.ts"])
        self.assertGreater(self.workspace.total_size(self.user_id), original_total)
        changes = self.workspace.list_task_file_changes(self.user_id, self.task_id)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["status"], "modified")

        diff = self.workspace.get_task_file_change_diff(self.user_id, self.task_id, "src/app.ts")
        self.assertIn("-const value = 1;", diff["diff"])
        self.assertIn("+const value = 2;", diff["diff"])

        self.workspace.revert_task_file_change(self.user_id, self.task_id, "src/app.ts")

        self.assertEqual(self.workspace.read_file(self.user_id, "src/app.ts"), "const value = 1;\n")
        self.assertEqual(self.workspace.list_task_file_changes(self.user_id, self.task_id), [])
        review_root = self.workspace.user_root(self.user_id) / ".agentcord" / f"task-{self.task_id}"
        self.assertFalse(review_root.exists())

    def test_accept_all_keeps_new_content_and_cleans_review_storage(self) -> None:
        self.workspace.stage_task_file_changes(self.user_id, self.task_id, ["src/new.ts"])
        self.workspace.write_file(self.user_id, "src/new.ts", "export const ok = true;\n")

        accepted = self.workspace.accept_all_task_file_changes(self.user_id, self.task_id)

        self.assertEqual(accepted, 1)
        self.assertEqual(self.workspace.read_file(self.user_id, "src/new.ts"), "export const ok = true;\n")
        self.assertEqual(self.workspace.list_task_file_changes(self.user_id, self.task_id), [])
        review_root = self.workspace.user_root(self.user_id) / ".agentcord" / f"task-{self.task_id}"
        self.assertFalse(review_root.exists())

    def test_collect_sync_candidates_ignores_agentcord_review_storage(self) -> None:
        self.workspace.write_file(self.user_id, "keep.txt", "ok\n")
        self.workspace.write_file(self.user_id, ".agentcord/task-9/manifest.json", "{}")
        self.workspace.write_file(self.user_id, ".agentcord/task-9/before/app.ts", "const old = 1;\n")

        manifest = self.workspace.collect_sync_candidates(self.user_id, ".")

        self.assertEqual([item["workspace_path"] for item in manifest["files"]], ["keep.txt"])
        skipped_paths = {item["path"] for item in manifest["skipped"]}
        self.assertIn(".agentcord", skipped_paths)

    def test_clear_task_review_storage_removes_task_snapshot_folder(self) -> None:
        self.workspace.write_file(self.user_id, ".agentcord/task-9/manifest.json", "{}")
        self.workspace.write_file(self.user_id, ".agentcord/task-9/before/app.ts", "const old = 1;\n")

        self.workspace.clear_task_review_storage(self.user_id, self.task_id)

        review_root = self.workspace.user_root(self.user_id) / ".agentcord" / f"task-{self.task_id}"
        self.assertFalse(review_root.exists())


if __name__ == "__main__":
    unittest.main()