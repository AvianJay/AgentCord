from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.workspace import WorkspaceError, WorkspaceManager


class WorkspacePullLimitTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.workspace = WorkspaceManager(Path(self._tempdir.name), limit_bytes=10)
        self.user_id = 123

    def test_collect_remote_sync_targets_rejects_projected_total_over_limit(self) -> None:
        self.workspace.write_file(self.user_id, "existing.txt", "12345")

        remote_files = [
            {
                "relative_path": "incoming.txt",
                "remote_path": "/incoming.txt",
                "size": 6,
                "mimetype": "text/plain",
            }
        ]

        with self.assertRaisesRegex(WorkspaceError, "拉取後工作區將超過 10 位元組上限"):
            self.workspace.collect_remote_sync_targets(
                self.user_id,
                ".",
                remote_files=remote_files,
            )

    def test_write_file_still_rejects_if_actual_pull_content_exceeds_limit(self) -> None:
        self.workspace.write_file(self.user_id, "existing.txt", "12345")

        manifest = self.workspace.collect_remote_sync_targets(
            self.user_id,
            ".",
            remote_files=[
                {
                    "relative_path": "incoming.txt",
                    "remote_path": "/incoming.txt",
                    "size": 4,
                    "mimetype": "text/plain",
                }
            ],
        )

        with self.assertRaisesRegex(WorkspaceError, "寫入遭拒：工作區將超過 10 位元組上限"):
            self.workspace.write_file(self.user_id, manifest["files"][0]["workspace_path"], "123456")


if __name__ == "__main__":
    unittest.main()