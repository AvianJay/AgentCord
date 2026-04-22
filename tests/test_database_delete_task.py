from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.database import Database
from agentcord.models import TaskStatus


class DatabaseDeleteTaskTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.db = Database(Path(self._tempdir.name) / "agentcord.db", default_credits=100)
        self.addCleanup(self.db.close)

    def test_delete_task_removes_only_matching_user_task(self) -> None:
        kept_task = self.db.create_task(2, "other user task", TaskStatus.PENDING)
        deleted_task = self.db.create_task(1, "delete me", TaskStatus.DONE)

        result = self.db.delete_task(1, deleted_task.id)

        self.assertEqual(result.id, deleted_task.id)
        self.assertEqual(result.title, "delete me")
        with self.assertRaisesRegex(ValueError, f"找不到任務 {deleted_task.id}。"):
            self.db.get_task(1, deleted_task.id)
        self.assertEqual(self.db.get_task(2, kept_task.id).title, "other user task")


if __name__ == "__main__":
    unittest.main()