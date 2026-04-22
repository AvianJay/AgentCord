from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from agentcord.live_agent import AgentConversationSession
from agentcord.models import TaskRecord, TaskStatus


class _FakeWorkspace:
    def list_task_file_changes(self, user_id: int, task_id: int) -> list[dict[str, str]]:
        del user_id, task_id
        return []


class _FakeBot:
    def __init__(self, *, fail_log_event: bool = False) -> None:
        self.workspace = _FakeWorkspace()
        self.fail_log_event = fail_log_event
        self.logged_events: list[tuple[str, str]] = []
        self.logged_exceptions: list[tuple[str, str]] = []

    async def log_event(self, title: str, description: str, **kwargs) -> None:
        del kwargs
        if self.fail_log_event:
            raise RuntimeError("webhook unavailable")
        self.logged_events.append((title, description))

    async def log_exception(self, title: str, error: BaseException, **kwargs) -> None:
        del kwargs
        self.logged_exceptions.append((title, str(error)))


class _RecordingSession(AgentConversationSession):
    def __init__(self, bot: _FakeBot) -> None:
        task = TaskRecord(
            id=9,
            user_id=123,
            title="task",
            status=TaskStatus.DONE,
            related_files=[],
        )
        user = SimpleNamespace(id=123)
        super().__init__(bot, user, task)
        self.worker_started = asyncio.Event()

    async def request_render(self, *, force: bool = False) -> None:
        del force
        return

    async def _worker_loop(self) -> None:
        self.worker_started.set()


class LiveAgentResilienceTests(unittest.IsolatedAsyncioTestCase):
    async def test_enqueue_prompt_starts_worker_even_if_logging_fails(self) -> None:
        session = _RecordingSession(_FakeBot(fail_log_event=True))

        await session.enqueue_prompt("hello world")
        await asyncio.wait_for(session.worker_started.wait(), timeout=1)

        self.assertEqual(session._prompt_queue.qsize(), 1)
        self.assertTrue(any(entry.text == "hello world" for entry in session._conversation_entries))

    async def test_tool_result_progress_is_logged(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)

        await session.handle_progress(
            {
                "type": "tool_result",
                "tool": "read_file",
                "status": "ok",
                "preview": '{"path": "src/app.ts"}',
            }
        )

        self.assertEqual(len(bot.logged_events), 1)
        title, description = bot.logged_events[0]
        self.assertEqual(title, "Agent Tool")
        self.assertIn("Tool: read_file", description)
        self.assertIn('src/app.ts', description)


if __name__ == "__main__":
    unittest.main()