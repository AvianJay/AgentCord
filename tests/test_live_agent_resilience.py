from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace
from unittest import mock
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
        self.interaction_messages: list[tuple[str, bool]] = []
        self.db = _FakeDB()

    async def log_event(self, title: str, description: str, **kwargs) -> None:
        del kwargs
        if self.fail_log_event:
            raise RuntimeError("webhook unavailable")
        self.logged_events.append((title, description))

    async def log_exception(self, title: str, error: BaseException, **kwargs) -> None:
        del kwargs
        self.logged_exceptions.append((title, str(error)))

    async def send_interaction_message(self, interaction, message: str | None = None, *, ephemeral: bool = False, **kwargs) -> None:
        del interaction, kwargs
        self.interaction_messages.append((message or "", ephemeral))


class _FakeDB:
    def __init__(self) -> None:
        self.task: TaskRecord | None = None

    def get_task_by_id(self, task_id: int) -> TaskRecord:
        assert self.task is not None
        self.assert_task_id(task_id)
        return self.task

    def update_task(self, task_id: int, status: TaskStatus, related_files: list[str], **kwargs) -> TaskRecord:
        assert self.task is not None
        self.assert_task_id(task_id)
        self.task = replace(
            self.task,
            status=status,
            related_files=list(related_files),
            summary=kwargs.get("summary", self.task.summary),
            plan=list(kwargs.get("plan", self.task.plan)),
            validations=list(kwargs.get("validations", self.task.validations)),
            messages=list(kwargs.get("messages", self.task.messages)),
            task_items=list(kwargs.get("task_items", self.task.task_items)),
            model=kwargs.get("model", self.task.model),
            context_length=kwargs.get("context_length", self.task.context_length),
            compression_count=kwargs.get("compression_count", self.task.compression_count),
        )
        return self.task

    def assert_task_id(self, task_id: int) -> None:
        assert self.task is not None
        if self.task.id != task_id:
            raise AssertionError(f"unexpected task id: {task_id}")


class _FakeChannelMessage:
    def __init__(self, message_id: int, channel) -> None:
        self.id = message_id
        self.channel = channel

    async def edit(self, *, content=None, view=None) -> None:
        del content, view


class _FakeInteractionMessage(_FakeChannelMessage):
    pass


class _FakeChannel:
    def __init__(self, fetched_message: _FakeChannelMessage) -> None:
        self.fetched_message = fetched_message
        self.fetch_calls: list[int] = []

    async def fetch_message(self, message_id: int) -> _FakeChannelMessage:
        self.fetch_calls.append(message_id)
        return self.fetched_message


class _FakeRunningTask:
    def __init__(self) -> None:
        self.cancel_called = False

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        self.cancel_called = True


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
        bot.db.task = task
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

    async def test_interaction_exception_uses_bot_fallback_sender(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)
        interaction = SimpleNamespace(response=SimpleNamespace(is_done=lambda: True))

        await session.handle_interaction_exception("AgentConversationView", RuntimeError("boom"), interaction)

        self.assertEqual(bot.interaction_messages, [("互動處理失敗：boom", True)])

    async def test_open_rehydrates_original_response_to_normal_message(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)
        channel_message = _FakeChannelMessage(42, None)
        channel = _FakeChannel(channel_message)
        channel_message.channel = channel
        interaction_message = _FakeInteractionMessage(42, channel)
        response = SimpleNamespace(send_message=self._async_noop)
        interaction = SimpleNamespace(
            guild=None,
            response=response,
            original_response=self._async_return(interaction_message),
        )

        with mock.patch("agentcord.live_agent.discord.InteractionMessage", _FakeInteractionMessage):
            await session.open(interaction)

        self.assertIs(session.message, channel_message)
        self.assertEqual(channel.fetch_calls, [42])

    async def test_handle_session_timeout_closes_idle_session_with_reopen_notice(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)

        await session.handle_session_timeout()

        self.assertTrue(session._closed)
        self.assertTrue(any("/agent-open 9" in entry.text for entry in session._activity_lines))

    async def test_handle_session_timeout_pauses_busy_session(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)
        session._current_run_task = _FakeRunningTask()
        await session._prompt_queue.put("queued prompt")

        await session.handle_session_timeout()

        self.assertTrue(session._timeout_pause_requested)
        self.assertEqual(session._prompt_queue.qsize(), 0)
        self.assertTrue(session._current_run_task.cancel_called)
        self.assertFalse(session._closed)
        self.assertTrue(any("/agent-open 9" in entry.text for entry in session._activity_lines))

    async def test_finalize_timeout_pause_marks_task_pending(self) -> None:
        bot = _FakeBot()
        session = _RecordingSession(bot)
        session.task_record = replace(session.task_record, status=TaskStatus.RUNNING, summary="working")
        bot.db.task = session.task_record
        session._timeout_pause_requested = True

        await session._finalize_timeout_pause()

        self.assertEqual(session.task_record.status, TaskStatus.PENDING)
        self.assertEqual(session.task_record.summary, "working")
        self.assertTrue(session._closed)
        self.assertEqual(bot.logged_events[-1][0], "Agent 暫停")

    @staticmethod
    async def _async_noop(*args, **kwargs) -> None:
        del args, kwargs

    @staticmethod
    def _async_return(value):
        async def _inner():
            return value

        return _inner


if __name__ == "__main__":
    unittest.main()