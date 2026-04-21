from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import discord

from agentcord.models import AgentTaskItem, TaskRecord, TaskStatus

if TYPE_CHECKING:
    from agentcord.bot import AgentCordBot


_MESSAGE_LIMIT = 2000


@dataclass(slots=True)
class ContextWindowState:
    model: str = ""
    context_length: int | None = None
    estimated_tokens: int = 0
    compression_count: int = 0
    history_messages: int = 0
    phase: str = "idle"


class AgentConversationSession:
    def __init__(self, bot: AgentCordBot, user: discord.abc.User, task: TaskRecord) -> None:
        self.bot = bot
        self.user = user
        self.task_record = task
        self.guild: discord.Guild | None = None
        self.task_items = list(task.task_items)
        self.context_state = ContextWindowState(
            model=task.model,
            context_length=task.context_length,
            compression_count=task.compression_count,
            history_messages=len(task.messages),
        )
        self.message: discord.Message | None = None
        self.view = AgentConversationView(self)
        self._activity_lines: deque[str] = deque()
        self._prompt_queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._current_run_task: asyncio.Task[object] | None = None
        self._render_task: asyncio.Task[None] | None = None
        self._render_lock = asyncio.Lock()
        self._closed = False
        if task.summary:
            self._append_activity(f"目前摘要：{task.summary}")

    @property
    def task_id(self) -> int:
        return self.task_record.id

    def is_busy(self) -> bool:
        return (self._current_run_task is not None and not self._current_run_task.done()) or not self._prompt_queue.empty()

    async def open(self, interaction: discord.Interaction, *, reopened: bool = False) -> None:
        self.guild = interaction.guild
        if reopened:
            self._append_activity(f"已重新打開對話 #{self.task_record.id}。")
        self.view.sync_layout()
        await interaction.response.send_message(view=self.view)
        self.message = await interaction.original_response()

    async def close(self, reason: str | None = None) -> None:
        self._closed = True
        if reason:
            self._append_activity(reason)
        self.view.disable_all_items()
        self.view.stop()
        await self.request_render(force=True)

    async def enqueue_prompt(self, prompt: str) -> None:
        if self._closed:
            raise ValueError("這個對話已關閉，請重新開啟新的對話。")
        await self._prompt_queue.put(prompt)
        await self.bot.log_event(
            "Agent 對話訊息",
            f"送出 agent 對話訊息。\nTask ID: {self.task_record.id}\nPrompt: {self._shorten(' '.join(prompt.split()), 300)}",
            user=self.user,
            guild=self.guild,
        )
        if self._current_run_task is not None and not self._current_run_task.done():
            self._append_activity("已將新的使用者訊息加入佇列。")
        else:
            self._append_activity("已收到新的使用者訊息，準備開始處理。")
        await self.request_render()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def interrupt(self) -> None:
        while not self._prompt_queue.empty():
            try:
                self._prompt_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._current_run_task is None or self._current_run_task.done():
            self._append_activity("目前沒有正在執行的 agent。")
            await self.request_render(force=True)
            return
        self._append_activity("已要求中斷目前的 agent 執行。")
        await self.bot.log_event(
            "Agent 中斷",
            f"要求中斷 agent 對話。\nTask ID: {self.task_record.id}",
            user=self.user,
            guild=self.guild,
            color=discord.Colour.orange(),
        )
        self._current_run_task.cancel()
        await self.request_render(force=True)

    async def handle_progress(self, event: dict[str, object]) -> None:
        event_type = str(event.get("type", ""))
        if event_type == "activity":
            self._append_activity(str(event.get("message", "")))
        elif event_type == "tasks":
            self.task_items = [
                AgentTaskItem(
                    title=str(item.get("title", "")),
                    status=str(item.get("status", "pending")),
                )
                for item in event.get("items", [])
                if isinstance(item, dict) and str(item.get("title", "")).strip()
            ]
        elif event_type == "context":
            model = event.get("model")
            if isinstance(model, str):
                self.context_state.model = model
            context_length = event.get("context_length")
            if isinstance(context_length, int):
                self.context_state.context_length = context_length
            estimated_tokens = event.get("estimated_tokens")
            if isinstance(estimated_tokens, int):
                self.context_state.estimated_tokens = estimated_tokens
            compression_count = event.get("compression_count")
            if isinstance(compression_count, int):
                self.context_state.compression_count = compression_count
            history_messages = event.get("history_messages")
            if isinstance(history_messages, int):
                self.context_state.history_messages = history_messages
            phase = event.get("phase")
            if isinstance(phase, str):
                self.context_state.phase = phase
        await self.request_render()

    async def request_render(self, *, force: bool = False) -> None:
        if self.message is None:
            return
        if force:
            if self._render_task is not None and not self._render_task.done():
                self._render_task.cancel()
                self._render_task = None
            await self._render_now()
            return
        if self._render_task is not None and not self._render_task.done():
            return
        self._render_task = asyncio.create_task(self._delayed_render())

    async def _delayed_render(self) -> None:
        try:
            await asyncio.sleep(0.6)
            await self._render_now()
        except asyncio.CancelledError:
            return
        finally:
            self._render_task = None

    async def _render_now(self) -> None:
        if self.message is None:
            return
        async with self._render_lock:
            self.view.sync_layout()
            try:
                await self.message.edit(content=None, view=self.view)
            except discord.HTTPException:
                return

    async def _worker_loop(self) -> None:
        while not self._prompt_queue.empty() and not self._closed:
            prompt = await self._prompt_queue.get()
            self._append_activity(f"開始處理使用者訊息：{prompt}")
            await self.request_render(force=True)
            try:
                self._current_run_task = asyncio.create_task(
                    self.bot.agent.run(
                        self.user.id,
                        prompt,
                        task=self.task_record,
                        progress_callback=self.handle_progress,
                    )
                )
                result = await self._current_run_task
            except asyncio.CancelledError:
                self.task_record = self.bot.db.get_task_by_id(self.task_record.id)
                self.task_record = self.bot.db.update_task(
                    self.task_record.id,
                    TaskStatus.CANCELLED,
                    self.task_record.related_files,
                    summary=self.task_record.summary or "已中斷目前執行。",
                    plan=self.task_record.plan,
                    validations=self.task_record.validations,
                    messages=self.task_record.messages,
                    task_items=self.task_items,
                    model=self.context_state.model or self.task_record.model,
                    context_length=self.context_state.context_length,
                    compression_count=self.context_state.compression_count,
                )
                self._append_activity("本輪執行已被中斷。")
                await self.bot.log_event(
                    "Agent 已中斷",
                    f"agent 對話已中斷。\nTask ID: {self.task_record.id}",
                    user=self.user,
                    guild=self.guild,
                    color=discord.Colour.orange(),
                )
            except Exception as exc:  # noqa: BLE001
                self.task_record = self.bot.db.get_task_by_id(self.task_record.id)
                self.task_record = self.bot.db.update_task(
                    self.task_record.id,
                    TaskStatus.FAILED,
                    self.task_record.related_files,
                    summary=f"執行失敗：{exc}",
                    plan=self.task_record.plan,
                    validations=self.task_record.validations,
                    messages=self.task_record.messages,
                    task_items=self.task_items,
                    model=self.context_state.model or self.task_record.model,
                    context_length=self.context_state.context_length,
                    compression_count=self.context_state.compression_count,
                )
                self._append_activity(f"本輪執行失敗：{exc}")
                await self.bot.log_exception(
                    "Agent 執行失敗",
                    exc,
                    user=self.user,
                    guild=self.guild,
                    details=f"Task ID: {self.task_record.id}",
                )
            else:
                self.task_record = self.bot.db.get_task(self.user.id, result.task_id or self.task_record.id)
                self.task_items = list(result.task_items)
                self.context_state.model = result.model
                self.context_state.context_length = result.context_length
                self.context_state.estimated_tokens = result.estimated_tokens
                self.context_state.compression_count = result.compression_count
                self.context_state.history_messages = len(result.messages)
                self.context_state.phase = "completed"
                self._append_activity(f"本輪完成：{result.summary}")
                await self.bot.log_event(
                    "Agent 完成",
                    f"agent 對話完成。\nTask ID: {self.task_record.id}\nSummary: {self._shorten(result.summary, 300)}",
                    user=self.user,
                    guild=self.guild,
                    fields=[
                        ("檔案數", str(len(result.related_files)), True),
                        ("驗證數", str(len(result.validations)), True),
                    ],
                    color=discord.Colour.green(),
                )
            finally:
                self._current_run_task = None
                await self.request_render(force=True)

    def render_content(self) -> str:
        return self._render_activity_display()

    def _render_activity_display(self) -> str:
        activity_lines = [f"-# {line}" for line in self._activity_lines] or ["-# 等待新的 agent 訊息。"]
        content = self._compose_activity_display(activity_lines)
        while len(content) > _MESSAGE_LIMIT and len(activity_lines) > 1:
            activity_lines = activity_lines[1:]
            content = self._compose_activity_display(activity_lines)
        if len(content) > _MESSAGE_LIMIT:
            return content[: _MESSAGE_LIMIT - 3] + "..."
        return content

    def _compose_activity_display(self, activity_lines: list[str]) -> str:
        parts = ["AI 在幹什麼", *activity_lines]
        return "\n".join(parts)

    def _render_tasks_block(self) -> str:
        header = ["tasks"]
        if not self.task_items:
            return "\n".join([*header, "(無)"])
        status_map = {
            "pending": "[ ]",
            "todo": "[ ]",
            "in_progress": "[>]",
            "running": "[>]",
            "doing": "[>]",
            "done": "[x]",
            "completed": "[x]",
            "cancelled": "[-]",
        }
        lines = [
            f"{status_map.get(item.status, '[ ]')} {self._shorten(item.title, 110)}"
            for item in self.task_items[:12]
        ]
        return "\n".join([*header, *lines])

    def _render_context_block(self) -> str:
        context_limit = self.context_state.context_length or 0
        context_text = (
            f"{self.context_state.estimated_tokens} / {context_limit} tokens"
            if context_limit
            else f"{self.context_state.estimated_tokens} / ? tokens"
        )
        lines = [
            "context manager window",
            f"模型：{self.context_state.model or self.task_record.model or '(未設定)'}",
            f"上下文：{context_text}",
            f"壓縮次數：{self.context_state.compression_count}",
            f"歷史訊息：{self.context_state.history_messages}",
            f"階段：{self.context_state.phase}",
        ]
        return "\n".join(lines)

    def _render_operations_block(self) -> str:
        lines = [
            "要進行的操作",
            f"會話 ID：#{self.task_record.id}",
            f"狀態：{self._status_label()}",
            f"排隊訊息：{self._prompt_queue.qsize()}",
            "可用按鈕：中斷 / 傳送訊息 / 重新整理",
        ]
        return "\n".join(lines)

    def _status_label(self) -> str:
        if self._current_run_task is not None and not self._current_run_task.done():
            return "執行中"
        if self.task_record.status is TaskStatus.DONE:
            return "已完成"
        if self.task_record.status is TaskStatus.CANCELLED:
            return "已中斷"
        if self.task_record.status is TaskStatus.FAILED:
            return "失敗"
        if self.task_record.status is TaskStatus.RUNNING:
            return "執行中"
        return "待命"

    def _append_activity(self, text: str) -> None:
        normalized = " ".join(str(text).split())
        if not normalized:
            return
        normalized = self._shorten(normalized, 180)
        if self._activity_lines and self._activity_lines[-1] == normalized:
            return
        self._activity_lines.append(normalized)

    def _shorten(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."


class AgentConversationView(discord.ui.LayoutView):
    def __init__(self, session: AgentConversationSession) -> None:
        super().__init__(timeout=None)
        self.session = session
        self._activity_display = discord.ui.TextDisplay("AI 在幹什麼\n-# 等待新的 agent 訊息。")
        self._tasks_display = discord.ui.TextDisplay("tasks\n(無)")
        self._context_display = discord.ui.TextDisplay("context manager window\n模型：(未設定)")
        self._operations_display = discord.ui.TextDisplay("要進行的操作\n會話 ID：#0")
        self._interrupt_button = discord.ui.Button(label="中斷", style=discord.ButtonStyle.danger)
        self._interrupt_button.callback = self._on_interrupt
        self._send_message_button = discord.ui.Button(label="傳送訊息", style=discord.ButtonStyle.primary)
        self._send_message_button.callback = self._on_send_message
        self._refresh_button = discord.ui.Button(label="重新整理", style=discord.ButtonStyle.secondary)
        self._refresh_button.callback = self._on_refresh

        self.add_item(self._activity_display)
        self.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.large))
        self.add_item(self._tasks_display)
        self.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.large))
        self.add_item(self._context_display)
        self.add_item(discord.ui.Separator(spacing=discord.SeparatorSpacing.large))
        self.add_item(self._operations_display)
        self.add_item(
            discord.ui.ActionRow(
                self._interrupt_button,
                self._send_message_button,
                self._refresh_button,
            )
        )

    async def _on_interrupt(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。")
            return
        await interaction.response.defer()
        await self.session.interrupt()

    async def _on_send_message(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。")
            return
        await interaction.response.send_modal(AgentMessageModal(self.session))

    async def _on_refresh(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。")
            return
        await interaction.response.defer()
        await self.session.request_render(force=True)

    def sync_buttons(self) -> None:
        self._interrupt_button.disabled = not self.session.is_busy()

    def sync_layout(self) -> None:
        self._activity_display.content = self.session._render_activity_display()
        self._tasks_display.content = self.session._render_tasks_block()
        self._context_display.content = self.session._render_context_block()
        self._operations_display.content = self.session._render_operations_block()
        self.sync_buttons()


class AgentMessageModal(discord.ui.Modal):
    prompt = discord.ui.TextInput(
        label="傳送給 agent 的訊息",
        placeholder="輸入新的需求、補充限制或追問...",
        style=discord.TextStyle.paragraph,
        max_length=1500,
    )

    def __init__(self, session: AgentConversationSession) -> None:
        super().__init__(title="傳送訊息")
        self.session = session

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        await self.session.enqueue_prompt(str(self.prompt))