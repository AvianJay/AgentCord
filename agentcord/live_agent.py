from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import discord

from agentcord.models import AgentTaskItem, ConversationMessage, TaskRecord, TaskStatus

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


@dataclass(slots=True)
class ActivityEntry:
    text: str
    key: str | None = None
    transient: bool = False


@dataclass(slots=True)
class ConversationEntry:
    role: str
    text: str


@dataclass(slots=True)
class DisplayEntry:
    kind: str
    text: str
    key: str | None = None
    transient: bool = False


@dataclass(slots=True)
class ChoiceOptionEntry:
    label: str
    value: str
    description: str = ""


@dataclass(slots=True)
class PendingChoiceState:
    prompt: str
    options: list[ChoiceOptionEntry]
    placeholder: str
    future: asyncio.Future[dict[str, Any]]
    min_values: int = 1
    max_values: int = 1
    allow_freeform: bool = False
    freeform_placeholder: str = ""


class AgentConversationSession:
    def __init__(self, bot: AgentCordBot, user: discord.abc.User, task: TaskRecord) -> None:
        self.bot = bot
        self.user = user
        self.task_record = task
        self.guild: discord.Guild | None = None
        self.task_items = list(task.task_items)
        self._conversation_entries: deque[ConversationEntry] = deque(self._load_conversation_entries(task.messages))
        self._display_entries: deque[DisplayEntry] = deque(self._load_display_entries(task.messages))
        self._pending_choice: PendingChoiceState | None = None
        self.context_state = ContextWindowState(
            model=task.model,
            context_length=task.context_length,
            compression_count=task.compression_count,
            history_messages=len(task.messages),
        )
        self.message: discord.Message | None = None
        self.view = AgentConversationView(self)
        self._activity_lines: deque[ActivityEntry] = deque()
        self._prompt_queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._current_run_task: asyncio.Task[object] | None = None
        self._render_task: asyncio.Task[None] | None = None
        self._render_lock = asyncio.Lock()
        self._closed = False
        self._run_sequence = 0
        self._active_run_scope: str | None = None
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
        self._cancel_pending_choice("對話已關閉。")
        if reason:
            self._append_activity(reason)
        self.view.disable_all_items()
        self.view.stop()
        await self.request_render(force=True)

    async def enqueue_prompt(self, prompt: str) -> None:
        normalized_prompt = str(prompt).strip()
        if not normalized_prompt:
            raise ValueError("訊息不可為空白。")
        if self._closed:
            raise ValueError("這個對話已關閉，請重新開啟新的對話。")
        self._append_conversation("user", normalized_prompt)
        await self._prompt_queue.put(normalized_prompt)
        await self.bot.log_event(
            "Agent 對話訊息",
            f"送出 agent 對話訊息。\nTask ID: {self.task_record.id}\nPrompt: {self._shorten(' '.join(normalized_prompt.split()), 300)}",
            user=self.user,
            guild=self.guild,
        )
        if self._current_run_task is not None and not self._current_run_task.done():
            self._append_activity("已將新的使用者訊息加入佇列。")
        await self.request_render()
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def interrupt(self) -> None:
        self._cancel_pending_choice("已取消等待使用者選擇。")
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

    async def handle_progress(self, event: dict[str, object]) -> object | None:
        event_type = str(event.get("type", ""))
        if event_type == "activity":
            raw_key = str(event["activity_key"]) if isinstance(event.get("activity_key"), str) else None
            scoped_key = self._scope_activity_key(raw_key)
            message = str(event.get("message", ""))
            if raw_key is not None and raw_key.startswith("decision:") and "決策已完成" in message:
                self._remove_activity(scoped_key)
            else:
                self._append_activity(
                    message,
                    key=scoped_key,
                    transient=bool(event.get("transient", False)),
                )
        elif event_type == "activity_remove":
            raw_key = str(event["activity_key"]) if isinstance(event.get("activity_key"), str) else None
            self._remove_activity(self._scope_activity_key(raw_key))
        elif event_type == "chat_message":
            role = str(event.get("role") or "assistant")
            message = str(event.get("message") or "")
            self._append_conversation(role, message)
        elif event_type == "choice_request":
            return await self._request_user_choice(event)
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
        return None

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
            self._run_sequence += 1
            self._active_run_scope = f"run:{self._run_sequence}"
            self._append_activity("開始處理新的使用者訊息。")
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
                    messages=self._build_persisted_messages(),
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
                    messages=self._build_persisted_messages(),
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
                self._append_conversation("assistant", result.summary)
                self._append_activity("本輪已完成。")
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
                self._active_run_scope = None
                await self.request_render(force=True)

    def render_content(self) -> str:
        return self._render_main_display()

    def _render_main_display(self) -> str:
        lines = [entry.text for entry in self._display_entries] or ["-# 等待新的 agent 訊息。"]
        content = self._compose_main_display(lines)
        while len(content) > _MESSAGE_LIMIT and len(lines) > 1:
            lines = lines[1:]
            content = self._compose_main_display(lines)
        if len(content) > _MESSAGE_LIMIT:
            return content[: _MESSAGE_LIMIT - 3] + "..."
        return content

    def _compose_main_display(self, lines: list[str]) -> str:
        return "\n".join(lines)

    def _render_choice_block(self) -> str:
        if self._pending_choice is None:
            return "## 請選擇\n(目前沒有待選項目)"
        lines = ["## 請選擇", self._shorten(" ".join(self._pending_choice.prompt.split()), 180)]
        if self._pending_choice.options:
            if self._pending_choice.min_values == self._pending_choice.max_values:
                lines.append(f"需要選擇 {self._pending_choice.min_values} 項。")
            else:
                lines.append(f"可選擇 {self._pending_choice.min_values} 到 {self._pending_choice.max_values} 項。")
            for index, option in enumerate(self._pending_choice.options, start=1):
                option_line = f"{index}. {self._shorten(option.label, 90)}"
                if option.description:
                    option_line = f"{option_line} | {self._shorten(option.description, 70)}"
                lines.append(option_line)
        else:
            lines.append("(此題沒有預設選項，請改用下方自由輸入。)")
        if self._pending_choice.allow_freeform:
            lines.append("可改用自由輸入。")
        return "\n".join(lines)

    def _render_tasks_block(self) -> str:
        header = ["## 待辦事項"]
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
            f"-# {self.context_state.model or self.task_record.model or '(未設定)'}",
            f"-# {context_text} | {self.context_state.phase}",
            # f"-# 壓縮次數：{self.context_state.compression_count}",
            # f"-# 歷史訊息：{self.context_state.history_messages}",
            # f"-# 階段：{self.context_state.phase}",
        ]
        return "\n".join(lines)

    def _render_operations_block(self) -> str:
        lines = [
            f"-# #{self.task_record.id} | {self._status_label()} | {self._prompt_queue.qsize()} 則訊息待送出",
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

    def _load_conversation_entries(self, messages: list[ConversationMessage]) -> list[ConversationEntry]:
        entries: list[ConversationEntry] = []
        for message in messages:
            if message.role not in {"user", "assistant"}:
                continue
            normalized = str(message.content).strip()
            if not normalized:
                continue
            entries.append(ConversationEntry(role=message.role, text=normalized))
        return entries

    def _load_display_entries(self, messages: list[ConversationMessage]) -> list[DisplayEntry]:
        entries: list[DisplayEntry] = []
        for message in messages:
            if message.role not in {"user", "assistant"}:
                continue
            normalized = str(message.content).strip()
            if not normalized:
                continue
            entries.append(
                DisplayEntry(
                    kind="conversation",
                    text=self._format_conversation_text(message.role, normalized),
                )
            )
        return entries

    def _build_persisted_messages(self) -> list[ConversationMessage]:
        system_messages = [message for message in self.task_record.messages if message.role == "system"]
        conversation_messages = [
            ConversationMessage(role=entry.role, content=entry.text)
            for entry in self._conversation_entries
        ]
        return [*system_messages, *conversation_messages]

    def _append_conversation(self, role: str, text: str) -> None:
        normalized = str(text).strip()
        if not normalized or role not in {"user", "assistant"}:
            return
        if self._conversation_entries:
            last_entry = self._conversation_entries[-1]
            if last_entry.role == role and last_entry.text == normalized:
                return
        self._conversation_entries.append(ConversationEntry(role=role, text=normalized))
        formatted = self._format_conversation_text(role, normalized)
        if self._display_entries:
            last_entry = self._display_entries[-1]
            if last_entry.kind == "conversation" and last_entry.text == formatted:
                return
        self._display_entries.append(DisplayEntry(kind="conversation", text=formatted))

    def _format_conversation_entry(self, entry: ConversationEntry) -> str:
        return self._format_conversation_text(entry.role, entry.text)

    def _format_conversation_text(self, role: str, text: str) -> str:
        formatted = self._shorten(" ".join(text.split()), 220)
        if role == "user":
            return f"> {formatted}"
        return formatted

    async def _request_user_choice(self, event: dict[str, object]) -> dict[str, Any]:
        if self._pending_choice is not None:
            raise ValueError("目前已有一個待選的使用者選項。")
        prompt = str(event.get("message") or "").strip()
        if not prompt:
            raise ValueError("選項請求缺少 message。")
        raw_options = event.get("options")
        if raw_options is None:
            raw_options = []
        if not isinstance(raw_options, list):
            raise ValueError("選項請求的 options 格式無效。")
        options: list[ChoiceOptionEntry] = []
        for index, item in enumerate(raw_options[:25], start=1):
            if not isinstance(item, dict):
                raise ValueError(f"選項請求的 options[{index}] 格式無效。")
            label = str(item.get("label") or "").strip()
            value = str(item.get("value") or "").strip()
            description = str(item.get("description") or "").strip()
            if not label or not value:
                raise ValueError(f"選項請求的 options[{index}] 缺少 label 或 value。")
            options.append(ChoiceOptionEntry(label=label, value=value, description=description))
        allow_freeform = self._coerce_bool(event.get("allow_freeform"), default=False)
        if not options and not allow_freeform:
            raise ValueError("選項請求至少需要 options 或 allow_freeform=true。")
        min_values = self._coerce_choice_count(event.get("min_values"), default=1)
        max_values = self._coerce_choice_count(event.get("max_values"), default=1)
        if options:
            max_values = min(max_values, len(options))
            min_values = min(min_values, max_values)
        else:
            min_values = 0
            max_values = 0

        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_choice = PendingChoiceState(
            prompt=prompt,
            options=options,
            placeholder=str(event.get("placeholder") or "").strip() or "請選擇一個選項",
            future=future,
            min_values=min_values,
            max_values=max_values,
            allow_freeform=allow_freeform,
            freeform_placeholder=str(event.get("freeform_placeholder") or "").strip(),
        )
        self._append_conversation("assistant", prompt)
        self._append_activity("等待使用者從選項中做決定。", key="choice")
        await self.request_render(force=True)
        try:
            return await future
        finally:
            self._pending_choice = None
            self._remove_activity("choice")
            await self.request_render(force=True)

    async def submit_choice(self, values: list[str] | str) -> None:
        if self._pending_choice is None:
            raise ValueError("目前沒有待選項目。")
        raw_values = [values] if isinstance(values, str) else list(values)
        normalized_values = [str(value).strip() for value in raw_values if str(value).strip()]
        value_set = set(normalized_values)
        selected = [option for option in self._pending_choice.options if option.value in value_set]
        if not selected:
            raise ValueError("找不到對應的選項。")
        if len(selected) < self._pending_choice.min_values or len(selected) > self._pending_choice.max_values:
            raise ValueError("選取項目數量不符合限制。")
        selected_payload = [{"label": option.label, "value": option.value} for option in selected]
        self._append_conversation("user", self._format_selected_choice_text(selected))
        if not self._pending_choice.future.done():
            self._pending_choice.future.set_result(
                {
                    "mode": "selection",
                    "label": selected[0].label,
                    "value": selected[0].value,
                    "selections": selected_payload,
                    "input": "",
                }
            )
        await self.bot.log_event(
            "Agent 選項回覆",
            f"使用者回覆 agent 選項。\nTask ID: {self.task_record.id}\n選項: {self._shorten(self._format_selected_choice_text(selected), 150)}",
            user=self.user,
            guild=self.guild,
        )

    async def submit_freeform(self, text: str) -> None:
        if self._pending_choice is None:
            raise ValueError("目前沒有待選項目。")
        if not self._pending_choice.allow_freeform:
            raise ValueError("目前不接受自由輸入。")
        normalized = str(text).strip()
        if not normalized:
            raise ValueError("自由輸入內容不可為空白。")
        self._append_conversation("user", normalized)
        if not self._pending_choice.future.done():
            self._pending_choice.future.set_result(
                {
                    "mode": "freeform",
                    "label": normalized,
                    "value": normalized,
                    "selections": [],
                    "input": normalized,
                }
            )
        await self.bot.log_event(
            "Agent 自由輸入回覆",
            f"使用者以自由輸入回覆 agent。\nTask ID: {self.task_record.id}\n內容: {self._shorten(normalized, 150)}",
            user=self.user,
            guild=self.guild,
        )

    def _cancel_pending_choice(self, reason: str) -> None:
        if self._pending_choice is None:
            return
        if not self._pending_choice.future.done():
            self._pending_choice.future.set_exception(ValueError(reason))

    def _append_activity(self, text: str, *, key: str | None = None, transient: bool = False) -> None:
        normalized = " ".join(str(text).split())
        if not normalized:
            return
        normalized = self._shorten(normalized, 180)
        display_text = f"-# {normalized}"
        if key is not None:
            for entry in self._activity_lines:
                if entry.key != key:
                    continue
                if entry.text == normalized and entry.transient == transient:
                    return
                entry.text = normalized
                entry.transient = transient
                self._upsert_display_activity(display_text, key=key, transient=transient)
                return
        elif transient is False:
            self._clear_transient_activities()
            if self._activity_lines:
                last_entry = self._activity_lines[-1]
                if last_entry.key is None and last_entry.text == normalized:
                    return
        self._activity_lines.append(ActivityEntry(text=normalized, key=key, transient=transient))
        self._upsert_display_activity(display_text, key=key, transient=transient)

    def _clear_transient_activities(self) -> None:
        if self._activity_lines:
            retained_entries = [entry for entry in self._activity_lines if not entry.transient]
            self._activity_lines = deque(retained_entries)
        self._display_entries = deque(entry for entry in self._display_entries if not entry.transient)

    def _remove_activity(self, key: str | None) -> None:
        if key is None or not self._activity_lines:
            return
        self._activity_lines = deque(entry for entry in self._activity_lines if entry.key != key)
        self._display_entries = deque(entry for entry in self._display_entries if entry.key != key)

    def _upsert_display_activity(self, text: str, *, key: str | None, transient: bool) -> None:
        if key is not None:
            for entry in self._display_entries:
                if entry.kind != "activity" or entry.key != key:
                    continue
                entry.text = text
                entry.transient = transient
                return
        elif transient is False and self._display_entries:
            last_entry = self._display_entries[-1]
            if last_entry.kind == "activity" and last_entry.key is None and last_entry.text == text:
                return
        self._display_entries.append(DisplayEntry(kind="activity", text=text, key=key, transient=transient))

    def _scope_activity_key(self, key: str | None) -> str | None:
        if key is None or self._active_run_scope is None:
            return key
        return f"{self._active_run_scope}:{key}"

    def _shorten(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _format_selected_choice_text(self, selected: list[ChoiceOptionEntry]) -> str:
        return "、".join(option.label for option in selected)

    @staticmethod
    def _coerce_bool(value: object, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "y", "on"}:
                return True
            if lowered in {"0", "false", "no", "n", "off"}:
                return False
        return default

    @staticmethod
    def _coerce_choice_count(value: object, *, default: int) -> int:
        if isinstance(value, bool) or value is None:
            return default
        if isinstance(value, int):
            return max(1, min(value, 25))
        if isinstance(value, float) and value.is_integer():
            return max(1, min(int(value), 25))
        return default


class AgentConversationView(discord.ui.LayoutView):
    def __init__(self, session: AgentConversationSession) -> None:
        super().__init__(timeout=None)
        self.session = session
        self._main_display = discord.ui.TextDisplay("-# 等待新的 agent 訊息。")
        self._choice_display = discord.ui.TextDisplay("## 請選擇\n(目前沒有待選項目)")
        self._tasks_display = discord.ui.TextDisplay("## 待辦事項\n(無)")
        self._context_display = discord.ui.TextDisplay("-# (未設定)")
        self._operations_display = discord.ui.TextDisplay("-# #0")
        self._after_main_separator = discord.ui.Separator(spacing=discord.SeparatorSpacing.large)
        self._after_choice_separator = discord.ui.Separator(spacing=discord.SeparatorSpacing.large)
        self._after_tasks_separator = discord.ui.Separator(spacing=discord.SeparatorSpacing.large)
        self._after_context_separator = discord.ui.Separator(spacing=discord.SeparatorSpacing.large)
        self._choice_select = AgentChoiceSelect(self)
        self._choice_select_row = discord.ui.ActionRow(self._choice_select)
        self._choice_freeform_button = discord.ui.Button(label="自由輸入", style=discord.ButtonStyle.primary)
        self._choice_freeform_button.callback = self._on_freeform_choice
        self._choice_cancel_button = discord.ui.Button(label="取消選擇", style=discord.ButtonStyle.secondary)
        self._choice_cancel_button.callback = self._on_cancel_choice
        self._choice_actions_row = discord.ui.ActionRow(self._choice_freeform_button, self._choice_cancel_button)
        self._interrupt_button = discord.ui.Button(label="中斷", style=discord.ButtonStyle.danger)
        self._interrupt_button.callback = self._on_interrupt
        self._send_message_button = discord.ui.Button(label="傳送訊息", style=discord.ButtonStyle.primary)
        self._send_message_button.callback = self._on_send_message
        self._refresh_button = discord.ui.Button(label="重新整理", style=discord.ButtonStyle.secondary)
        self._refresh_button.callback = self._on_refresh
        self._actions_row = discord.ui.ActionRow(
            self._interrupt_button,
            self._send_message_button,
            self._refresh_button,
        )
        self._container = discord.ui.Container(accent_colour=discord.Colour.blurple())
        self._rebuild_container()
        self.add_item(self._container)

    async def _on_interrupt(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        await interaction.response.defer()
        await self.session.interrupt()

    async def _on_send_message(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        await interaction.response.send_modal(AgentMessageModal(self.session))

    async def _on_refresh(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        await interaction.response.defer()
        await self.session.request_render(force=True)

    async def _on_cancel_choice(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        await interaction.response.defer()
        self.session._cancel_pending_choice("使用者取消了選項選擇。")
        await self.session.request_render(force=True)

    async def _on_freeform_choice(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        if self.session._pending_choice is None or not self.session._pending_choice.allow_freeform:
            await interaction.response.send_message("目前不接受自由輸入。", ephemeral=True)
            return
        await interaction.response.send_modal(AgentChoiceFreeformModal(self.session))

    def disable_all_items(self) -> None:
        self._choice_select.disabled = True
        self._choice_freeform_button.disabled = True
        self._choice_cancel_button.disabled = True
        self._interrupt_button.disabled = True
        self._send_message_button.disabled = True
        self._refresh_button.disabled = True

    def sync_buttons(self) -> None:
        if self.session._closed:
            self.disable_all_items()
            return
        self._choice_select.sync_from_session()
        self._choice_freeform_button.disabled = (
            self.session._pending_choice is None or not self.session._pending_choice.allow_freeform
        )
        self._choice_cancel_button.disabled = self.session._pending_choice is None
        self._interrupt_button.disabled = not self.session.is_busy()

    def _rebuild_container(self) -> None:
        self._container.clear_items()
        self._container.add_item(self._main_display)
        self._container.add_item(self._after_main_separator)
        if self.session._pending_choice is not None:
            self._container.add_item(self._choice_display)
            if self.session._pending_choice.options:
                self._container.add_item(self._choice_select_row)
            self._container.add_item(self._choice_actions_row)
            self._container.add_item(self._after_choice_separator)
        if self.session.task_items:
            self._container.add_item(self._tasks_display)
            self._container.add_item(self._after_tasks_separator)
        self._container.add_item(self._context_display)
        self._container.add_item(self._after_context_separator)
        self._container.add_item(self._operations_display)
        self._container.add_item(self._actions_row)

    def sync_layout(self) -> None:
        self._main_display.content = self.session._render_main_display()
        self._choice_display.content = self.session._render_choice_block()
        self._tasks_display.content = self.session._render_tasks_block()
        self._context_display.content = self.session._render_context_block()
        self._operations_display.content = self.session._render_operations_block()
        self.sync_buttons()
        self._rebuild_container()


class AgentChoiceSelect(discord.ui.Select):
    def __init__(self, view: AgentConversationView) -> None:
        super().__init__(
            placeholder="目前沒有待選項目",
            min_values=1,
            max_values=1,
            options=[discord.SelectOption(label="目前沒有待選項目", value="__none__")],
            disabled=True,
        )
        self.parent_view = view

    async def callback(self, interaction: discord.Interaction) -> None:
        if interaction.user.id != self.parent_view.session.user.id:
            await interaction.response.send_message("只有原本的使用者可以操作這個對話。", ephemeral=True)
            return
        if self.parent_view.session._pending_choice is None:
            await interaction.response.send_message("目前沒有待選項目。", ephemeral=True)
            return
        await interaction.response.defer()
        await self.parent_view.session.submit_choice(list(self.values))

    def sync_from_session(self) -> None:
        pending_choice = self.parent_view.session._pending_choice
        if pending_choice is None:
            self.disabled = True
            self.placeholder = "目前沒有待選項目"
            self.min_values = 1
            self.max_values = 1
            self.options = [discord.SelectOption(label="目前沒有待選項目", value="__none__")]
            return
        if not pending_choice.options:
            self.disabled = True
            self.placeholder = "請改用自由輸入"
            self.min_values = 1
            self.max_values = 1
            self.options = [discord.SelectOption(label="請改用自由輸入", value="__freeform__")]
            return
        self.disabled = False
        self.placeholder = pending_choice.placeholder
        self.min_values = max(1, pending_choice.min_values)
        self.max_values = max(self.min_values, pending_choice.max_values)
        self.options = [
            discord.SelectOption(
                label=self.parent_view.session._shorten(option.label, 100),
                value=option.value,
                description=self.parent_view.session._shorten(option.description, 100) if option.description else None,
            )
            for option in pending_choice.options[:25]
        ]


class AgentChoiceFreeformModal(discord.ui.Modal):
    def __init__(self, session: AgentConversationSession) -> None:
        super().__init__(title="自由輸入回覆")
        self.session = session
        pending_choice = session._pending_choice
        placeholder = "輸入選項外的補充、答案或偏好..."
        if pending_choice is not None and pending_choice.freeform_placeholder:
            placeholder = pending_choice.freeform_placeholder
        self.response_input = discord.ui.TextInput(
            label="回覆內容",
            placeholder=placeholder,
            style=discord.TextStyle.paragraph,
            max_length=1500,
        )
        self.add_item(self.response_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer()
        await self.session.submit_freeform(str(self.response_input))


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