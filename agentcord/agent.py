from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp

from agentcord.ai import PollinationsProvider, create_provider, parse_json_object, resolve_pollinations_model
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import AIUsage, AgentTaskItem, ConversationMessage, Provider, TaskRecord, TaskStatus, UserModelConfig, estimate_tokens
from agentcord.workspace import WorkspaceError, WorkspaceManager

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass(slots=True)
class AgentRunResult:
    summary: str
    plan: list[str]
    related_files: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
    messages: list[ConversationMessage] = field(default_factory=list)
    task_items: list[AgentTaskItem] = field(default_factory=list)
    model: str = ""
    context_length: int | None = None
    estimated_tokens: int = 0
    compression_count: int = 0
    task_id: int | None = None


@dataclass(slots=True)
class AgentPlanResult:
    plan: list[str]
    model: str = ""
    context_length: int | None = None
    usage: AIUsage | None = None


@dataclass(frozen=True, slots=True)
class AgentToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler_name: str

    def as_function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class CreditManager:
    def __init__(self, db: Database, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def ensure_affordable(self, user_id: int, config: UserModelConfig, input_text: str) -> None:
        estimated_input_tokens = max(1, len(input_text) // 4)
        reserve = self.settings.credit_reserve_output_tokens
        rate = self.settings.get_model_rate(config.provider, config.model)
        estimated_cost = (estimated_input_tokens + reserve) * rate
        if self.db.get_credits(user_id) < estimated_cost:
            raise ValueError(
                f"額度不足。預估至少需要 {estimated_cost:.2f}，"
                f"目前可用 {self.db.get_credits(user_id):.2f}。"
            )

    def charge(self, user_id: int, amount: float) -> float:
        return self.db.consume_credits(user_id, amount)


class CodingAgent:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        workspace: WorkspaceManager,
        session: aiohttp.ClientSession,
    ) -> None:
        self.settings = settings
        self.db = db
        self.workspace = workspace
        self.session = session
        self.credits = CreditManager(db, settings)
        self._tool_aliases = {"rmdir": "remove_folder"}
        self._tool_specs = {spec.name: spec for spec in self._build_tool_specs()}

    async def run(
        self,
        user_id: int,
        prompt: str,
        *,
        task: TaskRecord | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentRunResult:
        config = self.db.get_model_config(user_id, self.settings.default_pollinations_model)
        provider = create_provider(self.session, self.settings, config)
        model_info = None
        if config.provider is Provider.POLLINATIONS:
            model_info = await resolve_pollinations_model(self.session, self.settings, config.model)
        context_length = model_info.context_length if model_info is not None else None
        actual_model = config.model

        history_messages = list(task.messages) if task is not None else []
        history_messages.append(ConversationMessage(role="user", content=prompt))
        compression_count = task.compression_count if task is not None else 0
        history_messages, compression_count = await self._compress_context_if_needed(
            user_id,
            provider,
            config,
            history_messages,
            context_length,
            compression_count,
            progress_callback,
        )

        if task is None:
            task = self.db.create_task(
                user_id,
                title=prompt[:120],
                status=TaskStatus.RUNNING,
                messages=history_messages,
                model=actual_model,
                context_length=context_length,
                compression_count=compression_count,
            )
        else:
            task = self.db.update_task(
                task.id,
                TaskStatus.RUNNING,
                task.related_files,
                messages=history_messages,
                model=actual_model,
                context_length=context_length,
                compression_count=compression_count,
            )

        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "activity_key": "task",
                "message": f"任務 #{task.id} 執行中。",
            },
        )

        plan = list(task.plan) if task.plan else []
        transcript: list[dict[str, str]] = []
        changed_files: set[str] = set(task.related_files)
        validations: list[str] = []
        final_summary = task.summary or "未進行任何變更。"
        current_task_items = list(task.task_items)
        estimated_tokens = 0

        for iteration in range(1, self.settings.agent_max_iterations + 1):
            context = self._build_iteration_context(user_id, prompt, plan, transcript, history_messages, current_task_items)
            estimated_tokens = estimate_tokens(context)
            await self._emit_progress(
                progress_callback,
                {
                    "type": "context",
                    "model": actual_model,
                    "context_length": context_length,
                    "estimated_tokens": estimated_tokens,
                    "compression_count": compression_count,
                    "history_messages": len(history_messages),
                    "phase": f"iteration-{iteration}",
                },
            )
            self.credits.ensure_affordable(user_id, config, context)
            await self._emit_activity(
                progress_callback,
                f"第 {iteration} 輪決策生成中。",
                activity_key=f"decision:{iteration}",
            )
            step_response = await provider.stream_generate(
                [
                    {"role": "system", "content": self._build_agent_system_prompt()},
                    {"role": "user", "content": context},
                ],
                on_delta=self._build_stream_progress_callback(
                    f"第 {iteration} 輪決策生成中",
                    progress_callback,
                    activity_key=f"decision:{iteration}",
                ),
            )
            self.credits.charge(user_id, step_response.usage.cost)
            actual_model = step_response.model or actual_model
            await self._emit_progress(
                progress_callback,
                {
                    "type": "context",
                    "model": actual_model,
                    "context_length": context_length,
                    "estimated_tokens": estimated_tokens,
                    "compression_count": compression_count,
                    "history_messages": len(history_messages),
                    "phase": f"iteration-{iteration}",
                },
            )
            decision = parse_json_object(step_response.content)
            await self._remove_activity(progress_callback, activity_key=f"decision:{iteration}")
            tool_results, touched_files, current_task_items = await self._execute_actions(
                user_id,
                self._extract_decision_actions(decision),
                current_task_items,
                progress_callback,
                iteration,
            )
            changed_files.update(touched_files)

            validations.extend(self._validate_changed_python_files(user_id, touched_files))
            if validations:
                await self._emit_activity(
                    progress_callback,
                    f"目前累積 {len(validations)} 筆驗證結果。",
                    activity_key="validation",
                )
            transcript.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "decision": decision,
                            "tool_results": tool_results,
                            "validations": validations[-len(touched_files) :] if touched_files else [],
                        },
                        ensure_ascii=False,
                    ),
                }
            )
            final_summary = str(decision.get("summary", final_summary))
            await self._emit_activity(
                progress_callback,
                f"目前摘要：{final_summary}",
                activity_key="summary",
            )
            if decision.get("done"):
                break

        related_files = sorted(changed_files)
        history_messages.append(ConversationMessage(role="assistant", content=final_summary))
        task = self.db.update_task(
            task.id,
            TaskStatus.DONE,
            related_files,
            summary=final_summary,
            plan=plan,
            validations=validations,
            messages=history_messages,
            task_items=current_task_items,
            model=actual_model,
            context_length=context_length,
            compression_count=compression_count,
        )
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "activity_key": "task",
                "message": f"任務 #{task.id} 已完成。",
            },
        )
        return AgentRunResult(
            summary=final_summary,
            plan=plan,
            related_files=related_files,
            validations=validations,
            messages=history_messages,
            task_items=current_task_items,
            model=actual_model,
            context_length=context_length,
            estimated_tokens=estimated_tokens,
            compression_count=compression_count,
            task_id=task.id,
        )

    async def plan(
        self,
        user_id: int,
        prompt: str,
        *,
        task: TaskRecord | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> AgentPlanResult:
        config = self.db.get_model_config(user_id, self.settings.default_pollinations_model)
        provider = create_provider(self.session, self.settings, config)
        model_info = None
        if config.provider is Provider.POLLINATIONS:
            model_info = await resolve_pollinations_model(self.session, self.settings, config.model)
        context_length = model_info.context_length if model_info is not None else None
        current_model = config.model
        history_messages = list(task.messages) if task is not None else []
        if prompt.strip():
            history_messages.append(ConversationMessage(role="user", content=prompt))
        plan, resolved_model, usage = await self._create_plan(
            user_id,
            prompt,
            provider,
            config,
            history_messages,
            progress_callback,
            context_length,
            current_model,
        )
        return AgentPlanResult(
            plan=plan,
            model=resolved_model,
            context_length=context_length,
            usage=usage,
        )

    async def _create_plan(
        self,
        user_id: int,
        prompt: str,
        provider: Any,
        config: UserModelConfig,
        history_messages: list[ConversationMessage],
        progress_callback: ProgressCallback | None,
        context_length: int | None,
        current_model: str,
    ) -> tuple[list[str], str, AIUsage]:
        planning_prompt = (
            "請為下列程式任務建立精簡的執行計畫。"
            "請只回傳 JSON，最上層需包含名為 plan 的鍵，內容是繁體中文短句列表。\n\n"
            f"對話歷史：\n{self._render_conversation_history(history_messages)}\n\n"
            f"工作區樹狀內容：\n{self.workspace.dump_tree(user_id)}\n\n"
            f"任務：\n{prompt}"
        )
        await self._emit_progress(
            progress_callback,
            {
                "type": "context",
                "model": current_model,
                "context_length": context_length,
                "estimated_tokens": estimate_tokens(planning_prompt),
                "compression_count": 0,
                "history_messages": len(history_messages),
                "phase": "planning",
            },
        )
        self.credits.ensure_affordable(user_id, config, planning_prompt)
        await self._emit_activity(
            progress_callback,
            "計畫生成中。",
            activity_key="plan",
        )
        response = await provider.stream_generate(
            [
                {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ],
            on_delta=self._build_stream_progress_callback(
                "計畫生成中",
                progress_callback,
                activity_key="plan",
            ),
        )
        self.credits.charge(user_id, response.usage.cost)
        resolved_model = response.model or current_model
        await self._emit_progress(
            progress_callback,
            {
                "type": "context",
                "model": resolved_model,
                "context_length": context_length,
                "estimated_tokens": estimate_tokens(planning_prompt),
                "compression_count": 0,
                "history_messages": len(history_messages),
                "phase": "planning",
            },
        )
        data = parse_json_object(response.content)
        plan = [str(item) for item in data.get("plan", []) if str(item).strip()]
        await self._emit_activity(
            progress_callback,
            f"計畫生成已完成，共 {len(plan or [])} 個步驟。",
            activity_key="plan",
        )
        return plan or ["檢查需求", "更新檔案", "驗證語法"], resolved_model, response.usage

    def _build_iteration_context(
        self,
        user_id: int,
        prompt: str,
        plan: list[str],
        transcript: list[dict[str, str]],
        history_messages: list[ConversationMessage],
        current_task_items: list[AgentTaskItem],
    ) -> str:
        return (
            f"使用者需求：\n{prompt}\n\n"
            f"對話歷史：\n{self._render_conversation_history(history_messages)}\n\n"
            f"目前計畫：\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"目前 tasks：\n{json.dumps([{"title": item.title, "status": item.status} for item in current_task_items], ensure_ascii=False, indent=2)}\n\n"
            f"工作區樹狀內容：\n{self.workspace.dump_tree(user_id)}\n\n"
            f"先前工具紀錄：\n{json.dumps(transcript[-6:], ensure_ascii=False, indent=2)}\n\n"
            "請回傳 JSON，必須包含 summary、done、related_files、actions 這些鍵。"
            f"actions 最多只能有 {self.settings.agent_max_actions_per_iteration} 個。"
        )

    def _build_tool_specs(self) -> list[AgentToolSpec]:
        return [
            AgentToolSpec(
                name="list_files",
                description="列出工作區路徑下的檔案或資料夾。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "工作區相對路徑，預設為 ."},
                    },
                },
                handler_name="_tool_list_files",
            ),
            AgentToolSpec(
                name="read_file",
                description="讀取 UTF-8 文字檔內容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要讀取的工作區檔案路徑。"},
                    },
                    "required": ["path"],
                },
                handler_name="_tool_read_file",
            ),
            AgentToolSpec(
                name="write_file",
                description="直接寫入完整檔案內容；若是修改既有檔案，通常應優先使用 apply_patch。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要寫入的工作區檔案路徑。"},
                        "content": {"type": "string", "description": "完整 UTF-8 文字內容。"},
                    },
                    "required": ["path", "content"],
                },
                handler_name="_tool_write_file",
            ),
            AgentToolSpec(
                name="apply_patch",
                description="以 unified diff / patch 形式修改既有檔案。",
                parameters={
                    "type": "object",
                    "properties": {
                        "diff": {"type": "string", "description": "要套用的 unified diff。"},
                    },
                    "required": ["diff"],
                },
                handler_name="_tool_apply_patch",
            ),
            AgentToolSpec(
                name="delete_file",
                description="刪除單一檔案。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要刪除的工作區檔案路徑。"},
                    },
                    "required": ["path"],
                },
                handler_name="_tool_delete_file",
            ),
            AgentToolSpec(
                name="create_folder",
                description="建立資料夾。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要建立的工作區資料夾路徑。"},
                    },
                    "required": ["path"],
                },
                handler_name="_tool_create_folder",
            ),
            AgentToolSpec(
                name="remove_folder",
                description="刪除資料夾；force=true 時可遞迴刪除非空資料夾。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要刪除的工作區資料夾路徑。"},
                        "force": {"type": "boolean", "description": "是否遞迴刪除非空資料夾。"},
                    },
                    "required": ["path"],
                },
                handler_name="_tool_remove_folder",
            ),
            AgentToolSpec(
                name="py_compile_check",
                description="對 Python 檔案做語法檢查。",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "要檢查的 Python 檔案路徑。"},
                    },
                    "required": ["path"],
                },
                handler_name="_tool_py_compile_check",
            ),
            AgentToolSpec(
                name="search_web",
                description="搜尋網路並回傳結構化結果。",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜尋關鍵字。"},
                    },
                    "required": ["query"],
                },
                handler_name="_tool_search_web",
            ),
            AgentToolSpec(
                name="fetch_url",
                description="抓取公開網址的內容。",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "要抓取的完整 URL。"},
                    },
                    "required": ["url"],
                },
                handler_name="_tool_fetch_url",
            ),
            AgentToolSpec(
                name="tasks",
                description="更新目前工作清單，讓使用者看到進度。",
                parameters={
                    "type": "object",
                    "properties": {
                        "items": {
                            "type": "array",
                            "description": "task 清單。",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "text": {"type": "string"},
                                    "status": {"type": "string"},
                                    "done": {"type": "boolean"},
                                },
                            },
                        },
                    },
                    "required": ["items"],
                },
                handler_name="_tool_tasks",
            ),
        ]

    def _build_ai_tools(self) -> list[dict[str, Any]]:
        return [spec.as_function_schema() for spec in self._tool_specs.values()]

    def _build_agent_system_prompt(self) -> str:
        tool_schema = json.dumps(self._build_ai_tools(), ensure_ascii=False, indent=2)
        return (
            _AGENT_SYSTEM_PROMPT_PREFIX
            + "\n可用工具（function schema）：\n"
            + tool_schema
            + "\n\n"
            + "回傳 JSON schema：\n"
            + "{\n"
            + '  "summary": "簡短摘要",\n'
            + '  "done": true|false,\n'
            + '  "related_files": ["path"],\n'
            + '  "actions": [\n'
            + '    {"tool": "list_files", "path": "."},\n'
            + '    {"tool": "read_file", "path": "src/app.py"}\n'
            + "  ]\n"
            + "}\n"
            + "規則：\n"
            + f"- actions 最多只能有 {self.settings.agent_max_actions_per_iteration} 個。\n"
            + "- `tool` 必須對應到上面的 function.name。\n"
            + "- 其他欄位必須符合對應 function.parameters。\n"
            + "- 若不需要任何工具，actions 請回傳空陣列。\n"
            + "- summary 與 related_files 請反映這一輪實際結果。"
        )

    def _extract_decision_actions(self, decision: dict[str, Any]) -> list[dict[str, Any]]:
        raw_actions: Any = decision.get("actions")
        if raw_actions is None:
            raw_actions = decision.get("tool_calls")
        if raw_actions is None and any(key in decision for key in ("tool", "name", "function")):
            raw_actions = [decision]
        return self._normalize_actions(raw_actions)

    def _normalize_actions(self, raw_actions: Any) -> list[dict[str, Any]]:
        if raw_actions is None:
            return []
        if isinstance(raw_actions, str):
            try:
                parsed = json.loads(raw_actions)
            except json.JSONDecodeError:
                try:
                    parsed = parse_json_object(raw_actions)
                except ValueError:
                    return []
            return self._normalize_actions(parsed)

        if isinstance(raw_actions, dict):
            if isinstance(raw_actions.get("tool_calls"), list):
                candidates = raw_actions["tool_calls"]
            elif isinstance(raw_actions.get("actions"), list):
                candidates = raw_actions["actions"]
            else:
                candidates = [raw_actions]
        elif isinstance(raw_actions, list):
            candidates = raw_actions
        else:
            return []

        normalized_actions: list[dict[str, Any]] = []
        for raw_action in candidates:
            normalized_action = self._normalize_action_call(raw_action)
            if normalized_action is not None:
                normalized_actions.append(normalized_action)
        return normalized_actions

    def _normalize_action_call(self, raw_action: Any) -> dict[str, Any] | None:
        if not isinstance(raw_action, dict):
            return None

        function = raw_action.get("function")
        if isinstance(function, dict):
            name = self._normalize_tool_name(function.get("name"))
            if not name:
                return None
            return {"tool": name, **self._safe_parse_tool_arguments(function.get("arguments"))}

        name = self._normalize_tool_name(raw_action.get("tool") or raw_action.get("name"))
        if not name:
            return None

        normalized = {key: value for key, value in raw_action.items() if key not in {"id", "function", "name"}}
        normalized["tool"] = name
        if "arguments" in normalized:
            arguments = self._safe_parse_tool_arguments(normalized.pop("arguments"))
            normalized.update(arguments)
        return normalized

    def _safe_parse_tool_arguments(self, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if arguments is None:
            return {}
        raw = str(arguments).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = parse_json_object(raw)
            except ValueError:
                return {}
        return parsed if isinstance(parsed, dict) else {}

    def _normalize_tool_name(self, tool_name: Any) -> str:
        normalized = str(tool_name or "").strip().lower()
        if not normalized:
            return ""
        return self._tool_aliases.get(normalized, normalized)

    async def _execute_actions(
        self,
        user_id: int,
        actions: list[dict[str, Any]],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
        iteration: int,
    ) -> tuple[list[dict[str, Any]], list[str], list[AgentTaskItem]]:
        results: list[dict[str, Any]] = []
        touched_files: list[str] = []
        normalized_actions = self._normalize_actions(actions)
        for action_index, action in enumerate(normalized_actions[: self.settings.agent_max_actions_per_iteration], start=1):
            tool_name = self._normalize_tool_name(action.get("tool"))
            action["tool"] = tool_name
            activity_key = f"tool:{iteration}:{action_index}"
            await self._emit_activity(
                progress_callback,
                self._format_tool_start_message(tool_name, action),
                activity_key=activity_key,
            )
            spec = self._tool_specs.get(tool_name)
            if spec is None:
                results.append({"tool": tool_name or "(unknown)", "error": "不支援的工具。"})
                await self._emit_activity(
                    progress_callback,
                    f"工具 {tool_name or '(unknown)'} 失敗：不支援的工具。",
                    activity_key=activity_key,
                )
                continue
            try:
                result_payload, tool_touched_files, current_task_items = await self._execute_tool(
                    spec,
                    user_id,
                    action,
                    current_task_items,
                    progress_callback,
                )
                touched_files.extend(tool_touched_files)
                results.append({"tool": tool_name, **result_payload})
            except (WorkspaceError, KeyError, ValueError, aiohttp.ClientError) as exc:
                results.append({"tool": tool_name, "error": str(exc)})
                await self._emit_activity(
                    progress_callback,
                    f"{self._format_tool_label(tool_name)}失敗：{exc}",
                    activity_key=activity_key,
                )
                continue
            await self._emit_activity(
                progress_callback,
                self._format_tool_finish_message(tool_name, action),
                activity_key=activity_key,
            )
        return results, touched_files, current_task_items

    async def _execute_tool(
        self,
        spec: AgentToolSpec,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        handler = getattr(self, spec.handler_name)
        return await handler(user_id, action, current_task_items, progress_callback)

    async def _tool_list_files(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = str(action.get("path") or ".")
        entries = self.workspace.list_files(user_id, path)
        return (
            {
                "path": path,
                "result": [
                    {"path": entry.path, "kind": entry.kind, "size": entry.size}
                    for entry in entries
                ],
            },
            [],
            current_task_items,
        )

    async def _tool_read_file(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        return ({"path": path, "result": self.workspace.read_file(user_id, path)}, [], current_task_items)

    async def _tool_write_file(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        content = self._require_string_argument(action, "content", allow_empty=True)
        self.workspace.write_file(user_id, path, content)
        return ({"path": path, "result": "ok"}, [path], current_task_items)

    async def _tool_apply_patch(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        diff = self._require_string_argument(action, "diff", allow_empty=True)
        changed = self.workspace.apply_patch(user_id, diff)
        return ({"result": changed}, changed, current_task_items)

    async def _tool_delete_file(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        self.workspace.delete_file(user_id, path)
        return ({"path": path, "result": "ok"}, [path], current_task_items)

    async def _tool_create_folder(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        created_path = self.workspace.create_folder(user_id, path)
        return ({"path": created_path, "result": "ok"}, [], current_task_items)

    async def _tool_remove_folder(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        force = self._coerce_bool(action.get("force"), default=False)
        removed_path = self.workspace.remove_folder(user_id, path, force=force)
        return ({"path": removed_path, "force": force, "result": "ok"}, [removed_path], current_task_items)

    async def _tool_py_compile_check(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        path = self._require_string_argument(action, "path")
        return ({"path": path, "result": self.workspace.py_compile_check(user_id, path)}, [], current_task_items)

    async def _tool_search_web(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        query = self._require_string_argument(action, "query")
        outcome = await self._search_web(user_id, query)
        return ({"query": query, "result": outcome}, [], current_task_items)

    async def _tool_fetch_url(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del progress_callback
        url = self._require_string_argument(action, "url")
        outcome = await self._fetch_url(user_id, url)
        return ({"url": url, "result": outcome}, [], current_task_items)

    async def _tool_tasks(
        self,
        user_id: int,
        action: dict[str, Any],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[dict[str, Any], list[str], list[AgentTaskItem]]:
        del user_id
        current_task_items = self._coerce_task_items(action.get("items", []))
        await self._emit_progress(
            progress_callback,
            {
                "type": "tasks",
                "items": [
                    {"title": item.title, "status": item.status}
                    for item in current_task_items
                ],
            },
        )
        return (
            {
                "result": [
                    {"title": item.title, "status": item.status}
                    for item in current_task_items
                ],
            },
            [],
            current_task_items,
        )

    def _require_string_argument(self, action: dict[str, Any], key: str, *, allow_empty: bool = False) -> str:
        value = action.get(key)
        if not isinstance(value, str):
            raise ValueError(f"工具 {action.get('tool')} 缺少字串參數：{key}。")
        if not allow_empty and not value.strip():
            raise ValueError(f"工具 {action.get('tool')} 缺少字串參數：{key}。")
        return value if allow_empty else value.strip()

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
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

    def _validate_changed_python_files(self, user_id: int, touched_files: list[str]) -> list[str]:
        validations: list[str] = []
        for path in sorted({item for item in touched_files if item.endswith(".py")}):
            try:
                validations.append(self.workspace.py_compile_check(user_id, path))
            except Exception as exc:  # noqa: BLE001
                validations.append(f"{path} 的語法錯誤：{exc}")
        return validations

    async def _compress_context_if_needed(
        self,
        user_id: int,
        provider: Any,
        config: UserModelConfig,
        history_messages: list[ConversationMessage],
        context_length: int | None,
        compression_count: int,
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[ConversationMessage], int]:
        if context_length is None or len(history_messages) <= 6:
            return history_messages, compression_count

        token_budget = max(2048, int(context_length * 0.6) - self.settings.credit_reserve_output_tokens)
        while estimate_tokens(self._render_conversation_history(history_messages)) > token_budget and len(history_messages) > 6:
            older_messages = history_messages[:-6]
            recent_messages = history_messages[-6:]
            summary_prompt = (
                "請將以下較早的對話內容壓縮成一段繁體中文摘要，保留需求、限制、重要決策與未完成事項。"
                "不要輸出 JSON。\n\n"
                f"對話內容：\n{self._render_conversation_history(older_messages)}"
            )
            self.credits.ensure_affordable(user_id, config, summary_prompt)
            await self._emit_activity(
                progress_callback,
                "上下文壓縮中。",
                activity_key="compression",
            )
            response = await provider.generate(
                [
                    {"role": "system", "content": _COMPRESSION_SYSTEM_PROMPT},
                    {"role": "user", "content": summary_prompt},
                ]
            )
            self.credits.charge(user_id, response.usage.cost)
            history_messages = [
                ConversationMessage(role="system", content=f"較早對話摘要：{response.content.strip()}"),
                *recent_messages,
            ]
            compression_count += 1
            await self._emit_activity(
                progress_callback,
                f"上下文壓縮已完成，目前已壓縮 {compression_count} 次。",
                activity_key="compression",
            )
        return history_messages, compression_count

    async def _emit_activity(
        self,
        progress_callback: ProgressCallback | None,
        message: str,
        *,
        activity_key: str | None = None,
        transient: bool = False,
    ) -> None:
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "message": message,
                "activity_key": activity_key,
                "transient": transient,
            },
        )

    async def _remove_activity(
        self,
        progress_callback: ProgressCallback | None,
        *,
        activity_key: str,
    ) -> None:
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity_remove",
                "activity_key": activity_key,
            },
        )

    async def _emit_progress(self, progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        maybe_result = progress_callback(event)
        if maybe_result is not None:
            await maybe_result

    def _build_stream_progress_callback(
        self,
        message_prefix: str,
        progress_callback: ProgressCallback | None,
        *,
        activity_key: str,
    ) -> Callable[[str], Awaitable[None] | None] | None:
        if progress_callback is None:
            return None

        received_chars = 0
        next_emit_threshold = 120

        async def on_delta(delta: str) -> None:
            nonlocal received_chars, next_emit_threshold
            received_chars += len(delta)
            if received_chars < next_emit_threshold:
                return
            next_emit_threshold += 240
            await self._emit_activity(
                progress_callback,
                f"{message_prefix}（已接收約 {received_chars} 字元）",
                activity_key=activity_key,
            )

        return on_delta

    def _render_conversation_history(self, messages: list[ConversationMessage]) -> str:
        if not messages:
            return "(無)"
        return "\n".join(
            f"[{message.role}] {message.content}"
            for message in messages[-12:]
        )

    def _coerce_task_items(self, raw_items: Any) -> list[AgentTaskItem]:
        items: list[AgentTaskItem] = []
        if not isinstance(raw_items, list):
            return items
        for raw_item in raw_items:
            if isinstance(raw_item, str) and raw_item.strip():
                items.append(AgentTaskItem(title=raw_item.strip(), status="pending"))
                continue
            if not isinstance(raw_item, dict):
                continue
            title = str(raw_item.get("title") or raw_item.get("text") or "").strip()
            if not title:
                continue
            if raw_item.get("done") is True:
                status = "done"
            else:
                status = str(raw_item.get("status") or "pending").strip() or "pending"
            items.append(AgentTaskItem(title=title, status=status))
        return items

    def _format_tool_label(self, tool_name: Any) -> str:
        labels = {
            "read_file": "讀取檔案",
            "write_file": "寫入檔案",
            "list_files": "列出路徑",
            "delete_file": "刪除檔案",
            "create_folder": "建立資料夾",
            "remove_folder": "刪除資料夾",
            "apply_patch": "套用 patch",
            "py_compile_check": "語法檢查",
            "search_web": "搜尋網路",
            "fetch_url": "抓取網址",
            "tasks": "更新 tasks 清單",
        }
        normalized = self._normalize_tool_name(tool_name)
        return labels.get(normalized, f"工具 {tool_name}")

    def _format_tool_start_message(self, tool_name: Any, action: dict[str, Any]) -> str:
        tool_name = self._normalize_tool_name(tool_name)
        if tool_name == "read_file":
            return f"讀取檔案中：{action.get('path', '')}"
        if tool_name == "write_file":
            return f"寫入檔案中：{action.get('path', '')}"
        if tool_name == "list_files":
            return f"列出路徑中：{action.get('path', '.') }"
        if tool_name == "delete_file":
            return f"刪除檔案中：{action.get('path', '')}"
        if tool_name == "create_folder":
            return f"建立資料夾中：{action.get('path', '')}"
        if tool_name == "remove_folder":
            return f"刪除資料夾中：{action.get('path', '')}"
        if tool_name == "apply_patch":
            return "套用 patch 中。"
        if tool_name == "py_compile_check":
            return f"語法檢查中：{action.get('path', '')}"
        if tool_name == "search_web":
            return f"搜尋網路中：{action.get('query', '')}"
        if tool_name == "fetch_url":
            return f"抓取網址中：{action.get('url', '')}"
        if tool_name == "tasks":
            return "更新 tasks 清單中。"
        return f"執行工具中：{tool_name}"

    def _format_tool_finish_message(self, tool_name: Any, action: dict[str, Any]) -> str:
        tool_name = self._normalize_tool_name(tool_name)
        if tool_name in {"read_file", "write_file", "delete_file", "create_folder", "remove_folder", "py_compile_check"}:
            return f"{self._format_tool_label(tool_name)}已完成：{action.get('path', '')}"
        if tool_name == "list_files":
            return f"列出路徑已完成：{action.get('path', '.') }"
        if tool_name == "search_web":
            return "搜尋網路已完成。"
        if tool_name == "fetch_url":
            return "抓取網址已完成。"
        if tool_name == "apply_patch":
            return "套用 patch 已完成。"
        if tool_name == "tasks":
            return "tasks 清單已更新。"
        return f"執行工具已完成：{tool_name}"

    async def _search_web(self, user_id: int, query: str) -> dict[str, Any]:
        config = UserModelConfig(
            provider=Provider.POLLINATIONS,
            model="gemini-search",
            api_key=self.settings.pollinations_api_key,
        )
        provider = PollinationsProvider(self.session, self.settings, config)
        response = await provider.generate(
            [
                {
                    "role": "system",
                    "content": (
                        "請搜尋網路並回傳 JSON，最上層需有 results 清單。"
                        "每個項目都必須包含 title、url、summary。"
                    ),
                },
                {"role": "user", "content": query},
            ]
        )
        data = parse_json_object(response.content)
        urls = [item["url"] for item in data.get("results", []) if isinstance(item, dict) and item.get("url")]
        self.db.remember_search_urls(user_id, urls)
        return data

    async def _fetch_url(self, user_id: int, url: str) -> str:
        request_kwargs = self._build_proxy_request_kwargs()
        async with self.session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=45),
            allow_redirects=True,
            **request_kwargs,
        ) as response:
            response.raise_for_status()
            return await response.text()

    def _build_proxy_request_kwargs(self) -> dict[str, Any]:
        if not self.settings.proxy_url:
            return {}
        request_kwargs: dict[str, Any] = {"proxy": self.settings.proxy_url}
        if self.settings.proxy_username:
            request_kwargs["proxy_auth"] = aiohttp.BasicAuth(
                self.settings.proxy_username,
                self.settings.proxy_password,
            )
        if self.settings.proxy_headers:
            request_kwargs["proxy_headers"] = self.settings.proxy_headers
        return request_kwargs


_PLANNING_SYSTEM_PROMPT = """
你是 Discord bot 工作區的 AI 程式規劃器。
只回傳合法 JSON。
plan 內容請使用繁體中文。
"""


_COMPRESSION_SYSTEM_PROMPT = """
你是對話上下文壓縮器。
請用繁體中文輸出精簡摘要，保留需求、限制、重要檔案與未完成事項。
不要輸出 JSON。
"""


_AGENT_SYSTEM_PROMPT_PREFIX = """
你是運行在受限文字檔工作區中的 AI 程式代理。
重要規則：
- 使用者不能執行程式碼。
- 只可使用下方列出的 tools。
- 編輯既有檔案時優先使用 apply_patch。
- 只可寫入 UTF-8 文字檔。
- 只回傳合法 JSON。
- summary 與 related_files 內容請使用繁體中文。
- fetch_url 可直接抓取公開網址內容；若設定了 PROXY_* 環境變數，會透過 proxy 抓取，不需要先經過 search_web。
- 如果目前工作有明確步驟，請使用 tasks 工具更新工作清單，好讓使用者看到目前進度。
"""
