from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import aiohttp

from agentcord.ai import PollinationsProvider, create_provider, parse_json_object, resolve_pollinations_model
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import AgentTaskItem, ConversationMessage, Provider, TaskRecord, TaskStatus, UserModelConfig, estimate_tokens
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
                model=config.model,
                context_length=context_length,
                compression_count=compression_count,
            )
        else:
            task = self.db.update_task(
                task.id,
                TaskStatus.RUNNING,
                task.related_files,
                messages=history_messages,
                model=config.model,
                context_length=context_length,
                compression_count=compression_count,
            )

        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "message": f"開始處理任務 #{task.id}。",
            },
        )

        plan = await self._create_plan(user_id, prompt, provider, config, history_messages, progress_callback, context_length)
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
                    "model": config.model,
                    "context_length": context_length,
                    "estimated_tokens": estimated_tokens,
                    "compression_count": compression_count,
                    "history_messages": len(history_messages),
                    "phase": f"iteration-{iteration}",
                },
            )
            self.credits.ensure_affordable(user_id, config, context)
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": f"第 {iteration} 輪決策生成中。",
                },
            )
            step_response = await provider.stream_generate(
                [
                    {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                on_delta=self._build_stream_progress_callback("決策生成", progress_callback),
            )
            self.credits.charge(user_id, step_response.usage.cost)
            decision = parse_json_object(step_response.content)
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": f"第 {iteration} 輪收到決策，準備執行工具。",
                },
            )
            tool_results, touched_files, current_task_items = await self._execute_actions(
                user_id,
                decision.get("actions", []),
                current_task_items,
                progress_callback,
            )
            changed_files.update(touched_files)

            validations.extend(self._validate_changed_python_files(user_id, touched_files))
            if validations:
                await self._emit_progress(
                    progress_callback,
                    {
                        "type": "activity",
                        "message": f"目前累積 {len(validations)} 筆驗證結果。",
                    },
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
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": f"第 {iteration} 輪摘要：{final_summary}",
                },
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
            model=config.model,
            context_length=context_length,
            compression_count=compression_count,
        )
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
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
            model=config.model,
            context_length=context_length,
            estimated_tokens=estimated_tokens,
            compression_count=compression_count,
            task_id=task.id,
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
    ) -> list[str]:
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
                "model": config.model,
                "context_length": context_length,
                "estimated_tokens": estimate_tokens(planning_prompt),
                "compression_count": 0,
                "history_messages": len(history_messages),
                "phase": "planning",
            },
        )
        self.credits.ensure_affordable(user_id, config, planning_prompt)
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "message": "開始生成執行計畫。",
            },
        )
        response = await provider.stream_generate(
            [
                {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ],
            on_delta=self._build_stream_progress_callback("計畫生成", progress_callback),
        )
        self.credits.charge(user_id, response.usage.cost)
        data = parse_json_object(response.content)
        plan = [str(item) for item in data.get("plan", []) if str(item).strip()]
        await self._emit_progress(
            progress_callback,
            {
                "type": "activity",
                "message": f"計畫生成完成，共 {len(plan or [])} 個步驟。",
            },
        )
        return plan or ["檢查需求", "更新檔案", "驗證語法"]

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

    async def _execute_actions(
        self,
        user_id: int,
        actions: list[dict[str, Any]],
        current_task_items: list[AgentTaskItem],
        progress_callback: ProgressCallback | None,
    ) -> tuple[list[dict[str, Any]], list[str], list[AgentTaskItem]]:
        results: list[dict[str, Any]] = []
        touched_files: list[str] = []
        for action in actions[: self.settings.agent_max_actions_per_iteration]:
            tool_name = action.get("tool")
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": self._format_tool_start_message(tool_name, action),
                },
            )
            try:
                if tool_name == "read_file":
                    results.append({"tool": tool_name, "path": action["path"], "result": self.workspace.read_file(user_id, action["path"])})
                elif tool_name == "write_file":
                    self.workspace.write_file(user_id, action["path"], action["content"])
                    touched_files.append(action["path"])
                    results.append({"tool": tool_name, "path": action["path"], "result": "ok"})
                elif tool_name == "list_files":
                    entries = self.workspace.list_files(user_id, action.get("path", "."))
                    results.append(
                        {
                            "tool": tool_name,
                            "path": action.get("path", "."),
                            "result": [
                                {"path": entry.path, "kind": entry.kind, "size": entry.size}
                                for entry in entries
                            ],
                        }
                    )
                elif tool_name == "delete_file":
                    self.workspace.delete_file(user_id, action["path"])
                    touched_files.append(action["path"])
                    results.append({"tool": tool_name, "path": action["path"], "result": "ok"})
                elif tool_name == "create_folder":
                    created_path = self.workspace.create_folder(user_id, action["path"])
                    results.append({"tool": tool_name, "path": created_path, "result": "ok"})
                elif tool_name == "apply_patch":
                    changed = self.workspace.apply_patch(user_id, action["diff"])
                    touched_files.extend(changed)
                    results.append({"tool": tool_name, "result": changed})
                elif tool_name == "py_compile_check":
                    outcome = self.workspace.py_compile_check(user_id, action["path"])
                    results.append({"tool": tool_name, "path": action["path"], "result": outcome})
                elif tool_name == "search_web":
                    outcome = await self._search_web(user_id, str(action["query"]))
                    results.append({"tool": tool_name, "query": action["query"], "result": outcome})
                elif tool_name == "fetch_url":
                    outcome = await self._fetch_url(user_id, str(action["url"]))
                    results.append({"tool": tool_name, "url": action["url"], "result": outcome})
                elif tool_name == "tasks":
                    current_task_items = self._coerce_task_items(action.get("items", []))
                    results.append(
                        {
                            "tool": tool_name,
                            "result": [
                                {"title": item.title, "status": item.status}
                                for item in current_task_items
                            ],
                        }
                    )
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
                else:
                    results.append({"tool": tool_name, "error": "不支援的工具。"})
            except (WorkspaceError, KeyError, ValueError, aiohttp.ClientError) as exc:
                results.append({"tool": tool_name, "error": str(exc)})
                await self._emit_progress(
                    progress_callback,
                    {
                        "type": "activity",
                        "message": f"工具 {tool_name} 執行失敗：{exc}",
                    },
                )
                continue
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": self._format_tool_finish_message(tool_name, action),
                },
            )
        return results, touched_files, current_task_items

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
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": "上下文接近上限，正在自動壓縮舊對話。",
                },
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
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": f"上下文壓縮完成，目前已壓縮 {compression_count} 次。",
                },
            )
        return history_messages, compression_count

    async def _emit_progress(self, progress_callback: ProgressCallback | None, event: dict[str, Any]) -> None:
        if progress_callback is None:
            return
        maybe_result = progress_callback(event)
        if maybe_result is not None:
            await maybe_result

    def _build_stream_progress_callback(
        self,
        label: str,
        progress_callback: ProgressCallback | None,
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
            await self._emit_progress(
                progress_callback,
                {
                    "type": "activity",
                    "message": f"{label} 串流中，已接收約 {received_chars} 字元。",
                },
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

    def _format_tool_start_message(self, tool_name: Any, action: dict[str, Any]) -> str:
        if tool_name == "read_file":
            return f"開始讀取檔案：{action.get('path', '')}"
        if tool_name == "write_file":
            return f"開始寫入檔案：{action.get('path', '')}"
        if tool_name == "list_files":
            return f"開始列出路徑：{action.get('path', '.') }"
        if tool_name == "delete_file":
            return f"開始刪除檔案：{action.get('path', '')}"
        if tool_name == "create_folder":
            return f"開始建立資料夾：{action.get('path', '')}"
        if tool_name == "apply_patch":
            return "開始套用 patch。"
        if tool_name == "py_compile_check":
            return f"開始語法檢查：{action.get('path', '')}"
        if tool_name == "search_web":
            return f"開始搜尋網路：{action.get('query', '')}"
        if tool_name == "fetch_url":
            return f"開始抓取網址：{action.get('url', '')}"
        if tool_name == "tasks":
            return "開始更新 tasks 清單。"
        return f"開始執行工具：{tool_name}"

    def _format_tool_finish_message(self, tool_name: Any, action: dict[str, Any]) -> str:
        if tool_name in {"read_file", "write_file", "delete_file", "create_folder", "py_compile_check"}:
            return f"工具 {tool_name} 已完成：{action.get('path', '')}"
        if tool_name == "list_files":
            return f"工具 list_files 已完成：{action.get('path', '.') }"
        if tool_name == "search_web":
            return "工具 search_web 已完成。"
        if tool_name == "fetch_url":
            return "工具 fetch_url 已完成。"
        if tool_name == "apply_patch":
            return "工具 apply_patch 已完成。"
        if tool_name == "tasks":
            return "tools 區塊中的 tasks 已更新。"
        return f"工具 {tool_name} 已完成。"

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


_AGENT_SYSTEM_PROMPT = """
你是運行在受限文字檔工作區中的 AI 程式代理。
重要規則：
- 使用者不能執行程式碼。
- 只能使用這些工具：read_file, write_file, list_files, delete_file, create_folder, apply_patch, py_compile_check, search_web, fetch_url, tasks。
- 編輯既有檔案時優先使用 apply_patch。
- 只可寫入 UTF-8 文字檔。
- 只回傳合法 JSON。
- summary 與 plan 內容請使用繁體中文。
- fetch_url 可直接抓取公開網址內容；若設定了 PROXY_* 環境變數，會透過 proxy 抓取，不需要先經過 search_web。
- 如果目前工作有明確步驟，請使用 tasks 工具更新工作清單，好讓使用者看到目前進度。
- JSON schema:
  {
        "summary": "簡短摘要",
    "done": true|false,
    "related_files": ["path"],
    "actions": [
      {"tool":"list_files","path":"."},
      {"tool":"read_file","path":"src/app.py"},
      {"tool":"write_file","path":"README.md","content":"..."},
      {"tool":"apply_patch","diff":"--- a/file.py\\n+++ b/file.py\\n@@ ..."},
      {"tool":"create_folder","path":"src"},
      {"tool":"delete_file","path":"old.py"},
      {"tool":"py_compile_check","path":"app.py"},
      {"tool":"search_web","query":"..."},
            {"tool":"fetch_url","url":"https://..."},
            {"tool":"tasks","items":[{"title":"檢查需求","status":"in_progress"},{"title":"更新 bot.py","status":"pending"}]}
    ]
  }
"""
