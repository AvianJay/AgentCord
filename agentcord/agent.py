from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from agentcord.ai import PollinationsProvider, create_provider, parse_json_object
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import Provider, TaskStatus, UserModelConfig
from agentcord.workspace import WorkspaceError, WorkspaceManager


@dataclass(slots=True)
class AgentRunResult:
    summary: str
    plan: list[str]
    related_files: list[str] = field(default_factory=list)
    validations: list[str] = field(default_factory=list)
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
                f"Insufficient credits. Estimated minimum required: {estimated_cost:.2f}, "
                f"available: {self.db.get_credits(user_id):.2f}."
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

    async def run(self, user_id: int, prompt: str) -> AgentRunResult:
        config = self.db.get_model_config(user_id, self.settings.default_pollinations_model)
        provider = create_provider(self.session, self.settings, config)
        task = self.db.create_task(user_id, title=prompt[:120], status=TaskStatus.RUNNING)

        plan = await self._create_plan(user_id, prompt, provider, config)
        transcript: list[dict[str, str]] = []
        changed_files: set[str] = set()
        validations: list[str] = []
        final_summary = "No changes were made."

        for _ in range(self.settings.agent_max_iterations):
            context = self._build_iteration_context(user_id, prompt, plan, transcript)
            self.credits.ensure_affordable(user_id, config, context)
            step_response = await provider.generate(
                [
                    {"role": "system", "content": _AGENT_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ]
            )
            self.credits.charge(user_id, step_response.usage.cost)
            decision = parse_json_object(step_response.content)
            tool_results, touched_files = await self._execute_actions(user_id, decision.get("actions", []))
            changed_files.update(touched_files)

            validations.extend(self._validate_changed_python_files(user_id, touched_files))
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
            if decision.get("done"):
                break

        related_files = sorted(changed_files)
        self.db.update_task(task.id, TaskStatus.DONE, related_files)
        return AgentRunResult(
            summary=final_summary,
            plan=plan,
            related_files=related_files,
            validations=validations,
            task_id=task.id,
        )

    async def _create_plan(
        self,
        user_id: int,
        prompt: str,
        provider: Any,
        config: UserModelConfig,
    ) -> list[str]:
        planning_prompt = (
            "Create a concise execution plan for the coding task below. "
            "Return JSON with a single top-level key named plan containing a list of short strings.\n\n"
            f"Workspace tree:\n{self.workspace.dump_tree(user_id)}\n\n"
            f"Task:\n{prompt}"
        )
        self.credits.ensure_affordable(user_id, config, planning_prompt)
        response = await provider.generate(
            [
                {"role": "system", "content": _PLANNING_SYSTEM_PROMPT},
                {"role": "user", "content": planning_prompt},
            ]
        )
        self.credits.charge(user_id, response.usage.cost)
        data = parse_json_object(response.content)
        plan = [str(item) for item in data.get("plan", []) if str(item).strip()]
        return plan or ["Review the request", "Update files", "Validate syntax"]

    def _build_iteration_context(
        self,
        user_id: int,
        prompt: str,
        plan: list[str],
        transcript: list[dict[str, str]],
    ) -> str:
        return (
            f"User request:\n{prompt}\n\n"
            f"Current plan:\n{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            f"Workspace tree:\n{self.workspace.dump_tree(user_id)}\n\n"
            f"Prior tool transcript:\n{json.dumps(transcript[-6:], ensure_ascii=False, indent=2)}\n\n"
            "Return JSON with keys summary, done, related_files, and actions. "
            f"Use at most {self.settings.agent_max_actions_per_iteration} actions."
        )

    async def _execute_actions(self, user_id: int, actions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
        results: list[dict[str, Any]] = []
        touched_files: list[str] = []
        for action in actions[: self.settings.agent_max_actions_per_iteration]:
            tool_name = action.get("tool")
            try:
                if tool_name == "read_file":
                    results.append({"tool": tool_name, "path": action["path"], "result": self.workspace.read_file(user_id, action["path"])})
                elif tool_name == "write_file":
                    self.workspace.write_file(user_id, action["path"], action["content"])
                    touched_files.append(action["path"])
                    results.append({"tool": tool_name, "path": action["path"], "result": "ok"})
                elif tool_name == "list_files":
                    entries = self.workspace.list_files(user_id, action.get("path", "."))
                    results.append({"tool": tool_name, "path": action.get("path", "."), "result": [entry.__dict__ for entry in entries]})
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
                else:
                    results.append({"tool": tool_name, "error": "Unsupported tool."})
            except (WorkspaceError, KeyError, ValueError, aiohttp.ClientError) as exc:
                results.append({"tool": tool_name, "error": str(exc)})
        return results, touched_files

    def _validate_changed_python_files(self, user_id: int, touched_files: list[str]) -> list[str]:
        validations: list[str] = []
        for path in sorted({item for item in touched_files if item.endswith(".py")}):
            try:
                validations.append(self.workspace.py_compile_check(user_id, path))
            except Exception as exc:  # noqa: BLE001
                validations.append(f"Syntax error in {path}: {exc}")
        return validations

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
                        "Search the web and return JSON with a top-level results list. "
                        "Each item must include title, url, and summary."
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
        if not self.db.is_allowed_url(user_id, url):
            raise ValueError("fetch_url is only allowed for URLs returned by search_web.")
        async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=45)) as response:
            response.raise_for_status()
            return await response.text()


_PLANNING_SYSTEM_PROMPT = """
You are an AI coding planner for a Discord bot workspace.
Return valid JSON only.
"""


_AGENT_SYSTEM_PROMPT = """
You are an AI coding agent operating in a sandboxed text-file workspace.
Important rules:
- Users cannot execute code.
- Only use these tools: read_file, write_file, list_files, delete_file, create_folder, apply_patch, py_compile_check, search_web, fetch_url.
- Prefer apply_patch when editing an existing file.
- Only write UTF-8 text files.
- Return valid JSON only.
- JSON schema:
  {
    "summary": "short summary",
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
      {"tool":"fetch_url","url":"https://..."}
    ]
  }
"""
