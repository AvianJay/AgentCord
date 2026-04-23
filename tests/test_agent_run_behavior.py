from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agentcord.agent import CodingAgent
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import AIResponse, AIUsage, ProviderModelInfo
from agentcord.workspace import WorkspaceManager


class _FakeProvider:
    def __init__(self, decisions: list[dict[str, object]]) -> None:
        self.decisions = list(decisions)
        self.contexts: list[str] = []

    async def stream_generate(self, messages, on_delta=None):
        del on_delta
        self.contexts.append(str(messages[1]["content"]))
        decision = self.decisions.pop(0)
        return AIResponse(
            content=json.dumps(decision, ensure_ascii=False),
            usage=AIUsage(input_tokens=10, output_tokens=10, cost=0.0, model_rate=0.0),
            model="fake-model",
        )


class AgentRunBehaviorTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.settings = Settings(
            discord_token="",
            discord_application_id=None,
            bot_owner_id=None,
            discord_log_webhook="",
            data_dir=root / "data",
            workspace_limit_bytes=4096,
            default_credits=100,
            default_pollinations_model="openai",
            pollinations_api_key="",
            custom_provider_base_url="",
            proxy_url="",
            proxy_username="",
            proxy_password="",
            agent_max_iterations=4,
            agent_max_actions_per_iteration=4,
            credit_reserve_output_tokens=1024,
        )
        self.db = Database(root / "agentcord.db", default_credits=100)
        self.addCleanup(self.db.close)
        self.workspace = WorkspaceManager(root / "workspaces", limit_bytes=4096)
        self.agent = CodingAgent(self.settings, self.db, self.workspace, None)

    async def test_run_persists_tool_results_in_history(self) -> None:
        provider = _FakeProvider(
            [
                {
                    "summary": "先建立進度",
                    "done": True,
                    "related_files": [],
                    "actions": [
                        {
                            "tool": "tasks",
                            "items": [
                                {"title": "第一步", "status": "done"},
                                {"title": "第二步", "status": "done"},
                            ],
                        },
                        {"tool": "send_message", "message": "已完成第一步"},
                    ],
                }
            ]
        )

        with (
            mock.patch("agentcord.agent.create_provider", return_value=provider),
            mock.patch(
                "agentcord.agent.resolve_model_info",
                new=mock.AsyncMock(return_value=ProviderModelInfo(name="fake-model", context_length=32000)),
            ),
        ):
            result = await self.agent.run(123, "請繼續完成整個任務")

        self.assertEqual(len(provider.contexts), 1)
        tool_messages = [message for message in result.messages if message.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertIn("已完成第一步", tool_messages[0].content)
        self.assertIn("tool_results", tool_messages[0].content)
        self.assertEqual(result.summary, "先建立進度")
        self.assertTrue(all(item.status == "done" for item in result.task_items))


if __name__ == "__main__":
    unittest.main()