from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aiohttp

from agentcord.agent import CodingAgent
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import AIResponse, AIUsage, ProviderModelInfo
from agentcord.workspace import WorkspaceManager


class _FakeProvider:
    def __init__(self, decisions: list[object], *, repair_responses: list[object] | None = None) -> None:
        self.decisions = list(decisions)
        self.repair_responses = list(repair_responses or [])
        self.contexts: list[str] = []
        self.generate_messages: list[list[dict[str, str]]] = []

    async def stream_generate(self, messages, on_delta=None):
        del on_delta
        self.contexts.append(str(messages[1]["content"]))
        decision = self.decisions.pop(0)
        return AIResponse(
            content=decision if isinstance(decision, str) else json.dumps(decision, ensure_ascii=False),
            usage=AIUsage(input_tokens=10, output_tokens=10, cost=0.0, model_rate=0.0),
            model="fake-model",
        )

    async def generate(self, messages):
        self.generate_messages.append(messages)
        response = self.repair_responses.pop(0)
        return AIResponse(
            content=response if isinstance(response, str) else json.dumps(response, ensure_ascii=False),
            usage=AIUsage(input_tokens=10, output_tokens=10, cost=0.0, model_rate=0.0),
            model="fake-model-repair",
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

    async def test_run_repairs_non_json_model_output_once(self) -> None:
        provider = _FakeProvider(
            ["看起來目前工具暫時無法使用。讓我再試一次讀取現有程式碼，才能規劃要加的功能。"],
            repair_responses=[
                {
                    "summary": "先重新讀取現有程式碼。",
                    "done": True,
                    "related_files": [],
                    "actions": [],
                }
            ],
        )

        with (
            mock.patch("agentcord.agent.create_provider", return_value=provider),
            mock.patch(
                "agentcord.agent.resolve_model_info",
                new=mock.AsyncMock(return_value=ProviderModelInfo(name="fake-model", context_length=32000)),
            ),
        ):
            result = await self.agent.run(123, "請繼續完成整個任務")

        self.assertEqual(result.summary, "先重新讀取現有程式碼。")
        self.assertEqual(len(provider.generate_messages), 1)
        self.assertIn("不是合法 JSON", provider.generate_messages[0][1]["content"])


class _NativeToolProvider:
    def __init__(self, responses: list[AIResponse]) -> None:
        self.responses = list(responses)
        self.tools_seen: list[object] = []
        self.system_prompts: list[str] = []

    async def stream_generate(self, messages, on_delta=None, **kwargs):
        del on_delta
        self.tools_seen.append(kwargs.get("tools"))
        self.system_prompts.append(str(messages[0]["content"]))
        return self.responses.pop(0)


def _native_response(content: str, tool_calls: list[dict[str, object]] | None = None) -> AIResponse:
    return AIResponse(
        content=content,
        usage=AIUsage(input_tokens=10, output_tokens=10, cost=0.0, model_rate=0.0),
        model="native-model",
        tool_calls=tool_calls or [],
    )


class AgentNativeToolTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_native_tool_calls_drive_actions(self) -> None:
        provider = _NativeToolProvider(
            [
                _native_response(
                    "",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "send_message",
                                "arguments": json.dumps({"message": "原生工具回報"}, ensure_ascii=False),
                            },
                        }
                    ],
                ),
                _native_response("整體需求已完成。"),
            ]
        )

        with (
            mock.patch("agentcord.agent.create_provider", return_value=provider),
            mock.patch(
                "agentcord.agent.resolve_model_info",
                new=mock.AsyncMock(
                    return_value=ProviderModelInfo(name="native-model", context_length=32000, tools=True)
                ),
            ),
        ):
            result = await self.agent.run(321, "請完成任務")

        self.assertIsNotNone(provider.tools_seen[0])
        self.assertEqual(result.summary, "整體需求已完成。")
        tool_messages = [message for message in result.messages if message.role == "tool"]
        self.assertIn("原生工具回報", tool_messages[0].content)

    async def test_native_unsupported_falls_back_to_simulated(self) -> None:
        class _FallbackProvider:
            def __init__(self) -> None:
                self.native_calls = 0
                self.simulated_calls = 0

            async def stream_generate(self, messages, on_delta=None, **kwargs):
                del on_delta, messages
                if kwargs.get("tools"):
                    self.native_calls += 1
                    raise aiohttp.ClientResponseError(
                        request_info=mock.Mock(),
                        history=(),
                        status=400,
                        message="tools not supported",
                    )
                self.simulated_calls += 1
                return AIResponse(
                    content=json.dumps(
                        {"summary": "模擬路徑完成", "done": True, "related_files": [], "actions": []},
                        ensure_ascii=False,
                    ),
                    usage=AIUsage(input_tokens=10, output_tokens=10, cost=0.0, model_rate=0.0),
                    model="fallback-model",
                )

        provider = _FallbackProvider()
        with (
            mock.patch("agentcord.agent.create_provider", return_value=provider),
            mock.patch(
                "agentcord.agent.resolve_model_info",
                new=mock.AsyncMock(
                    return_value=ProviderModelInfo(name="fallback-model", context_length=32000, tools=True)
                ),
            ),
        ):
            result = await self.agent.run(322, "請完成任務")

        self.assertEqual(provider.native_calls, 1)
        self.assertGreaterEqual(provider.simulated_calls, 1)
        self.assertEqual(result.summary, "模擬路徑完成")


if __name__ == "__main__":
    unittest.main()