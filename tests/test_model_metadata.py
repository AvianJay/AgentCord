from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agentcord import ai
from agentcord.agent import CodingAgent
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import AIUsage, Provider, ProviderModelInfo, UserModelConfig
from agentcord.workspace import WorkspaceManager


class _FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, payloads: dict[str, list[object]]) -> None:
        self._payloads = {url: list(items) for url, items in payloads.items()}
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        queue = self._payloads.get(url)
        if not queue:
            raise AssertionError(f"Unexpected GET {url}")
        return _FakeResponse(queue.pop(0))

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        queue = self._payloads.get(url)
        if not queue:
            raise AssertionError(f"Unexpected POST {url}")
        return _FakeResponse(queue.pop(0))


def _make_settings(root: Path) -> Settings:
    return Settings(
        discord_token="",
        discord_application_id=None,
        bot_owner_id=None,
        discord_log_webhook="",
        data_dir=root / "data",
        workspace_limit_bytes=1024 * 1024,
        default_credits=100,
        default_pollinations_model="openai",
        pollinations_api_key="",
        custom_provider_base_url="https://proxy.example.com/v1",
        proxy_url="",
        proxy_username="",
        proxy_password="",
        agent_max_iterations=4,
        agent_max_actions_per_iteration=4,
        credit_reserve_output_tokens=1024,
    )


class ProviderModelMetadataTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_openai_compatible_models_reads_context_length(self) -> None:
        session = _FakeSession(
            {
                "https://api.openai.com/v1/models": [
                    {
                        "data": [
                            {
                                "id": "gpt-4.1",
                                "description": "Flagship model",
                                "context_length": 128000,
                            },
                            {
                                "id": "gpt-4.1-mini",
                                "owned_by": "openai",
                                "max_context_length": 1047576,
                            },
                        ]
                    }
                ]
            }
        )
        settings = _make_settings(Path(tempfile.gettempdir()))

        models = await ai.fetch_provider_models(
            session,
            settings,
            Provider.OPENAI,
            "sk-test",
            force_refresh=True,
        )

        self.assertEqual([model.name for model in models], ["gpt-4.1", "gpt-4.1-mini"])
        self.assertEqual(models[0].context_length, 128000)
        self.assertEqual(models[1].context_length, 1047576)

    async def test_fetch_google_models_normalizes_name_and_filters_generation_models(self) -> None:
        session = _FakeSession(
            {
                "https://generativelanguage.googleapis.com/v1beta/models": [
                    {
                        "models": [
                            {
                                "name": "models/gemini-2.5-pro",
                                "displayName": "Gemini 2.5 Pro",
                                "description": "Reasoning model",
                                "inputTokenLimit": 1048576,
                                "supportedGenerationMethods": ["generateContent", "countTokens"],
                            },
                            {
                                "name": "models/text-embedding-004",
                                "displayName": "Embedding",
                                "inputTokenLimit": 2048,
                                "supportedGenerationMethods": ["embedContent"],
                            },
                        ]
                    }
                ]
            }
        )
        settings = _make_settings(Path(tempfile.gettempdir()))

        models = await ai.fetch_provider_models(
            session,
            settings,
            Provider.GOOGLE,
            "google-key",
            force_refresh=True,
        )
        resolved = await ai.resolve_provider_model(
            session,
            settings,
            Provider.GOOGLE,
            "google-key",
            "models/gemini-2.5-pro",
        )

        self.assertEqual([model.name for model in models], ["gemini-2.5-pro"])
        self.assertEqual(models[0].context_length, 1048576)
        self.assertIsNotNone(resolved)
        assert resolved is not None
        self.assertEqual(resolved.name, "gemini-2.5-pro")
        self.assertEqual(resolved.context_length, 1048576)

    async def test_fetch_poe_models_uses_poe_openai_compatible_catalog(self) -> None:
        session = _FakeSession(
            {
                "https://api.poe.com/v1/models": [
                    {
                        "data": [
                            {
                                "id": "Claude-Sonnet-4.5",
                                "description": "Claude Sonnet on Poe",
                                "owned_by": "Anthropic",
                            },
                            {
                                "id": "GPT-5-Pro",
                                "description": "OpenAI flagship on Poe",
                                "owned_by": "OpenAI",
                            },
                        ]
                    }
                ]
            }
        )
        settings = _make_settings(Path(tempfile.gettempdir()))

        models = await ai.fetch_provider_models(
            session,
            settings,
            Provider.POE,
            "poe-test-key",
            force_refresh=True,
        )

        self.assertEqual([model.name for model in models], ["Claude-Sonnet-4.5", "GPT-5-Pro"])
        self.assertEqual(models[0].description, "Claude Sonnet on Poe")
        self.assertIsNone(models[0].context_length)

    async def test_fetch_custom_models_uses_combined_apiurl_and_proxy(self) -> None:
        session = _FakeSession(
            {
                "https://api.example.com/v1/models": [
                    {
                        "data": [
                            {
                                "id": "custom-coder",
                                "description": "Private coding model",
                            }
                        ]
                    }
                ]
            }
        )
        settings = _make_settings(Path(tempfile.gettempdir()))
        settings.proxy_url = "http://proxy.local:8080"

        models = await ai.fetch_provider_models(
            session,
            settings,
            Provider.CUSTOM,
            "https://api.example.com/v1:secret-token",
            force_refresh=True,
        )

        self.assertEqual([model.name for model in models], ["custom-coder"])
        self.assertEqual(session.calls[0][0], "https://api.example.com/v1/models")
        self.assertEqual(session.calls[0][1]["headers"]["Authorization"], "Bearer secret-token")
        self.assertEqual(session.calls[0][1]["proxy"], "http://proxy.local:8080")

    def test_create_provider_supports_poe_openai_compatible_endpoint(self) -> None:
        settings = _make_settings(Path(tempfile.gettempdir()))
        config = UserModelConfig(provider=Provider.POE, api_key="poe-test-key", model="Claude-Sonnet-4.5")

        provider = ai.create_provider(object(), settings, config)

        self.assertIsInstance(provider, ai.OpenAICompatibleProvider)
        self.assertEqual(provider.base_url, "https://api.poe.com/v1")

    def test_create_provider_supports_custom_combined_apiurl_and_api_key(self) -> None:
        settings = _make_settings(Path(tempfile.gettempdir()))
        config = UserModelConfig(
            provider=Provider.CUSTOM,
            api_key="https://api.example.com/v1:secret-token",
            model="custom-coder",
        )

        provider = ai.create_provider(object(), settings, config)

        self.assertIsInstance(provider, ai.OpenAICompatibleProvider)
        self.assertEqual(provider.base_url, "https://api.example.com/v1")
        self.assertEqual(provider.request_api_key, "secret-token")
        self.assertTrue(provider.require_proxy)

    async def test_openai_compatible_provider_strips_thinking_preface(self) -> None:
        session = _FakeSession(
            {
                "https://api.openai.com/v1/chat/completions": [
                    {
                        "model": "gpt-4.1",
                        "choices": [
                            {
                                "message": {
                                    "content": "Thinking...\n\n真正答案"
                                }
                            }
                        ],
                    }
                ]
            }
        )
        settings = _make_settings(Path(tempfile.gettempdir()))
        provider = ai.OpenAICompatibleProvider(
            session,
            settings,
            UserModelConfig(provider=Provider.OPENAI, api_key="sk-test", model="gpt-4.1"),
            "https://api.openai.com/v1",
        )

        response = await provider.generate([{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "真正答案")

    def test_parse_json_object_ignores_thinking_blocks(self) -> None:
        payload = ai.parse_json_object("<think>private reasoning</think>\n```thinking\nstep 1\n```\n{\"plan\": [\"a\"]}")

        self.assertEqual(payload, {"plan": ["a"]})

    def test_parse_json_object_invalid_json_exposes_preview_on_error(self) -> None:
        with self.assertRaises(ai.ModelJSONParseError) as context:
            ai.parse_json_object("```text\nnot json yet\nline two\n```")

        self.assertIn("模型輸出不包含合法 JSON 物件。", str(context.exception))
        self.assertIn("輸出預覽：", str(context.exception))
        self.assertIn("not json yet", context.exception.output_preview)
        formatted = ai.format_exception_message(context.exception)
        self.assertIn("輸出預覽：", formatted)
        self.assertIn("not json yet", formatted)

    def test_parse_json_object_empty_output_uses_explicit_placeholder(self) -> None:
        with self.assertRaises(ai.ModelJSONParseError) as context:
            ai.parse_json_object("   \n\t  ")

        self.assertEqual(context.exception.output_preview, "(空白輸出)")
        self.assertIn("(空白輸出)", str(context.exception))

    def test_format_exception_message_redacts_google_api_key_in_url(self) -> None:
        error = RuntimeError(
            "402, message='Payment Required', url='https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key=AIzaSecretKey123'"
        )

        message = ai.format_exception_message(error)

        self.assertIn("key=%2A%2A%2A", message)
        self.assertNotIn("AIzaSecretKey123", message)


class AgentModelMetadataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        root = Path(self._tempdir.name)
        self.settings = _make_settings(root)
        self.db = Database(root / "agentcord.db", default_credits=100)
        self.addCleanup(self.db.close)
        self.workspace = WorkspaceManager(root / "workspaces", limit_bytes=1024 * 1024)
        self.agent = CodingAgent(self.settings, self.db, self.workspace, None)
        self.user_id = 555
        self.db.set_model_config(
            self.user_id,
            UserModelConfig(provider=Provider.OPENAI, api_key="sk-test", model="gpt-4.1"),
        )

    async def test_plan_uses_resolved_model_info_for_custom_provider_context_length(self) -> None:
        usage = AIUsage(input_tokens=10, output_tokens=20, cost=0.0, model_rate=0.0)
        fake_create_plan = AsyncMock(return_value=(["step 1"], "gpt-4.1", usage))

        with (
            patch("agentcord.agent.create_provider", return_value=object()),
            patch(
                "agentcord.agent.resolve_model_info",
                new=AsyncMock(return_value=ProviderModelInfo(name="gpt-4.1", context_length=128000)),
            ),
            patch.object(self.agent, "_create_plan", fake_create_plan),
        ):
            result = await self.agent.plan(self.user_id, "寫個測試")

        self.assertEqual(result.context_length, 128000)
        self.assertEqual(result.model, "gpt-4.1")
        self.assertEqual(fake_create_plan.await_args.args[6], 128000)


if __name__ == "__main__":
    unittest.main()