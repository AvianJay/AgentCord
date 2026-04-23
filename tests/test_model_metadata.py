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

    def test_create_provider_supports_poe_openai_compatible_endpoint(self) -> None:
        settings = _make_settings(Path(tempfile.gettempdir()))
        config = UserModelConfig(provider=Provider.POE, api_key="poe-test-key", model="Claude-Sonnet-4.5")

        provider = ai.create_provider(object(), settings, config)

        self.assertIsInstance(provider, ai.OpenAICompatibleProvider)
        self.assertEqual(provider.base_url, "https://api.poe.com/v1")


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