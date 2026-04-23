from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.agent import CreditManager
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import Provider, UserModelConfig


class CreditManagerTests(unittest.TestCase):
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
            workspace_limit_bytes=1024,
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
        self.credits = CreditManager(self.db, self.settings)
        self.user_id = 123

    def test_pollinations_model_still_charges_credits(self) -> None:
        config = UserModelConfig(provider=Provider.POLLINATIONS, model="openai")

        remaining = self.credits.charge(self.user_id, config, 12.5)

        self.assertEqual(remaining, 87.5)
        self.assertTrue(self.credits.should_charge(config))
        self.assertEqual(self.credits.billed_amount(config, 12.5), 12.5)

    def test_custom_model_provider_does_not_charge_credits(self) -> None:
        config = UserModelConfig(provider=Provider.OPENAI, model="gpt-4.1", api_key="sk-test")

        self.credits.ensure_affordable(self.user_id, config, "hello world")
        remaining = self.credits.charge(self.user_id, config, 12.5)

        self.assertEqual(remaining, 100.0)
        self.assertFalse(self.credits.should_charge(config))
        self.assertEqual(self.credits.billed_amount(config, 12.5), 0.0)


if __name__ == "__main__":
    unittest.main()