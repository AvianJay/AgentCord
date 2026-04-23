from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentcord.bot import AgentCordBot
from agentcord.config import Settings


class _TokenExpiredError(Exception):
    def __init__(self, *, status: int = 401, code: int = 50027, message: str = "Invalid Webhook Token") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class _FakeResponse:
    def __init__(self, *, done: bool, error: Exception | None = None) -> None:
        self._done = done
        self._error = error
        self.messages: list[tuple[str | None, bool]] = []

    def is_done(self) -> bool:
        return self._done

    async def send_message(self, message: str | None = None, *, ephemeral: bool = False, **kwargs) -> None:
        del kwargs
        if self._error is not None:
            raise self._error
        self.messages.append((message, ephemeral))
        self._done = True


class _FakeFollowup:
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self.messages: list[tuple[str | None, bool]] = []

    async def send(self, message: str | None = None, *, ephemeral: bool = False, **kwargs) -> None:
        del kwargs
        if self._error is not None:
            raise self._error
        self.messages.append((message, ephemeral))


class _FakeUser:
    def __init__(self, *, mention: str = "@user", fail_dm: bool = False) -> None:
        self.mention = mention
        self._fail_dm = fail_dm
        self.messages: list[str] = []

    async def send(self, message: str | None = None, **kwargs) -> None:
        del kwargs
        if self._fail_dm:
            raise RuntimeError("dm blocked")
        self.messages.append(message or "")


class _FakeChannel:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def send(self, message: str | None = None, **kwargs) -> None:
        del kwargs
        self.messages.append(message or "")


class _FakeInteraction:
    def __init__(self, *, response_done: bool, error: Exception, fail_dm: bool = False) -> None:
        self.response = _FakeResponse(done=response_done)
        self.followup = _FakeFollowup(error=error)
        self.user = _FakeUser(fail_dm=fail_dm)
        self.channel = _FakeChannel()
        self.guild = None
        self.command = None


class InteractionResponseFallbackTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        settings = Settings(
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
        self.bot = AgentCordBot(settings)
        self.addCleanup(self.bot.db.close)

    async def test_ephemeral_followup_falls_back_to_dm_when_token_expired(self) -> None:
        interaction = _FakeInteraction(response_done=True, error=_TokenExpiredError())

        await self.bot.send_interaction_message(interaction, "hello", ephemeral=True)

        self.assertEqual(interaction.user.messages, ["互動已逾時，改以私訊傳送。\nhello"])
        self.assertEqual(interaction.channel.messages, [])

    async def test_public_followup_falls_back_to_channel_when_token_expired(self) -> None:
        interaction = _FakeInteraction(response_done=True, error=_TokenExpiredError())

        await self.bot.send_interaction_message(interaction, "hello", ephemeral=False)

        self.assertEqual(interaction.channel.messages, ["hello"])

    async def test_ephemeral_fallback_uses_channel_when_dm_unavailable(self) -> None:
        interaction = _FakeInteraction(response_done=True, error=_TokenExpiredError(), fail_dm=True)

        await self.bot.send_interaction_message(interaction, "secret", ephemeral=True)

        self.assertEqual(interaction.channel.messages, ["@user 互動已逾時，改以私訊傳送。\nsecret"])


if __name__ == "__main__":
    unittest.main()