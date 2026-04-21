from __future__ import annotations

import asyncio
import io
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from agentcord.agent import CodingAgent, CreditManager
from agentcord.ai import create_provider
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.models import Provider, UserModelConfig
from agentcord.workspace import WorkspaceError, WorkspaceManager


def chunk_text(text: str, limit: int = 1800) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + limit])
        start += limit
    return chunks


class AgentCordBot(commands.Bot):
    def __init__(self, settings: Settings) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents, application_id=settings.discord_application_id)
        self.settings = settings
        self.db = Database(settings.data_dir / "agentcord.db", settings.default_credits)
        self.workspace = WorkspaceManager(settings.data_dir / "workspaces", settings.workspace_limit_bytes)
        self.http_session: aiohttp.ClientSession | None = None
        self.credits = CreditManager(self.db, settings)
        self.agent: CodingAgent | None = None

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession()
        self.agent = CodingAgent(self.settings, self.db, self.workspace, self.http_session)
        register_commands(self)
        await self.tree.sync()

    async def close(self) -> None:
        if self.http_session is not None:
            await self.http_session.close()
        self.db.close()
        await super().close()


def register_commands(bot: AgentCordBot) -> None:
    @bot.tree.command(name="ask", description="Ask the configured AI model a question.")
    @app_commands.describe(prompt="Your prompt for the AI assistant.")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        assert bot.http_session is not None
        config = bot.db.get_model_config(interaction.user.id, bot.settings.default_pollinations_model)
        provider = create_provider(bot.http_session, bot.settings, config)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are AgentCord, a coding-focused assistant inside Discord. "
                    "Do not claim code execution capabilities."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        bot.credits.ensure_affordable(interaction.user.id, config, prompt)
        await interaction.response.defer(ephemeral=True, thinking=True)
        response = await provider.generate(messages)
        remaining = bot.credits.charge(interaction.user.id, response.usage.cost)
        reply = (
            f"{response.content}\n\n"
            f"Credits used: {response.usage.cost:.2f} "
            f"(input={response.usage.input_tokens}, output={response.usage.output_tokens}, rate={response.usage.model_rate:.3f})\n"
            f"Remaining credits: {remaining:.2f}"
        )
        for index, chunk in enumerate(chunk_text(reply)):
            if index == 0:
                await interaction.followup.send(chunk, ephemeral=True)
            else:
                await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="agent", description="Run the multi-step coding agent.")
    @app_commands.describe(prompt="Describe what the agent should build or change.")
    async def agent_command(interaction: discord.Interaction, prompt: str) -> None:
        assert bot.agent is not None
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.agent.run(interaction.user.id, prompt)
        message = (
            f"Task #{result.task_id} complete.\n"
            f"Plan:\n- " + "\n- ".join(result.plan) + "\n\n"
            f"Summary:\n{result.summary}\n\n"
            f"Related files: {', '.join(result.related_files) if result.related_files else 'none'}\n"
            f"Validations:\n- " + ("\n- ".join(result.validations) if result.validations else "No Python files validated.")
        )
        for chunk in chunk_text(message):
            await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="call-ai-codeing", description="Alias for /agent.")
    @app_commands.describe(prompt="Describe what the agent should build or change.")
    async def agent_alias(interaction: discord.Interaction, prompt: str) -> None:
        await agent_command.callback(interaction, prompt)  # type: ignore[attr-defined]

    @bot.tree.command(name="set-model", description="Set your Pollinations model.")
    @app_commands.describe(model="The Pollinations model to use for /ask and /agent.")
    async def set_model(interaction: discord.Interaction, model: str) -> None:
        config = UserModelConfig(provider=Provider.POLLINATIONS, model=model, api_key=bot.settings.pollinations_api_key)
        bot.db.set_model_config(interaction.user.id, config)
        await interaction.response.send_message(
            f"Model set to Pollinations/{model}.", ephemeral=True
        )

    @bot.tree.command(name="custom-model", description="Configure a non-Pollinations provider.")
    @app_commands.describe(provider="openai / anthropic / google / xai / custom", api_key="Provider API key", model="Provider model name")
    async def custom_model(interaction: discord.Interaction, provider: str, api_key: str, model: str) -> None:
        provider_value = Provider(provider.lower())
        config = UserModelConfig(provider=provider_value, api_key=api_key.strip(), model=model.strip())
        bot.db.set_model_config(interaction.user.id, config)
        await interaction.response.send_message(
            f"Custom model set to {config.provider.value}/{config.model}.",
            ephemeral=True,
        )

    file_manager = app_commands.Group(name="file-manager", description="Browse and edit your workspace files.")

    @file_manager.command(name="list", description="List files in a folder.")
    @app_commands.describe(path="Folder path inside your workspace.")
    async def file_list(interaction: discord.Interaction, path: str = ".") -> None:
        entries = bot.workspace.list_files(interaction.user.id, path)
        total_size = bot.workspace.total_size(interaction.user.id)
        lines = [f"{entry.kind:6} {entry.size:>8} {entry.path}" for entry in entries] or ["(empty)"]
        await interaction.response.send_message(
            f"Workspace usage: {total_size}/{bot.settings.workspace_limit_bytes} bytes\n```text\n" + "\n".join(lines) + "\n```",
            ephemeral=True,
        )

    @file_manager.command(name="read", description="Read a text file.")
    async def file_read(interaction: discord.Interaction, path: str) -> None:
        content = bot.workspace.read_file(interaction.user.id, path)
        if len(content) <= 1800:
            await interaction.response.send_message(f"```text\n{content}\n```", ephemeral=True)
            return
        file = discord.File(io.BytesIO(content.encode("utf-8")), filename=Path(path).name or "file.txt")
        await interaction.response.send_message("File attached.", file=file, ephemeral=True)

    @file_manager.command(name="write", description="Write a UTF-8 text file.")
    async def file_write(interaction: discord.Interaction, path: str, content: str) -> None:
        size = bot.workspace.write_file(interaction.user.id, path, content)
        total_size = bot.workspace.total_size(interaction.user.id)
        await interaction.response.send_message(
            f"Wrote {size} bytes to {path}. Workspace usage: {total_size}/{bot.settings.workspace_limit_bytes}.",
            ephemeral=True,
        )

    @file_manager.command(name="delete", description="Delete a file.")
    async def file_delete(interaction: discord.Interaction, path: str) -> None:
        bot.workspace.delete_file(interaction.user.id, path)
        await interaction.response.send_message(f"Deleted {path}.", ephemeral=True)

    @file_manager.command(name="mkdir", description="Create a folder.")
    async def file_mkdir(interaction: discord.Interaction, path: str) -> None:
        created_path = bot.workspace.create_folder(interaction.user.id, path)
        await interaction.response.send_message(f"Created folder {created_path}.", ephemeral=True)

    bot.tree.add_command(file_manager)

    @bot.tree.command(name="export-zip", description="Export your workspace as a zip file.")
    async def export_zip(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        export_path = bot.settings.data_dir / "exports" / f"{interaction.user.id}.zip"
        bot.workspace.export_zip(interaction.user.id, export_path)
        file = discord.File(export_path, filename=f"workspace-{interaction.user.id}.zip")
        await interaction.followup.send("Workspace export ready.", file=file, ephemeral=True)

    @bot.command(name="add_credits")
    async def add_credits(ctx: commands.Context[AgentCordBot], member: discord.User, amount: float) -> None:
        if bot.settings.bot_owner_id is not None and ctx.author.id != bot.settings.bot_owner_id:
            await ctx.reply("Only the configured owner can modify credits.")
            return
        balance = bot.db.add_credits(member.id, amount)
        await ctx.reply(f"Updated {member.mention} credits by {amount:.2f}. New balance: {balance:.2f}")

    @ask.error
    @agent_command.error
    @agent_alias.error
    @set_model.error
    @custom_model.error
    @file_list.error
    @file_read.error
    @file_write.error
    @file_delete.error
    @file_mkdir.error
    @export_zip.error
    async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        original = getattr(error, "original", error)
        message = str(original)
        if isinstance(original, (WorkspaceError, ValueError, aiohttp.ClientError)):
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
            return
        raise error

    @add_credits.error
    async def on_add_credits_error(ctx: commands.Context[AgentCordBot], error: commands.CommandError) -> None:
        await ctx.reply(f"Command failed: {error}")


async def run_bot() -> None:
    settings = Settings.from_env()
    if not settings.discord_token:
        raise RuntimeError("DISCORD_TOKEN is required.")
    bot = AgentCordBot(settings)
    async with bot:
        await bot.start(settings.discord_token)


def main() -> None:
    asyncio.run(run_bot())
