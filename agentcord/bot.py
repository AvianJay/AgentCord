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


COMMAND_NAME_TRANSLATIONS = {
    "prompt": "提示",
    "model": "模型",
    "provider": "供應商",
    "api_key": "金鑰",
    "path": "路徑",
    "content": "內容",
}


class CommandNameTranslator(app_commands.Translator):
    async def translate(
        self,
        string: app_commands.locale_str,
        locale: discord.Locale,
        context: app_commands.TranslationContext,
    ) -> str | None:
        if locale != discord.Locale.taiwan_chinese:
            return None
        allowed_locations = {
            app_commands.TranslationContextLocation.command_name,
            app_commands.TranslationContextLocation.group_name,
            app_commands.TranslationContextLocation.choice_name,
            app_commands.TranslationContextLocation.parameter_name,
        }
        if context.location not in allowed_locations:
            return None
        name = getattr(getattr(context, "data", None), "name", None)
        if isinstance(name, str):
            return COMMAND_NAME_TRANSLATIONS.get(name)
        return COMMAND_NAME_TRANSLATIONS.get(string.message)


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
        await self.tree.set_translator(CommandNameTranslator())
        await self.tree.sync()

    async def close(self) -> None:
        if self.http_session is not None:
            await self.http_session.close()
        self.db.close()
        await super().close()


def register_commands(bot: AgentCordBot) -> None:
    @bot.tree.command(name="ask", description="向目前設定的 AI 模型提問。")
    @app_commands.describe(prompt="輸入要交給 AI 助手的提示內容。")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        assert bot.http_session is not None
        config = bot.db.get_model_config(interaction.user.id, bot.settings.default_pollinations_model)
        provider = create_provider(bot.http_session, bot.settings, config)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 AgentCord，是 Discord 內專注於程式開發的助理。"
                    "不要宣稱自己可以直接執行程式碼。"
                    "除非使用者明確要求其他語言，否則請一律使用繁體中文回答。"
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
            f"已使用額度：{response.usage.cost:.2f} "
            f"(輸入={response.usage.input_tokens}，輸出={response.usage.output_tokens}，單價={response.usage.model_rate:.3f})\n"
            f"剩餘額度：{remaining:.2f}"
        )
        for index, chunk in enumerate(chunk_text(reply)):
            if index == 0:
                await interaction.followup.send(chunk, ephemeral=True)
            else:
                await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="agent", description="執行多步驟程式代理。")
    @app_commands.describe(prompt="描述要讓代理建立或修改的內容。")
    async def agent_command(interaction: discord.Interaction, prompt: str) -> None:
        assert bot.agent is not None
        await interaction.response.defer(ephemeral=True, thinking=True)
        result = await bot.agent.run(interaction.user.id, prompt)
        message = (
            f"任務 #{result.task_id} 已完成。\n"
            f"計畫：\n- " + "\n- ".join(result.plan) + "\n\n"
            f"摘要：\n{result.summary}\n\n"
            f"相關檔案：{', '.join(result.related_files) if result.related_files else '無'}\n"
            f"驗證：\n- " + ("\n- ".join(result.validations) if result.validations else "沒有驗證任何 Python 檔案。")
        )
        for chunk in chunk_text(message):
            await interaction.followup.send(chunk, ephemeral=True)

    @bot.tree.command(name="call-ai-codeing", description="/agent 的別名。")
    @app_commands.describe(prompt="描述要讓代理建立或修改的內容。")
    async def agent_alias(interaction: discord.Interaction, prompt: str) -> None:
        await agent_command.callback(interaction, prompt)  # type: ignore[attr-defined]

    @bot.tree.command(name="set-model", description="設定你的 Pollinations 模型。")
    @app_commands.describe(model="設定 /ask 與 /agent 使用的 Pollinations 模型。")
    async def set_model(interaction: discord.Interaction, model: str) -> None:
        config = UserModelConfig(provider=Provider.POLLINATIONS, model=model, api_key=bot.settings.pollinations_api_key)
        bot.db.set_model_config(interaction.user.id, config)
        await interaction.response.send_message(
            f"模型已設定為 Pollinations/{model}。", ephemeral=True
        )

    @bot.tree.command(name="custom-model", description="設定非 Pollinations 的模型供應商。")
    @app_commands.describe(provider="供應商類型：openai / anthropic / google / xai / custom", api_key="供應商 API 金鑰。", model="供應商模型名稱。")
    async def custom_model(interaction: discord.Interaction, provider: str, api_key: str, model: str) -> None:
        try:
            provider_value = Provider(provider.lower())
        except ValueError as exc:
            raise ValueError(f"不支援的供應商：{provider}") from exc
        config = UserModelConfig(provider=provider_value, api_key=api_key.strip(), model=model.strip())
        bot.db.set_model_config(interaction.user.id, config)
        await interaction.response.send_message(
            f"自訂模型已設定為 {config.provider.value}/{config.model}。",
            ephemeral=True,
        )

    file_manager = app_commands.Group(name="file-manager", description="瀏覽並編輯你的工作區檔案。")

    @file_manager.command(name="list", description="列出資料夾中的檔案。")
    @app_commands.describe(path="工作區內的資料夾路徑。")
    async def file_list(interaction: discord.Interaction, path: str = ".") -> None:
        entries = bot.workspace.list_files(interaction.user.id, path)
        total_size = bot.workspace.total_size(interaction.user.id)
        kind_labels = {"file": "檔案", "folder": "資料夾"}
        lines = [
            f"{kind_labels.get(entry.kind, entry.kind)} {entry.size:>8} {entry.path}"
            for entry in entries
        ] or ["(空白)"]
        await interaction.response.send_message(
            f"工作區用量：{total_size}/{bot.settings.workspace_limit_bytes} 位元組\n```text\n" + "\n".join(lines) + "\n```",
            ephemeral=True,
        )

    @file_manager.command(name="read", description="讀取文字檔。")
    async def file_read(interaction: discord.Interaction, path: str) -> None:
        content = bot.workspace.read_file(interaction.user.id, path)
        if len(content) <= 1800:
            await interaction.response.send_message(f"```text\n{content}\n```", ephemeral=True)
            return
        file = discord.File(io.BytesIO(content.encode("utf-8")), filename=Path(path).name or "file.txt")
        await interaction.response.send_message("檔案已附加。", file=file, ephemeral=True)

    @file_manager.command(name="write", description="寫入 UTF-8 文字檔。")
    async def file_write(interaction: discord.Interaction, path: str, content: str) -> None:
        size = bot.workspace.write_file(interaction.user.id, path, content)
        total_size = bot.workspace.total_size(interaction.user.id)
        await interaction.response.send_message(
            f"已寫入 {size} 位元組到 {path}。工作區用量：{total_size}/{bot.settings.workspace_limit_bytes}。",
            ephemeral=True,
        )

    @file_manager.command(name="delete", description="刪除檔案。")
    async def file_delete(interaction: discord.Interaction, path: str) -> None:
        bot.workspace.delete_file(interaction.user.id, path)
        await interaction.response.send_message(f"已刪除 {path}。", ephemeral=True)

    @file_manager.command(name="mkdir", description="建立資料夾。")
    async def file_mkdir(interaction: discord.Interaction, path: str) -> None:
        created_path = bot.workspace.create_folder(interaction.user.id, path)
        await interaction.response.send_message(f"已建立資料夾 {created_path}。", ephemeral=True)

    bot.tree.add_command(file_manager)

    @bot.tree.command(name="export-zip", description="將工作區匯出為 zip 壓縮檔。")
    async def export_zip(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        export_path = bot.settings.data_dir / "exports" / f"{interaction.user.id}.zip"
        bot.workspace.export_zip(interaction.user.id, export_path)
        file = discord.File(export_path, filename=f"workspace-{interaction.user.id}.zip")
        await interaction.followup.send("工作區匯出已準備完成。", file=file, ephemeral=True)

    @bot.command(name="add_credits")
    async def add_credits(ctx: commands.Context[AgentCordBot], member: discord.User, amount: float) -> None:
        if bot.settings.bot_owner_id is not None and ctx.author.id != bot.settings.bot_owner_id:
            await ctx.reply("只有設定中的擁有者可以調整額度。")
            return
        balance = bot.db.add_credits(member.id, amount)
        await ctx.reply(f"已為 {member.mention} 調整 {amount:.2f} 額度。新餘額：{balance:.2f}")

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
            if isinstance(original, aiohttp.ClientError):
                message = f"網路請求失敗：{message or type(original).__name__}"
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
            return
        raise error

    @add_credits.error
    async def on_add_credits_error(ctx: commands.Context[AgentCordBot], error: commands.CommandError) -> None:
        await ctx.reply(f"指令執行失敗：{error}")


async def run_bot() -> None:
    settings = Settings.from_env()
    if not settings.discord_token:
        raise RuntimeError("必須設定 DISCORD_TOKEN。")
    bot = AgentCordBot(settings)
    async with bot:
        await bot.start(settings.discord_token)


def main() -> None:
    asyncio.run(run_bot())
