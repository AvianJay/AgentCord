from __future__ import annotations

import asyncio
import io
import traceback
from pathlib import Path

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from agentcord.agent import CodingAgent, CreditManager
from agentcord.ai import create_provider, fetch_pollinations_models, resolve_pollinations_model
from agentcord.config import Settings
from agentcord.database import Database
from agentcord.logger import DiscordWebhookLogger
from agentcord.live_agent import AgentConversationSession
from agentcord.models import Provider, TaskRecord, TaskStatus, UserModelConfig, UserPterodactylConfig
from agentcord.pterodactyl import fetch_pterodactyl_account
from agentcord.workspace import WorkspaceError, WorkspaceManager


COMMAND_NAME_TRANSLATIONS = {
    "ask": "詢問",
    "agent": "代理",
    "plan": "計畫",
    "call-ai-codeing": "叫ai寫程式",
    "agent-history": "代理歷史",
    "agent-open": "開啟代理對話",
    "delete-agent": "刪除代理對話",
    "set-model": "設定模型",
    "set-pterodactyl": "設定翼手龍",
    "custom-model": "自訂模型",
    "file-manager": "檔案管理",
    "list": "列表",
    "read": "讀取",
    "write": "寫入",
    "delete": "刪除",
    "mkdir": "建立資料夾",
    "rmdir": "刪除資料夾",
    "export-zip": "匯出壓縮檔",
    "import-zip": "匯入壓縮檔",
    "prompt": "提示",
    "model": "模型",
    "provider": "供應商",
    "api_key": "金鑰",
    "base_url": "api網址",
    "archive": "壓縮檔",
    "force": "強制",
    "path": "路徑",
    "content": "內容",
    "task_id": "任務編號",
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
        allowed_mentions = discord.AllowedMentions.none()
        super().__init__(command_prefix="!", intents=intents, application_id=settings.discord_application_id, allowed_mentions=allowed_mentions)
        self.settings = settings
        self.db = Database(settings.data_dir / "agentcord.db", settings.default_credits)
        self.workspace = WorkspaceManager(settings.data_dir / "workspaces", settings.workspace_limit_bytes)
        self.http_session: aiohttp.ClientSession | None = None
        self.credits = CreditManager(self.db, settings)
        self.agent: CodingAgent | None = None
        self.agent_sessions: dict[int, AgentConversationSession] = {}
        self.logger: DiscordWebhookLogger | None = None

    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession()
        if self.settings.discord_log_webhook:
            self.logger = DiscordWebhookLogger(self.settings.discord_log_webhook, self.http_session)
        self.agent = CodingAgent(self.settings, self.db, self.workspace, self.http_session)
        register_commands(self)
        await self.tree.set_translator(CommandNameTranslator())
        await self.tree.sync()

    async def close(self) -> None:
        for session in list(self.agent_sessions.values()):
            await session.close("Bot 正在關閉，這個對話已停止更新。")
        self.agent_sessions.clear()
        if self.logger is not None:
            await self.logger.close()
            self.logger = None
        if self.http_session is not None:
            await self.http_session.close()
        self.db.close()
        await super().close()

    async def log_event(
        self,
        title: str,
        description: str,
        *,
        user: discord.abc.User | None = None,
        guild: discord.Guild | None = None,
        color: discord.Colour | None = None,
        fields: list[tuple[str, str, bool]] | None = None,
    ) -> None:
        if self.logger is None:
            return
        await self.logger.log(
            title,
            description,
            user=user,
            guild=guild,
            color=color,
            fields=fields,
        )

    async def log_exception(
        self,
        title: str,
        error: BaseException,
        *,
        user: discord.abc.User | None = None,
        guild: discord.Guild | None = None,
        details: str = "",
        fields: list[tuple[str, str, bool]] | None = None,
    ) -> None:
        if self.logger is None:
            return
        await self.logger.log_exception(
            title,
            error,
            user=user,
            guild=guild,
            details=details,
            fields=fields,
        )


def register_commands(bot: AgentCordBot) -> None:
    def preview_text(text: str, limit: int = 300) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3] + "..."

    async def log_interaction(
        interaction: discord.Interaction,
        description: str,
        *,
        fields: list[tuple[str, str, bool]] | None = None,
        color: discord.Colour | None = None,
    ) -> None:
        command_name = interaction.command.qualified_name if interaction.command else "interaction"
        await bot.log_event(
            f"/{command_name}",
            description,
            user=interaction.user,
            guild=interaction.guild,
            color=color,
            fields=fields,
        )

    async def log_command_error(interaction: discord.Interaction, error: BaseException) -> None:
        command_name = interaction.command.qualified_name if interaction.command else "interaction"
        fields: list[tuple[str, str, bool]] = []
        if interaction.channel is not None and hasattr(interaction.channel, "id"):
            fields.append(("頻道", str(interaction.channel.id), True))
        await bot.log_exception(
            f"/{command_name} 失敗",
            error,
            user=interaction.user,
            guild=interaction.guild,
            details="應用程式指令執行失敗。",
            fields=fields,
        )

    async def close_replaceable_session(user_id: int) -> None:
        existing_session = bot.agent_sessions.get(user_id)
        if existing_session is None:
            return
        if existing_session.is_busy():
            raise ValueError("你已經有一個 agent 對話正在執行，請先等待完成或按下中斷。")
        await existing_session.close("這個對話已被新的 agent 會話取代。")
        if bot.agent_sessions.get(user_id) is existing_session:
            bot.agent_sessions.pop(user_id, None)

    async def open_agent_session(
        interaction: discord.Interaction,
        task: TaskRecord,
        *,
        reopened: bool = False,
    ) -> AgentConversationSession:
        await close_replaceable_session(interaction.user.id)
        session = AgentConversationSession(bot, interaction.user, task)
        bot.agent_sessions[interaction.user.id] = session
        await session.open(interaction, reopened=reopened)
        return session

    async def start_new_agent_session(
        interaction: discord.Interaction,
        prompt: str,
    ) -> AgentConversationSession:
        await close_replaceable_session(interaction.user.id)
        task = bot.db.create_task(interaction.user.id, title=prompt[:120], status=TaskStatus.PENDING)
        session = AgentConversationSession(bot, interaction.user, task)
        bot.agent_sessions[interaction.user.id] = session
        await session.open(interaction)
        return session

    def format_task_status(status: TaskStatus) -> str:
        if status is TaskStatus.DONE:
            return "完成"
        if status is TaskStatus.RUNNING:
            return "執行中"
        if status is TaskStatus.CANCELLED:
            return "已中斷"
        if status is TaskStatus.FAILED:
            return "失敗"
        return "待命"

    def format_model_choice_name(model_name: str, context_length: int | None, description: str) -> str:
        context_text = f"{context_length:,} ctx" if context_length else "ctx ?"
        label = f"{model_name} | {context_text}"
        if description:
            label = f"{label} | {description}"
        if len(label) > 100:
            return label[:97] + "..."
        return label

    def is_public_agent_command(interaction: discord.Interaction) -> bool:
        command = interaction.command
        if command is None:
            return False
        return command.name in {"agent", "plan", "call-ai-codeing", "agent-history", "agent-open"}

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
        await log_interaction(
            interaction,
            f"使用 /ask 提問。\nPrompt: {preview_text(prompt)}",
            fields=[
                ("模型", f"{config.provider.value}/{config.model}", True),
                ("花費", f"{response.usage.cost:.2f}", True),
            ],
        )
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
        session = await start_new_agent_session(interaction, prompt)
        await log_interaction(
            interaction,
            f"建立新的 agent 對話。\nPrompt: {preview_text(prompt)}",
            fields=[("Task ID", str(session.task_id), True)],
        )
        await session.enqueue_prompt(prompt)

    @bot.tree.command(name="plan", description="為任務生成執行計畫。")
    @app_commands.describe(prompt="描述要讓 AI 規劃的程式任務。")
    async def plan_command(interaction: discord.Interaction, prompt: str) -> None:
        assert bot.agent is not None
        await interaction.response.defer(thinking=True)
        result = await bot.agent.plan(interaction.user.id, prompt)
        usage = result.usage
        await log_interaction(
            interaction,
            f"建立獨立執行計畫。\nPrompt: {preview_text(prompt)}",
            fields=[
                ("模型", result.model or bot.db.get_model_config(interaction.user.id, bot.settings.default_pollinations_model).model, True),
                ("步驟", str(len(result.plan)), True),
            ],
        )
        lines = ["執行計畫："]
        for index, step in enumerate(result.plan, start=1):
            lines.append(f"{index}. {step}")
        if usage is not None:
            remaining = bot.db.get_credits(interaction.user.id)
            lines.extend(
                [
                    "",
                    f"模型：{result.model or '(未設定)'}",
                    f"已使用額度：{usage.cost:.2f} (輸入={usage.input_tokens}，輸出={usage.output_tokens}，單價={usage.model_rate:.3f})",
                    f"剩餘額度：{remaining:.2f}",
                ]
            )
        message = "\n".join(lines)
        for index, chunk in enumerate(chunk_text(message)):
            if index == 0:
                await interaction.followup.send(chunk)
            else:
                await interaction.followup.send(chunk)

    @bot.tree.command(name="call-ai-codeing", description="叫 AI 寫程式。")
    @app_commands.describe(prompt="描述要讓代理建立或修改的內容。")
    async def agent_alias(interaction: discord.Interaction, prompt: str) -> None:
        session = await start_new_agent_session(interaction, prompt)
        await log_interaction(
            interaction,
            f"透過 alias 建立新的 agent 對話。\nPrompt: {preview_text(prompt)}",
            fields=[("Task ID", str(session.task_id), True)],
        )
        await session.enqueue_prompt(prompt)

    @bot.tree.command(name="agent-history", description="查看最近 20 筆 agent 對話。")
    async def agent_history(interaction: discord.Interaction) -> None:
        tasks = bot.db.list_tasks(interaction.user.id, limit=20)
        await log_interaction(
            interaction,
            f"查看 agent 歷史，共 {len(tasks)} 筆。",
        )
        if not tasks:
            await interaction.response.send_message("目前沒有任何 agent 對話歷史。")
            return
        lines = ["最近 20 筆 agent 對話："]
        for task in tasks:
            timestamp = f"<t:{task.updated_at}:R>" if task.updated_at else "時間未知"
            model_name = task.model or bot.db.get_model_config(interaction.user.id, bot.settings.default_pollinations_model).model
            lines.append(
                f"#{task.id} [{format_task_status(task.status)}] {task.title} | {model_name} | {timestamp}"
            )
        lines.append("使用 /agent-open 並帶入 task_id 可重新打開對話。")
        message = "\n".join(lines)
        for index, chunk in enumerate(chunk_text(message)):
            if index == 0:
                await interaction.response.send_message(chunk)
            else:
                await interaction.followup.send(chunk)

    @bot.tree.command(name="agent-open", description="重新打開既有的 agent 對話。")
    @app_commands.describe(task_id="要重新打開的對話編號。", prompt="可選：重新打開後立刻送出的新訊息。")
    async def agent_open(interaction: discord.Interaction, task_id: int, prompt: str | None = None) -> None:
        task = bot.db.get_task(interaction.user.id, task_id)
        session = await open_agent_session(interaction, task, reopened=True)
        await log_interaction(
            interaction,
            "重新打開既有 agent 對話。" + (f"\nPrompt: {preview_text(prompt)}" if prompt else ""),
            fields=[("Task ID", str(task_id), True)],
        )
        if prompt and prompt.strip():
            await session.enqueue_prompt(prompt)

    @bot.tree.command(name="delete-agent", description="刪除既有的 agent 對話。")
    @app_commands.describe(task_id="要刪除的對話編號。")
    async def delete_agent(interaction: discord.Interaction, task_id: int) -> None:
        task = bot.db.get_task(interaction.user.id, task_id)
        existing_session = bot.agent_sessions.get(interaction.user.id)
        if existing_session is not None and existing_session.task_id == task_id:
            if existing_session.is_busy():
                raise ValueError("這個 agent 對話正在執行，請先等待完成或按下中斷。")
            await existing_session.close("這個對話已刪除。")
            if bot.agent_sessions.get(interaction.user.id) is existing_session:
                bot.agent_sessions.pop(interaction.user.id, None)

        bot.workspace.clear_task_review_storage(interaction.user.id, task_id)
        deleted_task = bot.db.delete_task(interaction.user.id, task_id)
        await log_interaction(
            interaction,
            "刪除既有 agent 對話。",
            fields=[
                ("Task ID", str(task_id), True),
                ("標題", preview_text(deleted_task.title, 100), False),
            ],
        )
        await interaction.response.send_message(
            f"已刪除 agent 對話 #{deleted_task.id}：{deleted_task.title}",
            ephemeral=True,
        )

    @bot.tree.command(name="set-model", description="設定你的 Pollinations 模型。")
    @app_commands.describe(model="設定 /ask 與 /agent 使用的 Pollinations 模型。")
    async def set_model(interaction: discord.Interaction, model: str) -> None:
        assert bot.http_session is not None
        model_info = await resolve_pollinations_model(bot.http_session, bot.settings, model)
        if model_info is None:
            raise ValueError(f"找不到 Pollinations 模型：{model}")
        if model_info.paid_only:
            raise ValueError("/set-model 只允許選擇免費模型。")
        config = UserModelConfig(provider=Provider.POLLINATIONS, model=model, api_key=bot.settings.pollinations_api_key)
        bot.db.set_model_config(interaction.user.id, config)
        await log_interaction(
            interaction,
            "更新 Pollinations 模型設定。",
            fields=[
                ("模型", model, True),
                ("Context", str(model_info.context_length or "?"), True),
            ],
        )
        await interaction.response.send_message(
            f"模型已設定為 Pollinations/{model}。", ephemeral=True
        )

    @bot.tree.command(name="set-pterodactyl", description="設定你的 Pterodactyl Client API。")
    @app_commands.describe(
        base_url="Pterodactyl 面板網址，或完整的 /api/client 網址。",
        api_key="Pterodactyl Client API 金鑰。",
    )
    async def set_pterodactyl(interaction: discord.Interaction, base_url: str, api_key: str) -> None:
        assert bot.http_session is not None
        await interaction.response.defer(ephemeral=True, thinking=True)
        config, account = await fetch_pterodactyl_account(bot.http_session, bot.settings, base_url, api_key)
        bot.db.set_pterodactyl_config(
            interaction.user.id,
            UserPterodactylConfig(base_url=config.base_url, api_key=config.api_key),
        )

        attributes = account.get("attributes", {}) if isinstance(account, dict) else {}
        username = str(attributes.get("username") or "").strip()
        email = str(attributes.get("email") or "").strip()
        account_label = username or email or "已通過驗證"
        if username and email:
            account_label = f"{username} ({email})"

        await log_interaction(
            interaction,
            "更新 Pterodactyl API 設定。",
            fields=[
                ("API URL", config.base_url, False),
                ("帳號", account_label, False),
            ],
        )
        await interaction.followup.send(
            f"Pterodactyl Client API 已設定完成。驗證帳號：{account_label}\nAPI URL：{config.base_url}",
            ephemeral=True,
        )

    @set_model.autocomplete("model")
    async def set_model_autocomplete(
        interaction: discord.Interaction,
        current: str,
    ) -> list[app_commands.Choice[str]]:
        assert bot.http_session is not None
        models = await fetch_pollinations_models(bot.http_session, bot.settings)
        current_lower = current.lower().strip()
        filtered = [
            model_info
            for model_info in models
            if not model_info.paid_only
            and (
                not current_lower
                or current_lower in model_info.name.lower()
                or current_lower in model_info.description.lower()
                or any(current_lower in alias.lower() for alias in model_info.aliases)
            )
        ]
        return [
            app_commands.Choice(
                name=format_model_choice_name(
                    model_info.name,
                    model_info.context_length,
                    model_info.description,
                ),
                value=model_info.name,
            )
            for model_info in filtered[:25]
        ]

    @bot.tree.command(name="custom-model", description="設定非 Pollinations 的模型供應商。")
    @app_commands.describe(provider="供應商類型：openai / anthropic / google / xai / custom", api_key="供應商 API 金鑰。", model="供應商模型名稱。")
    async def custom_model(interaction: discord.Interaction, provider: str, api_key: str, model: str) -> None:
        try:
            provider_value = Provider(provider.lower())
        except ValueError as exc:
            raise ValueError(f"不支援的供應商：{provider}") from exc
        config = UserModelConfig(provider=provider_value, api_key=api_key.strip(), model=model.strip())
        bot.db.set_model_config(interaction.user.id, config)
        await log_interaction(
            interaction,
            "更新自訂模型設定。",
            fields=[
                ("供應商", config.provider.value, True),
                ("模型", config.model, True),
            ],
        )
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
        await log_interaction(
            interaction,
            "列出工作區路徑。",
            fields=[
                ("路徑", path, True),
                ("項目數", str(len(entries)), True),
            ],
        )
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
        await log_interaction(
            interaction,
            "讀取工作區檔案。",
            fields=[
                ("路徑", path, True),
                ("大小", str(len(content.encode('utf-8'))), True),
            ],
        )
        if len(content) <= 1800:
            await interaction.response.send_message(f"```text\n{content}\n```", ephemeral=True)
            return
        file = discord.File(io.BytesIO(content.encode("utf-8")), filename=Path(path).name or "file.txt")
        await interaction.response.send_message("檔案已附加。", file=file, ephemeral=True)

    @file_manager.command(name="write", description="寫入 UTF-8 文字檔。")
    async def file_write(interaction: discord.Interaction, path: str, content: str) -> None:
        size = bot.workspace.write_file(interaction.user.id, path, content)
        total_size = bot.workspace.total_size(interaction.user.id)
        await log_interaction(
            interaction,
            "寫入工作區檔案。",
            fields=[
                ("路徑", path, True),
                ("位元組", str(size), True),
            ],
        )
        await interaction.response.send_message(
            f"已寫入 {size} 位元組到 {path}。工作區用量：{total_size}/{bot.settings.workspace_limit_bytes}。",
            ephemeral=True,
        )

    @file_manager.command(name="delete", description="刪除檔案。")
    async def file_delete(interaction: discord.Interaction, path: str) -> None:
        bot.workspace.delete_file(interaction.user.id, path)
        await log_interaction(
            interaction,
            "刪除工作區檔案。",
            fields=[("路徑", path, True)],
        )
        await interaction.response.send_message(f"已刪除 {path}。", ephemeral=True)

    @file_manager.command(name="mkdir", description="建立資料夾。")
    async def file_mkdir(interaction: discord.Interaction, path: str) -> None:
        created_path = bot.workspace.create_folder(interaction.user.id, path)
        await log_interaction(
            interaction,
            "建立工作區資料夾。",
            fields=[("路徑", created_path, True)],
        )
        await interaction.response.send_message(f"已建立資料夾 {created_path}。", ephemeral=True)

    async def handle_rmdir(interaction: discord.Interaction, path: str, force: bool) -> None:
        removed_path = bot.workspace.remove_folder(interaction.user.id, path, force=force)
        await log_interaction(
            interaction,
            "刪除工作區資料夾。",
            fields=[
                ("路徑", removed_path, True),
                ("Force", str(force), True),
            ],
        )
        suffix = "（已遞迴刪除內容）" if force else ""
        await interaction.response.send_message(f"已刪除資料夾 {removed_path}{suffix}。", ephemeral=True)

    @file_manager.command(name="rmdir", description="刪除資料夾。")
    @app_commands.describe(path="要刪除的工作區資料夾路徑。", force="是否遞迴刪除非空資料夾。")
    async def file_rmdir(interaction: discord.Interaction, path: str, force: bool = False) -> None:
        await handle_rmdir(interaction, path, force)

    @bot.tree.command(name="rmdir", description="刪除工作區資料夾。")
    @app_commands.describe(path="要刪除的工作區資料夾路徑。", force="是否遞迴刪除非空資料夾。")
    async def rmdir_command(interaction: discord.Interaction, path: str, force: bool = False) -> None:
        await handle_rmdir(interaction, path, force)

    bot.tree.add_command(file_manager)

    @bot.tree.command(name="export-zip", description="將工作區匯出為 zip 壓縮檔。")
    async def export_zip(interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        export_path = bot.settings.data_dir / "exports" / f"{interaction.user.id}.zip"
        bot.workspace.export_zip(interaction.user.id, export_path)
        file = discord.File(export_path, filename=f"workspace-{interaction.user.id}.zip")
        await log_interaction(
            interaction,
            "匯出工作區 zip。",
            fields=[("檔案", export_path.name, True)],
        )
        await interaction.followup.send("工作區匯出已準備完成。", file=file, ephemeral=True)

    @bot.tree.command(name="import-zip", description="從 zip 壓縮檔匯入工作區。")
    @app_commands.describe(archive="要匯入到工作區根目錄的 zip 壓縮檔。")
    async def import_zip(interaction: discord.Interaction, archive: discord.Attachment) -> None:
        if not archive.filename.lower().endswith(".zip"):
            raise ValueError("請上傳副檔名為 .zip 的壓縮檔。")
        await interaction.response.defer(ephemeral=True, thinking=True)
        archive_bytes = await archive.read()
        imported_paths = bot.workspace.import_zip(interaction.user.id, archive_bytes)
        total_size = bot.workspace.total_size(interaction.user.id)
        preview_paths = ", ".join(imported_paths[:5])
        if len(imported_paths) > 5:
            preview_paths += f" 等 {len(imported_paths)} 個檔案"
        await log_interaction(
            interaction,
            "匯入工作區 zip。",
            fields=[
                ("檔名", archive.filename, True),
                ("檔案數", str(len(imported_paths)), True),
                ("預覽", preview_text(preview_paths or "(空白)", 900), False),
            ],
        )
        await interaction.followup.send(
            f"已匯入 {len(imported_paths)} 個檔案。工作區用量：{total_size}/{bot.settings.workspace_limit_bytes} 位元組。",
            ephemeral=True,
        )

    @bot.command(name="add_credits")
    async def add_credits(ctx: commands.Context[AgentCordBot], member: discord.User, amount: float) -> None:
        if bot.settings.bot_owner_id is not None and ctx.author.id != bot.settings.bot_owner_id:
            await bot.log_event(
                "!add_credits",
                "非擁有者嘗試調整額度。",
                user=ctx.author,
                guild=ctx.guild,
                color=discord.Colour.orange(),
                fields=[
                    ("目標", str(member.id), True),
                    ("金額", f"{amount:.2f}", True),
                ],
            )
            await ctx.reply("只有設定中的擁有者可以調整額度。")
            return
        balance = bot.db.add_credits(member.id, amount)
        await bot.log_event(
            "!add_credits",
            "已調整使用者額度。",
            user=ctx.author,
            guild=ctx.guild,
            fields=[
                ("目標", f"{member} ({member.id})", False),
                ("金額", f"{amount:.2f}", True),
                ("新餘額", f"{balance:.2f}", True),
            ],
        )
        await ctx.reply(f"已為 {member.mention} 調整 {amount:.2f} 額度。新餘額：{balance:.2f}")

    @ask.error
    @agent_command.error
    @plan_command.error
    @agent_alias.error
    @agent_history.error
    @agent_open.error
    @delete_agent.error
    @set_model.error
    @set_pterodactyl.error
    @custom_model.error
    @file_list.error
    @file_read.error
    @file_write.error
    @file_delete.error
    @file_mkdir.error
    @file_rmdir.error
    @rmdir_command.error
    @export_zip.error
    @import_zip.error
    async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        original = getattr(error, "original", error)
        message = str(original)
        await log_command_error(interaction, original if isinstance(original, BaseException) else Exception(message))
        if isinstance(original, (WorkspaceError, ValueError, aiohttp.ClientError)):
            if isinstance(original, aiohttp.ClientError):
                message = f"網路請求失敗：{message or type(original).__name__}"
            use_ephemeral = not is_public_agent_command(interaction)
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=use_ephemeral)
            else:
                await interaction.response.send_message(message, ephemeral=use_ephemeral)
            return
        raise error

    @add_credits.error
    async def on_add_credits_error(ctx: commands.Context[AgentCordBot], error: commands.CommandError) -> None:
        await bot.log_exception(
            "!add_credits 失敗",
            error,
            user=ctx.author,
            guild=ctx.guild,
            details="prefix 指令執行失敗。",
        )
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
