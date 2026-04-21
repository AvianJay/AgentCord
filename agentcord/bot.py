from __future__ import annotations

import asyncio
import io
import json
import os
import py_compile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands

from .core import PointsManager, SearchAllowlist, WorkspaceError, WorkspaceManager, html_to_markdown

POLLINATIONS_MODELS = [
    "openai",
    "openai-large",
    "mistral",
    "qwen",
    "gemini",
    "deepseek",
    "llama",
    "gemini-search",
]

COMMAND_LOCALE_ZH = {
    "ask": "問",
    "agent": "代理",
    "call-ai-codeing": "叫ai寫程式",
    "file-manager": "檔案總管",
    "set-model": "設定模型",
    "custom-model": "自訂模型",
    "export-zip": "匯出zip",
}


@dataclass
class BotConfig:
    token: str
    owner_id: int
    pollinations_api_key: Optional[str]


class AIClient:
    def __init__(self, pollinations_api_key: Optional[str]) -> None:
        self.pollinations_api_key = pollinations_api_key

    async def call_pollinations(self, model: str, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.pollinations_api_key:
            headers["Authorization"] = f"Bearer {self.pollinations_api_key}"

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are AgentCord: a safe coding assistant for Discord. "
                        "Keep output concise and practical."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://text.pollinations.ai/openai", json=payload, headers=headers, timeout=90
            ) as resp:
                body = await resp.text()
                if resp.status >= 400:
                    raise WorkspaceError(f"Pollinations request failed ({resp.status}): {body[:500]}")
                data = json.loads(body)
                return data["choices"][0]["message"]["content"]

    async def call_custom_provider(
        self, provider: str, api_key: str, model: str, prompt: str
    ) -> str:
        p = provider.lower().strip()
        async with aiohttp.ClientSession() as session:
            if p in {"openai", "xai"}:
                base = "https://api.openai.com" if p == "openai" else "https://api.x.ai"
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                }
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                }
                async with session.post(
                    f"{base}/v1/chat/completions", json=payload, headers=headers, timeout=90
                ) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        raise WorkspaceError(f"{provider} request failed: {str(data)[:500]}")
                    return data["choices"][0]["message"]["content"]

            if p == "claude":
                headers = {
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": model,
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": prompt}],
                }
                async with session.post(
                    "https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=90
                ) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        raise WorkspaceError(f"Claude request failed: {str(data)[:500]}")
                    return data["content"][0]["text"]

            if p == "google":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                async with session.post(url, json=payload, timeout=90) as resp:
                    data = await resp.json()
                    if resp.status >= 400:
                        raise WorkspaceError(f"Google request failed: {str(data)[:500]}")
                    return data["candidates"][0]["content"]["parts"][0]["text"]

        raise WorkspaceError("Unsupported provider. Use openai/xai/claude/google.")


class AgentCordBot(commands.Bot):
    def __init__(self, cfg: BotConfig, data_root: Path) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix="!", intents=intents)

        self.cfg = cfg
        self.workspace = WorkspaceManager(data_root / "workspaces")
        self.points = PointsManager(data_root / "state" / "users.json")
        self.ai = AIClient(cfg.pollinations_api_key)
        self.allowlist = SearchAllowlist()

    async def setup_hook(self) -> None:
        await self.register_commands()
        await self.tree.sync()

    async def register_commands(self) -> None:
        async def send_long(interaction: discord.Interaction, title: str, text: str) -> None:
            chunks = [text[i : i + 1800] for i in range(0, len(text), 1800)] or ["(empty)"]
            await interaction.followup.send(f"**{title}**\n{chunks[0]}")
            for chunk in chunks[1:]:
                await interaction.followup.send(chunk)

        async def run_ai(interaction: discord.Interaction, prompt: str) -> None:
            user_id = interaction.user.id
            poll_model = self.points.get_pollinations_model(user_id)
            custom = self.points.get_custom_config(user_id)

            if custom.get("provider") and custom.get("api_key") and custom.get("model"):
                result = await self.ai.call_custom_provider(
                    provider=str(custom["provider"]),
                    api_key=str(custom["api_key"]),
                    model=str(custom["model"]),
                    prompt=prompt,
                )
                cost_model = poll_model
            else:
                result = await self.ai.call_pollinations(poll_model, prompt)
                cost_model = poll_model

            cost = self.points.charge(user_id, cost_model, prompt, result)
            left = self.points.get_points(user_id)
            await send_long(interaction, f"AI ({cost_model}) | -{cost} points | left {left}", result)

        @self.tree.command(name="ask", description="Ask AI quickly")
        @app_commands.describe(prompt="Your question")
        async def ask(interaction: discord.Interaction, prompt: str) -> None:
            await interaction.response.defer(thinking=True)
            await run_ai(interaction, prompt)

        ask.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["ask"]}

        @self.tree.command(name="agent", description="Coding agent with workspace context")
        @app_commands.describe(prompt="Task for the coding agent")
        async def agent(interaction: discord.Interaction, prompt: str) -> None:
            await interaction.response.defer(thinking=True)
            files = self.workspace.list_files(interaction.user.id)
            context = "\n".join(files[:100]) if files else "(no files)"
            full_prompt = (
                "You are an online code agent similar to VSCode Copilot. "
                "You can suggest edits based on file list.\n"
                f"Workspace files:\n{context}\n\nUser task:\n{prompt}"
            )
            await run_ai(interaction, full_prompt)

        agent.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["agent"]}

        @self.tree.command(name="call-ai-codeing", description="Alias of /agent")
        @app_commands.describe(prompt="Task for the coding agent")
        async def call_ai_codeing(interaction: discord.Interaction, prompt: str) -> None:
            await agent.callback(interaction, prompt)  # type: ignore[attr-defined]

        call_ai_codeing.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["call-ai-codeing"]}

        @self.tree.command(name="file-manager", description="Manage your virtual workspace files")
        @app_commands.describe(action="list/read/write/delete", path="relative file path", content="text content for write")
        async def file_manager(
            interaction: discord.Interaction,
            action: app_commands.Choice[str],
            path: Optional[str] = None,
            content: Optional[str] = None,
        ) -> None:
            await interaction.response.defer(thinking=True, ephemeral=True)
            uid = interaction.user.id
            try:
                if action.value == "list":
                    files = self.workspace.list_files(uid)
                    usage = self.workspace.usage_bytes(uid)
                    out = "\n".join(files) if files else "(no files)"
                    await send_long(interaction, f"Files ({usage}/{self.workspace.max_bytes} bytes)", out)
                    return
                if not path:
                    raise WorkspaceError("path is required for this action")
                if action.value == "read":
                    txt = self.workspace.read_text(uid, path)
                    await send_long(interaction, path, txt)
                elif action.value == "write":
                    if content is None:
                        raise WorkspaceError("content is required for write")
                    usage = self.workspace.write_text(uid, path, content)
                    await interaction.followup.send(f"Saved `{path}` ({usage}/{self.workspace.max_bytes} bytes)")
                elif action.value == "delete":
                    self.workspace.delete_file(uid, path)
                    await interaction.followup.send(f"Deleted `{path}`")
            except WorkspaceError as e:
                await interaction.followup.send(f"Error: {e}")

        file_manager.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["file-manager"]}
        file_manager.parameters[0].choices = [
            app_commands.Choice(name="list", value="list"),
            app_commands.Choice(name="read", value="read"),
            app_commands.Choice(name="write", value="write"),
            app_commands.Choice(name="delete", value="delete"),
        ]

        @self.tree.command(name="set-model", description="Set Pollinations model")
        @app_commands.describe(model="Model from Pollinations free models")
        async def set_model(
            interaction: discord.Interaction,
            model: app_commands.Choice[str],
        ) -> None:
            self.points.set_pollinations_model(interaction.user.id, model.value)
            await interaction.response.send_message(f"Pollinations model set to `{model.value}`", ephemeral=True)

        set_model.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["set-model"]}
        set_model.parameters[0].choices = [app_commands.Choice(name=m, value=m) for m in POLLINATIONS_MODELS]

        @self.tree.command(name="custom-model", description="Set custom provider/api key/model")
        async def custom_model(
            interaction: discord.Interaction,
            provider: str,
            api_key: str,
            model: str,
        ) -> None:
            self.points.set_custom_model(interaction.user.id, provider=provider, api_key=api_key, model=model)
            await interaction.response.send_message(
                f"Custom provider set: `{provider}` / `{model}`", ephemeral=True
            )

        custom_model.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["custom-model"]}

        @self.tree.command(name="export-zip", description="Export your workspace as zip")
        async def export_zip(interaction: discord.Interaction) -> None:
            data = self.workspace.export_zip(interaction.user.id)
            file_obj = discord.File(io.BytesIO(data), filename=f"workspace-{interaction.user.id}.zip")
            await interaction.response.send_message("Workspace exported:", file=file_obj, ephemeral=True)

        export_zip.name_localizations = {"zh-TW": COMMAND_LOCALE_ZH["export-zip"]}

        @self.tree.command(name="py-compile", description="Compile a Python file from your workspace")
        async def py_compile_cmd(interaction: discord.Interaction, path: str) -> None:
            await interaction.response.defer(thinking=True, ephemeral=True)
            try:
                p = self.workspace.workspace(interaction.user.id).resolve_path(path)
                if not p.exists() or not p.is_file():
                    raise WorkspaceError("file not found")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, lambda: py_compile.compile(str(p), doraise=True))
                await interaction.followup.send(f"`{path}` compiled successfully")
            except Exception as e:  # pragma: no cover - returns exact compile error text
                await interaction.followup.send(f"Compile failed: {e}")

        @self.tree.command(name="web-search", description="Search web via Pollinations gemini-search")
        async def web_search(interaction: discord.Interaction, query: str) -> None:
            await interaction.response.defer(thinking=True)
            prompt = (
                "Use gemini-search behavior. Search the web and return a short markdown summary "
                "with source URLs (full URLs). Query: "
                + query
            )
            result = await self.ai.call_pollinations("gemini-search", prompt)
            urls = SearchAllowlist.extract_urls(result)
            self.allowlist.register_urls(interaction.user.id, urls)
            await send_long(
                interaction,
                "Search Results (URLs now allowlisted for /read-web)",
                result,
            )

        @self.tree.command(name="read-web", description="Read web URL as markdown from last gemini-search results")
        async def read_web(interaction: discord.Interaction, url: str) -> None:
            await interaction.response.defer(thinking=True)
            if not self.allowlist.is_allowed(interaction.user.id, url):
                await interaction.followup.send(
                    "URL is not allowed. Use /web-search first, and only read URLs returned there."
                )
                return
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=60) as resp:
                    html = await resp.text()
                    if resp.status >= 400:
                        raise WorkspaceError(f"Fetch failed: {resp.status}")
            md = html_to_markdown(html)
            await send_long(interaction, f"Read {url}", md[:7000])

        @self.tree.command(name="points", description="Show your points and selected model")
        async def points(interaction: discord.Interaction) -> None:
            p = self.points.get_points(interaction.user.id)
            m = self.points.get_pollinations_model(interaction.user.id)
            await interaction.response.send_message(f"Points: `{p}` | Model: `{m}`", ephemeral=True)

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if message.content.startswith("!setpoints"):
            if message.author.id != self.cfg.owner_id:
                await message.reply("Only owner can use this command.")
                return
            parts = message.content.split()
            if len(parts) != 3 or not parts[1].isdigit() or not parts[2].isdigit():
                await message.reply("Usage: !setpoints <user_id> <points>")
                return
            user_id = int(parts[1])
            points = int(parts[2])
            self.points.set_points(user_id, points)
            await message.reply(f"Set points for {user_id} => {points}")
            return

        await self.process_commands(message)


def load_config() -> BotConfig:
    token = os.getenv("DISCORD_TOKEN", "")
    owner = int(os.getenv("BOT_OWNER_ID", "0"))
    if not token:
        raise RuntimeError("DISCORD_TOKEN is required")
    if owner == 0:
        raise RuntimeError("BOT_OWNER_ID is required")
    return BotConfig(
        token=token,
        owner_id=owner,
        pollinations_api_key=os.getenv("POLLINATIONS_API_KEY"),
    )


def run_bot() -> None:
    cfg = load_config()
    data_root = Path(os.getenv("AGENTCORD_DATA", "./.agentcord-data"))
    bot = AgentCordBot(cfg, data_root=data_root)
    bot.run(cfg.token)
