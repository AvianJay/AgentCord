from __future__ import annotations

import asyncio
import traceback
from collections import deque
from datetime import datetime, timezone
from typing import Iterable, Sequence

import aiohttp
import discord

from agentcord.ai import format_exception_message, sanitize_sensitive_text

DISCORD_LOG_BATCH_DELAY = 1.0
DISCORD_LOG_BATCH_SIZE = 10
_EMBED_DESCRIPTION_LIMIT = 4000
_EMBED_FIELD_VALUE_LIMIT = 1000


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _chunk_text(text: str, limit: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + limit] for index in range(0, len(text), limit)]


class DiscordWebhookLogger:
    def __init__(self, webhook_url: str, session: aiohttp.ClientSession) -> None:
        self.webhook_url = webhook_url.strip()
        self.session = session
        self._queue: deque[discord.Embed] = deque()
        self._worker_task: asyncio.Task[None] | None = None
        self._flush_lock = asyncio.Lock()
        self._closed = False
        self._webhook: discord.Webhook | None = None

    async def log(
        self,
        title: str,
        description: str,
        *,
        user: discord.abc.User | None = None,
        guild: discord.Guild | None = None,
        color: discord.Colour | None = None,
        fields: Sequence[tuple[str, str, bool]] | None = None,
    ) -> None:
        if self._closed or not self.webhook_url:
            return
        embeds = self._build_embeds(
            title,
            description,
            user=user,
            guild=guild,
            color=color or discord.Colour.blurple(),
            fields=fields,
        )
        self._queue.extend(embeds)
        self._ensure_worker()

    async def log_exception(
        self,
        title: str,
        error: BaseException,
        *,
        user: discord.abc.User | None = None,
        guild: discord.Guild | None = None,
        details: str = "",
        fields: Sequence[tuple[str, str, bool]] | None = None,
    ) -> None:
        trace = sanitize_sensitive_text("".join(traceback.format_exception(type(error), error, error.__traceback__)).strip())
        description_parts = []
        if details.strip():
            description_parts.append(sanitize_sensitive_text(details.strip()))
        description_parts.append(f"{type(error).__name__}: {format_exception_message(error)}")
        description_parts.append(trace)
        await self.log(
            title,
            "\n\n".join(part for part in description_parts if part),
            user=user,
            guild=guild,
            color=discord.Colour.red(),
            fields=fields,
        )

    async def flush(self) -> None:
        if not self.webhook_url:
            return
        async with self._flush_lock:
            while self._queue:
                batch: list[discord.Embed] = []
                while self._queue and len(batch) < DISCORD_LOG_BATCH_SIZE:
                    batch.append(self._queue.popleft())
                try:
                    await self._send_batch(batch)
                except Exception as exc:  # noqa: BLE001
                    print(f"[AgentCord Logger] 發送 webhook log 失敗: {exc}")

    async def close(self) -> None:
        self._closed = True
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        await self.flush()

    def _ensure_worker(self) -> None:
        if self._closed:
            return
        if self._worker_task is not None and not self._worker_task.done():
            return
        loop = asyncio.get_running_loop()
        self._worker_task = loop.create_task(self._worker())

    async def _worker(self) -> None:
        try:
            while self._queue and not self._closed:
                await asyncio.sleep(DISCORD_LOG_BATCH_DELAY)
                await self.flush()
        except asyncio.CancelledError:
            raise
        finally:
            self._worker_task = None
            if self._queue and not self._closed:
                self._ensure_worker()

    async def _send_batch(self, embeds: Iterable[discord.Embed]) -> None:
        webhook = self._get_webhook()
        embed_list = list(embeds)
        if not embed_list:
            return
        await webhook.send(
            embeds=embed_list,
            username="AgentCord Logger",
            allowed_mentions=discord.AllowedMentions.none(),
            wait=False,
        )

    def _get_webhook(self) -> discord.Webhook:
        if self._webhook is None:
            self._webhook = discord.Webhook.from_url(self.webhook_url, session=self.session)
        return self._webhook

    def _build_embeds(
        self,
        title: str,
        description: str,
        *,
        user: discord.abc.User | None,
        guild: discord.Guild | None,
        color: discord.Colour,
        fields: Sequence[tuple[str, str, bool]] | None,
    ) -> list[discord.Embed]:
        description_chunks = _chunk_text(description.strip() or "(空白)", _EMBED_DESCRIPTION_LIMIT)
        embeds: list[discord.Embed] = []
        for index, chunk in enumerate(description_chunks, start=1):
            embed_title = title if index == 1 else f"{title}（續 {index}）"
            embed = discord.Embed(
                title=_truncate(embed_title, 256),
                description=chunk,
                color=color,
                timestamp=datetime.now(timezone.utc),
            )
            if user is not None:
                display_name = getattr(user, "display_name", user.name)
                user_label = f"{display_name} ({user.name})" if display_name != user.name else user.name
                embed.set_author(name=user_label, icon_url=user.display_avatar.url)
            if guild is not None:
                footer_text = guild.name or str(guild.id)
                footer_icon = guild.icon.url if guild.icon else None
            else:
                footer_text = "DM"
                footer_icon = None
            embed.set_footer(text=footer_text, icon_url=footer_icon)
            if index == 1 and fields:
                for field_name, field_value, inline in fields:
                    embed.add_field(
                        name=_truncate(field_name, 256),
                        value=_truncate(field_value or "(空白)", _EMBED_FIELD_VALUE_LIMIT),
                        inline=inline,
                    )
            embeds.append(embed)
        return embeds