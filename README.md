# AgentCord

AI-powered coding agent bot for Discord, built with `discord.py`.

## Features

- `/ask` for lightweight AI chat
- `/agent` and `/call-ai-codeing` for multi-step coding workflows
- Per-user isolated workspaces with:
  - hierarchical folders
  - UTF-8 text files only
  - 5MB hard storage limit
  - path traversal protection
- Credits system with one-time default balance and owner overrides
- Pollinations as the default AI provider with per-user model switching
- Per-user Pterodactyl Client API credentials for agent-side server operations
- External providers for OpenAI, Anthropic, Google, xAI, Poe, and OpenAI-compatible custom endpoints
- Task tracking for agent runs
- Markdown memory notes stored in the workspace
- Web search via Pollinations `gemini-search` and direct webpage fetching through optional `PROXY_*` settings
- Zip export and import for workspace projects
- Pterodactyl-aware agent tools for startup config, power control, console reading, server file editing, and workspace-to-server sync

## Requirements

- Python 3.12+
- A Discord bot token

## Setup

```bash
python -m pip install -e .
```

Environment variables:

- `DISCORD_TOKEN` (required)
- `DISCORD_APPLICATION_ID` (recommended)
- `BOT_OWNER_ID` (required for `!add_credits`)
- `DISCORD_LOG_WEBHOOK` (optional, sends batched webhook embeds for command and agent activity logs)
- `POLLINATIONS_API_KEY` (optional, used for Pollinations and search)
- `PROXY_URL` (optional for `fetch_url` / Pterodactyl, but required when provider is `custom`; supports `http://`, `https://`, `socks4://`, `socks4a://`, `socks5://`, `socks5h://`)
- `PROXY_USERNAME` / `PROXY_PASSWORD` (optional, proxy credentials for supported proxy types)
- `PROXY_HOST` / `PROXY_PORT` / `PROXY_SCHEME` (optional fallback to build `PROXY_URL`)
- `PROXY_HEADERS_JSON` (optional JSON object for proxy request headers)
- `AGENTCORD_DATA_DIR` (defaults to `./data`)
- `AGENTCORD_DEFAULT_MODEL` (defaults to `openai`)
- `AGENTCORD_MODEL_RATES_JSON` (optional JSON override for credit rates)

## Run

```bash
python main.py
```

## Notes

- Users cannot execute code; the bot only generates and edits files.
- Python validation uses syntax-only compilation via `py_compile`.
- Existing file edits are designed to prefer unified diff patches over full overwrites during agent runs.
- `/custom-model` 使用 `provider=custom` 時，`api_key` 參數格式為 `{apiurl}:{apikey}`；自訂 API 會一律透過 proxy，以 OpenAI-compatible `/models` 與 `/chat/completions` 介面呼叫。
- `/import-zip` only accepts valid zip archives containing UTF-8 text files and rejects path traversal entries.
- When `DISCORD_LOG_WEBHOOK` is set, command usage, agent session actions, and execution errors are queued and sent to the webhook in embed batches.
- `/set-pterodactyl` validates the provided panel URL and Client API key against `GET /api/client/account` before saving them for the current user.
- SOCKS proxies require the installed `aiohttp-socks` dependency; the packaged dependencies include it by default.
- Agent workspace tree and Pterodactyl sync automatically ignore bulky generated directories such as `.venv`, `venv`, `node_modules`, and `__pycache__`, while still showing those directories at the tree level so the model can decide whether to add more ignore rules.
