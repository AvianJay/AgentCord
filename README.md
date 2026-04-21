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
- External providers for OpenAI, Anthropic, Google, xAI, and OpenAI-compatible custom endpoints
- Task tracking for agent runs
- Markdown memory notes stored in the workspace
- Guarded web search/fetch flow using Pollinations `gemini-search`
- Zip export for generated projects

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
- `POLLINATIONS_API_KEY` (optional, used for Pollinations and search)
- `AGENTCORD_CUSTOM_PROVIDER_BASE_URL` (required when provider is `custom`)
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
