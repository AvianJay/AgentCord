# AgentCord
Your code agent in Discord.

## Features
- `discord.py` slash-command based bot
- Virtual workspace per user (`5MB` quota)
- Pollinations free text model support (`/set-model`)
- Custom provider support (`/custom-model`) for `openai`, `xai`, `claude`, `google`
- Points system with per-model rates; owner text commands can update points/rates
- Web search + read flow with strict URL allowlist (`/web-search` then `/read-web`)

## Commands
Slash commands (with zh-TW localization):
- `/ask` (`問`)
- `/agent` (`代理`)
- `/call-ai-codeing` (`叫ai寫程式`) - alias of `/agent`
- `/file-manager` (`檔案總管`) - list/read/write/delete in virtual workspace
- `/set-model` (`設定模型`) - select Pollinations model
- `/custom-model` (`自訂模型`) - set `provider`, `api_key`, `model`
- `/export-zip` (`匯出zip`) - download workspace zip
- `/web-search` + `/read-web` - URL read restricted to URLs returned by search
- `/py-compile` - compile a Python file in workspace
- `/points` - view your points/model

Owner text commands:
- `!setpoints <user_id> <points>`
- `!setrate <model> <rate>`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Environment variables:
- `DISCORD_TOKEN` (required)
- `BOT_OWNER_ID` (required)
- `POLLINATIONS_API_KEY` (optional)
- `AGENTCORD_DATA` (optional, default `./.agentcord-data`)

Run:
```bash
python -m agentcord
```
