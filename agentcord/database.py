from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from agentcord.models import AgentTaskItem, ConversationMessage, Provider, TaskRecord, TaskStatus, UserModelConfig


class Database:
    def __init__(self, db_path: Path, default_credits: float) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_credits = default_credits
        self._connection = sqlite3.connect(self._db_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    credits REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_configs (
                    user_id INTEGER PRIMARY KEY,
                    provider TEXT NOT NULL,
                    api_key TEXT NOT NULL,
                    model TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    related_files TEXT NOT NULL DEFAULT '[]',
                    summary TEXT NOT NULL DEFAULT '',
                    plan TEXT NOT NULL DEFAULT '[]',
                    validations TEXT NOT NULL DEFAULT '[]',
                    messages TEXT NOT NULL DEFAULT '[]',
                    task_items TEXT NOT NULL DEFAULT '[]',
                    model TEXT NOT NULL DEFAULT '',
                    context_length INTEGER,
                    compression_count INTEGER NOT NULL DEFAULT 0,
                    created_at INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS allowed_urls (
                    user_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    PRIMARY KEY (user_id, url)
                );
                """
            )
        self._ensure_task_columns()

    def _ensure_task_columns(self) -> None:
        existing_columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(tasks)").fetchall()
        }
        expected_columns = {
            "summary": "TEXT NOT NULL DEFAULT ''",
            "plan": "TEXT NOT NULL DEFAULT '[]'",
            "validations": "TEXT NOT NULL DEFAULT '[]'",
            "messages": "TEXT NOT NULL DEFAULT '[]'",
            "task_items": "TEXT NOT NULL DEFAULT '[]'",
            "model": "TEXT NOT NULL DEFAULT ''",
            "context_length": "INTEGER",
            "compression_count": "INTEGER NOT NULL DEFAULT 0",
            "created_at": "INTEGER NOT NULL DEFAULT 0",
            "updated_at": "INTEGER NOT NULL DEFAULT 0",
        }
        with self._connection:
            for column_name, definition in expected_columns.items():
                if column_name in existing_columns:
                    continue
                self._connection.execute(
                    f"ALTER TABLE tasks ADD COLUMN {column_name} {definition}"
                )

    def ensure_user(self, user_id: int) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO users (user_id, credits) VALUES (?, ?)",
                (user_id, self._default_credits),
            )

    def get_credits(self, user_id: int) -> float:
        self.ensure_user(user_id)
        row = self._connection.execute("SELECT credits FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return self._default_credits
        return float(row["credits"])

    def add_credits(self, user_id: int, amount: float) -> float:
        self.ensure_user(user_id)
        with self._connection:
            self._connection.execute(
                "UPDATE users SET credits = credits + ? WHERE user_id = ?",
                (amount, user_id),
            )
        return self.get_credits(user_id)

    def consume_credits(self, user_id: int, amount: float) -> float:
        self.ensure_user(user_id)
        balance = self.get_credits(user_id)
        if balance < amount:
            raise ValueError("額度不足。")
        with self._connection:
            self._connection.execute(
                "UPDATE users SET credits = credits - ? WHERE user_id = ?",
                (amount, user_id),
            )
        return self.get_credits(user_id)

    def get_model_config(self, user_id: int, default_model: str) -> UserModelConfig:
        self.ensure_user(user_id)
        row = self._connection.execute(
            "SELECT provider, api_key, model FROM model_configs WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return UserModelConfig(model=default_model)
        return UserModelConfig(
            provider=Provider(row["provider"]),
            api_key=row["api_key"],
            model=row["model"],
        )

    def set_model_config(self, user_id: int, config: UserModelConfig) -> None:
        self.ensure_user(user_id)
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO model_configs (user_id, provider, api_key, model)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    provider = excluded.provider,
                    api_key = excluded.api_key,
                    model = excluded.model
                """,
                (user_id, config.provider.value, config.api_key, config.model),
            )

    def create_task(
        self,
        user_id: int,
        title: str,
        status: TaskStatus,
        related_files: list[str] | None = None,
        *,
        summary: str = "",
        plan: list[str] | None = None,
        validations: list[str] | None = None,
        messages: list[ConversationMessage] | None = None,
        task_items: list[AgentTaskItem] | None = None,
        model: str = "",
        context_length: int | None = None,
        compression_count: int = 0,
    ) -> TaskRecord:
        related = related_files or []
        now = int(time.time())
        with self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO tasks (
                    user_id,
                    title,
                    status,
                    related_files,
                    summary,
                    plan,
                    validations,
                    messages,
                    task_items,
                    model,
                    context_length,
                    compression_count,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    title,
                    status.value,
                    json.dumps(related, ensure_ascii=False),
                    summary,
                    json.dumps(plan or [], ensure_ascii=False),
                    json.dumps(validations or [], ensure_ascii=False),
                    self._serialize_messages(messages or []),
                    self._serialize_task_items(task_items or []),
                    model,
                    context_length,
                    compression_count,
                    now,
                    now,
                ),
            )
        self.prune_task_history(user_id)
        return TaskRecord(
            id=int(cursor.lastrowid),
            user_id=user_id,
            title=title,
            status=status,
            related_files=related,
            summary=summary,
            plan=plan or [],
            validations=validations or [],
            messages=messages or [],
            task_items=task_items or [],
            model=model,
            context_length=context_length,
            compression_count=compression_count,
            created_at=now,
            updated_at=now,
        )

    def update_task(
        self,
        task_id: int,
        status: TaskStatus,
        related_files: list[str],
        *,
        title: str | None = None,
        summary: str | None = None,
        plan: list[str] | None = None,
        validations: list[str] | None = None,
        messages: list[ConversationMessage] | None = None,
        task_items: list[AgentTaskItem] | None = None,
        model: str | None = None,
        context_length: int | None = None,
        compression_count: int | None = None,
    ) -> TaskRecord:
        current = self.get_task_by_id(task_id)
        now = int(time.time())
        with self._connection:
            self._connection.execute(
                """
                UPDATE tasks
                SET title = ?,
                    status = ?,
                    related_files = ?,
                    summary = ?,
                    plan = ?,
                    validations = ?,
                    messages = ?,
                    task_items = ?,
                    model = ?,
                    context_length = ?,
                    compression_count = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    title if title is not None else current.title,
                    status.value,
                    json.dumps(related_files, ensure_ascii=False),
                    summary if summary is not None else current.summary,
                    json.dumps(plan if plan is not None else current.plan, ensure_ascii=False),
                    json.dumps(validations if validations is not None else current.validations, ensure_ascii=False),
                    self._serialize_messages(messages if messages is not None else current.messages),
                    self._serialize_task_items(task_items if task_items is not None else current.task_items),
                    model if model is not None else current.model,
                    context_length if context_length is not None else current.context_length,
                    compression_count if compression_count is not None else current.compression_count,
                    now,
                    task_id,
                ),
            )
        return self.get_task_by_id(task_id)

    def get_task(self, user_id: int, task_id: int) -> TaskRecord:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE user_id = ? AND id = ?",
            (user_id, task_id),
        ).fetchone()
        if row is None:
            raise ValueError(f"找不到任務 {task_id}。")
        return self._row_to_task_record(row)

    def get_task_by_id(self, task_id: int) -> TaskRecord:
        row = self._connection.execute(
            "SELECT * FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"找不到任務 {task_id}。")
        return self._row_to_task_record(row)

    def list_tasks(self, user_id: int, limit: int = 20) -> list[TaskRecord]:
        rows = self._connection.execute(
            "SELECT * FROM tasks WHERE user_id = ? ORDER BY updated_at DESC, id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [self._row_to_task_record(row) for row in rows]

    def prune_task_history(self, user_id: int, keep: int = 20) -> None:
        rows = self._connection.execute(
            "SELECT id FROM tasks WHERE user_id = ? ORDER BY updated_at DESC, id DESC",
            (user_id,),
        ).fetchall()
        stale_ids = [int(row["id"]) for row in rows[keep:]]
        if not stale_ids:
            return
        placeholders = ", ".join("?" for _ in stale_ids)
        with self._connection:
            self._connection.execute(
                f"DELETE FROM tasks WHERE id IN ({placeholders})",
                stale_ids,
            )

    def remember_search_urls(self, user_id: int, urls: list[str]) -> None:
        with self._connection:
            self._connection.executemany(
                "INSERT OR IGNORE INTO allowed_urls (user_id, url) VALUES (?, ?)",
                [(user_id, url) for url in urls],
            )

    def is_allowed_url(self, user_id: int, url: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM allowed_urls WHERE user_id = ? AND url = ?",
            (user_id, url),
        ).fetchone()
        return row is not None

    def _row_to_task_record(self, row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            id=int(row["id"]),
            user_id=int(row["user_id"]),
            title=row["title"],
            status=TaskStatus(row["status"]),
            related_files=json.loads(row["related_files"]),
            summary=row["summary"],
            plan=json.loads(row["plan"]),
            validations=json.loads(row["validations"]),
            messages=self._deserialize_messages(row["messages"]),
            task_items=self._deserialize_task_items(row["task_items"]),
            model=row["model"],
            context_length=row["context_length"],
            compression_count=int(row["compression_count"] or 0),
            created_at=int(row["created_at"] or 0),
            updated_at=int(row["updated_at"] or 0),
        )

    def _serialize_messages(self, messages: list[ConversationMessage]) -> str:
        return json.dumps(
            [{"role": message.role, "content": message.content} for message in messages],
            ensure_ascii=False,
        )

    def _deserialize_messages(self, payload: str) -> list[ConversationMessage]:
        data = json.loads(payload or "[]")
        return [
            ConversationMessage(role=str(item.get("role", "user")), content=str(item.get("content", "")))
            for item in data
            if isinstance(item, dict)
        ]

    def _serialize_task_items(self, items: list[AgentTaskItem]) -> str:
        return json.dumps(
            [{"title": item.title, "status": item.status} for item in items],
            ensure_ascii=False,
        )

    def _deserialize_task_items(self, payload: str) -> list[AgentTaskItem]:
        data = json.loads(payload or "[]")
        return [
            AgentTaskItem(title=str(item.get("title", "")), status=str(item.get("status", "pending")))
            for item in data
            if isinstance(item, dict) and str(item.get("title", "")).strip()
        ]
