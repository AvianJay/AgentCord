from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from agentcord.models import Provider, TaskRecord, TaskStatus, UserModelConfig


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
                    related_files TEXT NOT NULL DEFAULT '[]'
                );

                CREATE TABLE IF NOT EXISTS allowed_urls (
                    user_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    PRIMARY KEY (user_id, url)
                );
                """
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

    def create_task(self, user_id: int, title: str, status: TaskStatus, related_files: list[str] | None = None) -> TaskRecord:
        related = related_files or []
        with self._connection:
            cursor = self._connection.execute(
                "INSERT INTO tasks (user_id, title, status, related_files) VALUES (?, ?, ?, ?)",
                (user_id, title, status.value, json.dumps(related)),
            )
        return TaskRecord(id=int(cursor.lastrowid), title=title, status=status, related_files=related)

    def update_task(self, task_id: int, status: TaskStatus, related_files: list[str]) -> TaskRecord:
        with self._connection:
            self._connection.execute(
                "UPDATE tasks SET status = ?, related_files = ? WHERE id = ?",
                (status.value, json.dumps(related_files), task_id),
            )
        row = self._connection.execute("SELECT id, title FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise ValueError(f"找不到任務 {task_id}。")
        return TaskRecord(
            id=int(row["id"]),
            title=row["title"],
            status=status,
            related_files=related_files,
        )

    def list_tasks(self, user_id: int) -> list[TaskRecord]:
        rows = self._connection.execute(
            "SELECT id, title, status, related_files FROM tasks WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [
            TaskRecord(
                id=int(row["id"]),
                title=row["title"],
                status=TaskStatus(row["status"]),
                related_files=json.loads(row["related_files"]),
            )
            for row in rows
        ]

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
