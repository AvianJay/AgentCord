from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse


MAX_WORKSPACE_BYTES = 5 * 1024 * 1024


class WorkspaceError(Exception):
    pass


@dataclass
class UserWorkspace:
    user_id: int
    root: Path

    def resolve_path(self, relative_path: str) -> Path:
        rel = Path(relative_path)
        if rel.is_absolute():
            raise WorkspaceError("Only relative paths are allowed.")
        root = self.root.resolve()
        candidate = (root / rel).resolve()
        if not candidate.is_relative_to(root):
            raise WorkspaceError("Path traversal is not allowed.")
        return candidate


class WorkspaceManager:
    def __init__(self, base_dir: Path, max_bytes: int = MAX_WORKSPACE_BYTES) -> None:
        self.base_dir = base_dir
        self.max_bytes = max_bytes
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def workspace(self, user_id: int) -> UserWorkspace:
        root = self.base_dir / str(user_id)
        root.mkdir(parents=True, exist_ok=True)
        return UserWorkspace(user_id=user_id, root=root)

    def usage_bytes(self, user_id: int) -> int:
        ws = self.workspace(user_id)
        total = 0
        for p in ws.root.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total

    def list_files(self, user_id: int) -> List[str]:
        ws = self.workspace(user_id)
        return sorted(
            str(p.relative_to(ws.root))
            for p in ws.root.rglob("*")
            if p.is_file()
        )

    def read_text(self, user_id: int, relative_path: str) -> str:
        ws = self.workspace(user_id)
        p = ws.resolve_path(relative_path)
        if not p.exists() or not p.is_file():
            raise WorkspaceError("File does not exist.")
        return p.read_text(encoding="utf-8")

    def delete_file(self, user_id: int, relative_path: str) -> None:
        ws = self.workspace(user_id)
        p = ws.resolve_path(relative_path)
        if p.exists() and p.is_file():
            p.unlink()

    def write_text(self, user_id: int, relative_path: str, content: str) -> int:
        ws = self.workspace(user_id)
        p = ws.resolve_path(relative_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        current_usage = self.usage_bytes(user_id)
        old_size = p.stat().st_size if p.exists() else 0
        new_size = len(content.encode("utf-8"))
        projected = current_usage - old_size + new_size
        if projected > self.max_bytes:
            raise WorkspaceError(
                f"Storage limit exceeded: {projected}/{self.max_bytes} bytes."
            )

        p.write_text(content, encoding="utf-8")
        return projected

    def export_zip(self, user_id: int) -> bytes:
        ws = self.workspace(user_id)
        mem = io.BytesIO()
        with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in ws.root.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=str(p.relative_to(ws.root)))
        return mem.getvalue()


class PointsManager:
    DEFAULT_RATES = {
        "openai-large": 8,
        "openai": 6,
        "mistral": 4,
        "qwen": 3,
        "gemini": 5,
        "deepseek": 4,
        "llama": 2,
    }

    def __init__(self, state_path: Path, default_points: int = 1000) -> None:
        self.state_path = state_path
        self.default_points = default_points
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state: Dict[str, Dict[str, object]] = self._load_state()

    def _load_state(self) -> Dict[str, Dict[str, object]]:
        if not self.state_path.exists():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _ensure_user(self, user_id: int) -> Dict[str, object]:
        key = str(user_id)
        if key not in self._state:
            self._state[key] = {
                "points": self.default_points,
                "pollinations_model": "openai",
                "custom_provider": None,
                "custom_api_key": None,
                "custom_model": None,
            }
            self._save()
        return self._state[key]

    def get_points(self, user_id: int) -> int:
        return int(self._ensure_user(user_id)["points"])

    def set_points(self, user_id: int, points: int) -> None:
        user = self._ensure_user(user_id)
        user["points"] = max(0, points)
        self._save()

    def charge(self, user_id: int, model: str, prompt: str, output: str) -> int:
        rate = self.DEFAULT_RATES.get(model, 6)
        token_like = max(1, (len(prompt) + len(output)) // 4)
        cost = max(1, (token_like * rate) // 100)
        user = self._ensure_user(user_id)
        points = int(user["points"])
        if points < cost:
            raise WorkspaceError(f"Insufficient points: required {cost}, left {points}.")
        user["points"] = points - cost
        self._save()
        return cost

    def get_pollinations_model(self, user_id: int) -> str:
        return str(self._ensure_user(user_id)["pollinations_model"])

    def set_pollinations_model(self, user_id: int, model: str) -> None:
        user = self._ensure_user(user_id)
        user["pollinations_model"] = model
        self._save()

    def set_custom_model(
        self, user_id: int, provider: str, api_key: str, model: str
    ) -> None:
        user = self._ensure_user(user_id)
        user["custom_provider"] = provider
        user["custom_api_key"] = api_key
        user["custom_model"] = model
        self._save()

    def get_custom_config(self, user_id: int) -> Dict[str, str | None]:
        user = self._ensure_user(user_id)
        return {
            "provider": user.get("custom_provider"),
            "api_key": user.get("custom_api_key"),
            "model": user.get("custom_model"),
        }


class SearchAllowlist:
    def __init__(self) -> None:
        self.allowed: Dict[int, set[str]] = {}

    def register_urls(self, user_id: int, urls: List[str]) -> None:
        clean = {self._normalize(u) for u in urls if self._is_http_url(u)}
        self.allowed[user_id] = clean

    def is_allowed(self, user_id: int, url: str) -> bool:
        return self._normalize(url) in self.allowed.get(user_id, set())

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        urls = re.findall(r"https?://[^\s)>\]\"]+", text)
        return [u.rstrip(".,);!?") for u in urls]

    @staticmethod
    def _normalize(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    @staticmethod
    def _is_http_url(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def html_to_markdown(html: str) -> str:
    text = re.sub(r"<script\b[\s\S]*?</script[^>]*>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style\b[\s\S]*?</style[^>]*>", "", text, flags=re.IGNORECASE)
    replacements = {
        r"</h[1-6]>": "\n\n",
        r"<h1[^>]*>": "# ",
        r"<h2[^>]*>": "## ",
        r"<h3[^>]*>": "### ",
        r"</p>": "\n\n",
        r"<p[^>]*>": "",
        r"<br\s*/?>": "\n",
        r"<li[^>]*>": "- ",
        r"</li>": "\n",
    }
    for pattern, value in replacements.items():
        text = re.sub(pattern, value, text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
