from __future__ import annotations

import fnmatch
import io
import json
import py_compile
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


@dataclass(slots=True)
class WorkspaceEntry:
    path: str
    kind: str
    size: int


class WorkspaceError(ValueError):
    """Raised for invalid workspace operations."""


_DEFAULT_SYNC_IGNORE_PATTERNS = (
    ".git",
    ".hg",
    ".svn",
    ".npm",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    ".next",
    ".nuxt",
    ".turbo",
    ".parcel-cache",
    ".cache",
    "dist",
    "build",
    "*.pyc",
    "*.pyo",
)


class WorkspaceManager:
    def __init__(self, base_dir: Path, limit_bytes: int) -> None:
        self._base_dir = base_dir
        self._limit_bytes = limit_bytes
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def user_root(self, user_id: int) -> Path:
        root = (self._base_dir / str(user_id)).resolve()
        root.mkdir(parents=True, exist_ok=True)
        return root

    def total_size(self, user_id: int) -> int:
        root = self.user_root(user_id)
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    def read_file(self, user_id: int, path: str) -> str:
        absolute = self._resolve_path(user_id, path)
        if not absolute.is_file():
            raise WorkspaceError(f"找不到檔案：{path}")
        return absolute.read_text(encoding="utf-8")

    def write_file(self, user_id: int, path: str, content: str) -> int:
        self._assert_text(content)
        absolute = self._resolve_path(user_id, path)
        absolute.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode("utf-8")
        existing_size = absolute.stat().st_size if absolute.exists() else 0
        new_total = self.total_size(user_id) - existing_size + len(encoded)
        if new_total > self._limit_bytes:
            raise WorkspaceError(
                f"寫入遭拒：工作區將超過 {self._limit_bytes} 位元組上限。"
            )
        absolute.write_bytes(encoded)
        return len(encoded)

    def list_files(self, user_id: int, path: str = ".") -> list[WorkspaceEntry]:
        absolute = self._resolve_path(user_id, path)
        if not absolute.exists():
            raise WorkspaceError(f"找不到路徑：{path}")
        if absolute.is_file():
            return [WorkspaceEntry(path=self._to_relative(user_id, absolute), kind="file", size=absolute.stat().st_size)]

        root = self.user_root(user_id)
        entries: list[WorkspaceEntry] = []
        for child in sorted(absolute.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
            rel = child.relative_to(root).as_posix()
            kind = "folder" if child.is_dir() else "file"
            size = 0 if child.is_dir() else child.stat().st_size
            entries.append(WorkspaceEntry(path=rel, kind=kind, size=size))
        return entries

    def default_sync_ignore_patterns(self) -> list[str]:
        return list(_DEFAULT_SYNC_IGNORE_PATTERNS)

    def collect_sync_candidates(
        self,
        user_id: int,
        path: str = ".",
        *,
        ignore_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        absolute = self._resolve_path(user_id, path)
        if not absolute.exists():
            raise WorkspaceError(f"找不到路徑：{path}")
        current_total = self.total_size(user_id)
        if current_total > self._limit_bytes:
            raise WorkspaceError(
                f"同步遭拒：工作區目前大小 {current_total} 位元組，已超過 {self._limit_bytes} 位元組上限。"
            )

        source_root = absolute if absolute.is_dir() else absolute.parent
        active_patterns = self._build_ignore_pattern_set(ignore_patterns)
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []

        def visit(current: Path) -> None:
            workspace_path = self._to_relative(user_id, current)
            sync_path = current.relative_to(source_root).as_posix() if current != source_root else "."
            is_dir = current.is_dir()
            if self._should_ignore_sync_path(workspace_path, sync_path, is_dir=is_dir, ignore_patterns=active_patterns):
                skipped.append({"path": workspace_path, "reason": "ignored"})
                return
            if is_dir:
                for child in sorted(current.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
                    visit(child)
                return
            files.append(
                {
                    "workspace_path": workspace_path,
                    "relative_path": sync_path,
                    "size": current.stat().st_size,
                }
            )

        if absolute.is_dir():
            for child in sorted(absolute.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
                visit(child)
        else:
            visit(absolute)

        return {
            "source_path": self._to_relative(user_id, absolute),
            "total_size": current_total,
            "limit_bytes": self._limit_bytes,
            "files": files,
            "skipped": skipped,
            "ignore_patterns": sorted(active_patterns),
        }

    def collect_remote_sync_targets(
        self,
        user_id: int,
        path: str = ".",
        *,
        remote_files: list[dict[str, Any]],
        ignore_patterns: list[str] | None = None,
    ) -> dict[str, Any]:
        absolute = self._resolve_path(user_id, path)
        if absolute.exists() and absolute.is_file():
            raise WorkspaceError("同步目的地路徑必須是資料夾。")

        current_total = self.total_size(user_id)
        if current_total > self._limit_bytes:
            raise WorkspaceError(
                f"同步遭拒：工作區目前大小 {current_total} 位元組，已超過 {self._limit_bytes} 位元組上限。"
            )

        root = self.user_root(user_id)
        target_root_relative = "." if absolute == root else absolute.relative_to(root).as_posix()
        active_patterns = self._build_ignore_pattern_set(ignore_patterns)
        files: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        existing_sizes: dict[str, int] = {}
        incoming_total = 0

        for item in remote_files:
            relative_path = self._normalize_relative_path(str(item.get("relative_path") or ""))
            if relative_path == ".":
                continue
            workspace_path = relative_path if target_root_relative == "." else f"{target_root_relative}/{relative_path}"
            if self._should_ignore_sync_path(
                workspace_path,
                relative_path,
                is_dir=False,
                ignore_patterns=active_patterns,
            ):
                skipped.append({"path": workspace_path, "reason": "ignored"})
                continue
            remote_path = str(item.get("remote_path") or "").strip()
            size_value = item.get("size", 0)
            size = size_value if isinstance(size_value, int) and size_value >= 0 else 0
            files.append(
                {
                    "workspace_path": workspace_path,
                    "relative_path": relative_path,
                    "remote_path": remote_path,
                    "size": size,
                    "mimetype": str(item.get("mimetype") or "").strip(),
                }
            )
            absolute_target = self._resolve_path(user_id, workspace_path)
            existing_sizes[workspace_path] = absolute_target.stat().st_size if absolute_target.exists() else 0
            incoming_total += size

        projected_total = current_total - sum(existing_sizes.values()) + incoming_total
        if projected_total > self._limit_bytes:
            raise WorkspaceError(
                f"同步遭拒：拉取後工作區將超過 {self._limit_bytes} 位元組上限。"
            )

        return {
            "target_path": target_root_relative,
            "total_size": current_total,
            "projected_total": projected_total,
            "limit_bytes": self._limit_bytes,
            "files": files,
            "skipped": skipped,
            "ignore_patterns": sorted(active_patterns),
        }

    def delete_file(self, user_id: int, path: str) -> None:
        absolute = self._resolve_path(user_id, path)
        if absolute.is_dir():
            raise WorkspaceError("delete_file 只支援刪除檔案。")
        if not absolute.exists():
            raise WorkspaceError(f"找不到檔案：{path}")
        absolute.unlink()

    def create_folder(self, user_id: int, path: str) -> str:
        absolute = self._resolve_path(user_id, path)
        absolute.mkdir(parents=True, exist_ok=True)
        return self._to_relative(user_id, absolute)

    def remove_folder(self, user_id: int, path: str, *, force: bool = False) -> str:
        absolute = self._resolve_path(user_id, path)
        root = self.user_root(user_id)
        if absolute == root:
            raise WorkspaceError("不允許刪除工作區根目錄。")
        if not absolute.exists():
            raise WorkspaceError(f"找不到資料夾：{path}")
        if absolute.is_file():
            raise WorkspaceError("rmdir 只支援刪除資料夾。")
        if force:
            shutil.rmtree(absolute)
            return self._to_relative(user_id, absolute)
        try:
            absolute.rmdir()
        except OSError as exc:
            raise WorkspaceError("資料夾不是空的；若要遞迴刪除請設 force=true。") from exc
        return self._to_relative(user_id, absolute)

    def export_zip(self, user_id: int, target: Path) -> Path:
        root = self.user_root(user_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    archive.write(file_path, arcname=file_path.relative_to(root))
        return target

    def import_zip(self, user_id: int, archive_bytes: bytes) -> list[str]:
        try:
            archive_buffer = io.BytesIO(archive_bytes)
            with zipfile.ZipFile(archive_buffer) as archive:
                file_members = [member for member in archive.infolist() if not member.is_dir()]
                if not file_members:
                    raise WorkspaceError("Zip 壓縮檔中沒有可匯入的檔案。")

                normalized_members: list[tuple[str, zipfile.ZipInfo]] = []
                seen_paths: set[str] = set()
                total_size = 0
                for member in file_members:
                    normalized_path = self._normalize_relative_path(member.filename)
                    if normalized_path == ".":
                        continue
                    if normalized_path in seen_paths:
                        raise WorkspaceError(f"Zip 壓縮檔包含重複路徑：{normalized_path}")
                    seen_paths.add(normalized_path)
                    total_size += member.file_size
                    normalized_members.append((normalized_path, member))

                if not normalized_members:
                    raise WorkspaceError("Zip 壓縮檔中沒有可匯入的檔案。")

                existing_total = self.total_size(user_id)
                existing_sizes: dict[str, int] = {}
                for normalized_path, _ in normalized_members:
                    absolute = self._resolve_path(user_id, normalized_path)
                    existing_sizes[normalized_path] = absolute.stat().st_size if absolute.exists() else 0
                projected_total = existing_total - sum(existing_sizes.values()) + total_size
                if projected_total > self._limit_bytes:
                    raise WorkspaceError(
                        f"匯入遭拒：工作區將超過 {self._limit_bytes} 位元組上限。"
                    )

                imported_paths: list[str] = []
                for normalized_path, member in normalized_members:
                    absolute = self._resolve_path(user_id, normalized_path)
                    absolute.parent.mkdir(parents=True, exist_ok=True)
                    extracted = archive.read(member)
                    try:
                        text_content = extracted.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise WorkspaceError(f"只允許匯入 UTF-8 文字檔：{normalized_path}") from exc
                    self._assert_text(text_content)
                    absolute.write_bytes(extracted)
                    imported_paths.append(normalized_path)
                return imported_paths
        except zipfile.BadZipFile as exc:
            raise WorkspaceError("上傳的檔案不是有效的 zip 壓縮檔。") from exc

    def py_compile_check(self, user_id: int, path: str) -> str:
        absolute = self._resolve_path(user_id, path)
        if absolute.suffix != ".py":
            return f"已略過 {path} 的語法驗證：僅支援 Python 檔案。"
        if not absolute.exists():
            raise WorkspaceError(f"找不到檔案：{path}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            pyc_path = Path(tmp_dir) / "check.pyc"
            py_compile.compile(str(absolute), cfile=str(pyc_path), doraise=True)
        return f"語法檢查通過：{path}"

    def apply_patch(self, user_id: int, diff_text: str) -> list[str]:
        file_diffs = self._parse_unified_diff(diff_text)
        changed_paths: list[str] = []
        updated_contents: dict[str, str | None] = {}

        for file_diff in file_diffs:
            old_path = file_diff["old_path"]
            new_path = file_diff["new_path"]
            if old_path == "/dev/null":
                original_lines: list[str] = []
                source_path = new_path
            else:
                original_text = self.read_file(user_id, old_path)
                original_lines = original_text.splitlines(keepends=True)
                source_path = old_path

            if new_path == "/dev/null":
                updated_contents[source_path] = None
                changed_paths.append(source_path)
                continue

            result_lines: list[str] = []
            cursor = 0
            for hunk in file_diff["hunks"]:
                old_start = hunk["old_start"]
                result_lines.extend(original_lines[cursor : old_start - 1])
                cursor = old_start - 1
                for line in hunk["lines"]:
                    if not line:
                        continue
                    prefix = line[0]
                    payload = line[1:]
                    if prefix == " ":
                        if cursor >= len(original_lines) or original_lines[cursor] != payload:
                            raise WorkspaceError(f"Patch 上下文與 {source_path} 不符。")
                        result_lines.append(payload)
                        cursor += 1
                    elif prefix == "-":
                        if cursor >= len(original_lines) or original_lines[cursor] != payload:
                            raise WorkspaceError(f"Patch 刪除內容與 {source_path} 不符。")
                        cursor += 1
                    elif prefix == "+":
                        result_lines.append(payload)
                if hunk["old_count"] == 0 and old_start == 0:
                    cursor = 0
            result_lines.extend(original_lines[cursor:])
            updated_contents[new_path] = "".join(result_lines)
            changed_paths.append(new_path)

        self._write_patch_results(user_id, updated_contents)
        return changed_paths

    def _write_patch_results(self, user_id: int, updated_contents: dict[str, str | None]) -> None:
        current_total = self.total_size(user_id)
        for relative_path, new_content in updated_contents.items():
            absolute = self._resolve_path(user_id, relative_path)
            current_size = absolute.stat().st_size if absolute.exists() else 0
            current_total -= current_size
            if new_content is None:
                continue
            self._assert_text(new_content)
            current_total += len(new_content.encode("utf-8"))
        if current_total > self._limit_bytes:
            raise WorkspaceError(
                f"Patch 遭拒：工作區將超過 {self._limit_bytes} 位元組上限。"
            )

        for relative_path, new_content in updated_contents.items():
            absolute = self._resolve_path(user_id, relative_path)
            if new_content is None:
                if absolute.exists():
                    absolute.unlink()
                continue
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_text(new_content, encoding="utf-8")

    def _resolve_path(self, user_id: int, path: str) -> Path:
        relative = self._normalize_relative_path(path)
        root = self.user_root(user_id)
        if relative == ".":
            return root
        absolute = (root / relative).resolve()
        if root not in absolute.parents and absolute != root:
            raise WorkspaceError("不允許路徑跳脫。")
        return absolute

    def _normalize_relative_path(self, path: str) -> str:
        cleaned = path.strip().replace("\\", "/")
        if not cleaned or cleaned == "/":
            return "."
        pure = PurePosixPath(cleaned)
        if pure.is_absolute() or ".." in pure.parts:
            raise WorkspaceError("不允許路徑跳脫。")
        parts = [part for part in pure.parts if part not in ("", ".")]
        if not parts:
            return "."
        return PurePosixPath(*parts).as_posix()

    def _to_relative(self, user_id: int, absolute: Path) -> str:
        root = self.user_root(user_id)
        return absolute.relative_to(root).as_posix()

    def _assert_text(self, content: str) -> None:
        if "\x00" in content:
            raise WorkspaceError("只允許 UTF-8 文字檔。")
        content.encode("utf-8")

    def _parse_unified_diff(self, diff_text: str) -> list[dict]:
        lines = diff_text.splitlines(keepends=True)
        if not lines:
            raise WorkspaceError("Patch 內容不可為空。")

        files: list[dict] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("diff --git "):
                index += 1
                continue
            if line.startswith("--- "):
                old_path = self._strip_diff_path(line[4:].strip())
                index += 1
                if index >= len(lines) or not lines[index].startswith("+++ "):
                    raise WorkspaceError("Patch 格式無效：缺少新路徑。")
                new_path = self._strip_diff_path(lines[index][4:].strip())
                index += 1
                hunks: list[dict] = []
                while index < len(lines) and not lines[index].startswith("--- "):
                    if lines[index].startswith("@@"):
                        header = lines[index]
                        index += 1
                        old_start, old_count, new_start, new_count = self._parse_hunk_header(header)
                        hunk_lines: list[str] = []
                        while index < len(lines) and not lines[index].startswith("@@") and not lines[index].startswith("--- "):
                            if lines[index].startswith("\\ No newline at end of file"):
                                index += 1
                                continue
                            hunk_lines.append(lines[index])
                            index += 1
                        hunks.append(
                            {
                                "old_start": old_start,
                                "old_count": old_count,
                                "new_start": new_start,
                                "new_count": new_count,
                                "lines": hunk_lines,
                            }
                        )
                    else:
                        index += 1
                files.append({"old_path": old_path, "new_path": new_path, "hunks": hunks})
                continue
            index += 1
        if not files:
            raise WorkspaceError("Patch 格式無效：找不到檔案區段。")
        return files

    def _parse_hunk_header(self, header: str) -> tuple[int, int, int, int]:
        metadata = header.split("@@")[1].strip()
        old_chunk, new_chunk = metadata.split(" ")
        old_start, old_count = self._parse_hunk_range(old_chunk)
        new_start, new_count = self._parse_hunk_range(new_chunk)
        return old_start, old_count, new_start, new_count

    def _parse_hunk_range(self, value: str) -> tuple[int, int]:
        signless = value[1:]
        if "," in signless:
            start_str, count_str = signless.split(",", 1)
            return int(start_str), int(count_str)
        return int(signless), 1

    def _strip_diff_path(self, path: str) -> str:
        if path in {"/dev/null", "dev/null"}:
            return "/dev/null"
        path = path.removeprefix("a/").removeprefix("b/")
        return self._normalize_relative_path(path)

    def dump_tree(self, user_id: int) -> str:
        root = self.user_root(user_id)
        items: list[dict[str, Any]] = []
        ignore_patterns = self._build_ignore_pattern_set(None)
        max_depth = 5
        max_entries = 500
        truncated = False

        def visit(directory: Path, depth: int) -> None:
            nonlocal truncated
            for child in sorted(directory.iterdir(), key=lambda item: (item.is_file(), item.name.lower())):
                if len(items) >= max_entries:
                    truncated = True
                    return
                relative_path = child.relative_to(root).as_posix()
                is_dir = child.is_dir()
                if self._should_ignore_sync_path(relative_path, relative_path, is_dir=is_dir, ignore_patterns=ignore_patterns):
                    items.append(
                        {
                            "path": relative_path,
                            "kind": "folder" if is_dir else "file",
                            "ignored": True,
                        }
                    )
                    continue
                if is_dir:
                    entry: dict[str, Any] = {"path": relative_path, "kind": "folder"}
                    if depth >= max_depth:
                        entry["truncated"] = True
                        items.append(entry)
                        continue
                    items.append(entry)
                    visit(child, depth + 1)
                    continue
                items.append(
                    {
                        "path": relative_path,
                        "kind": "file",
                        "size": child.stat().st_size,
                    }
                )

        visit(root, 0)
        if truncated:
            items.append(
                {
                    "path": "...",
                    "kind": "meta",
                    "truncated": True,
                    "reason": f"僅顯示前 {max_entries} 個項目。",
                }
            )
        return json.dumps(items, ensure_ascii=False, indent=2)

    def _build_ignore_pattern_set(self, extra_patterns: list[str] | None) -> set[str]:
        patterns = {
            pattern.strip().replace("\\", "/").lower()
            for pattern in _DEFAULT_SYNC_IGNORE_PATTERNS
            if pattern.strip()
        }
        for pattern in extra_patterns or []:
            normalized = str(pattern).strip().replace("\\", "/").lower()
            if normalized:
                patterns.add(normalized)
        return patterns

    def _should_ignore_sync_path(
        self,
        workspace_path: str,
        sync_path: str,
        *,
        is_dir: bool,
        ignore_patterns: set[str],
    ) -> bool:
        candidates = {
            workspace_path.strip("/").lower(),
            sync_path.strip("/").lower(),
        }
        candidates = {candidate for candidate in candidates if candidate and candidate != "."}
        names = {PurePosixPath(candidate).name for candidate in candidates}
        for pattern in ignore_patterns:
            if any(fnmatch.fnmatch(candidate, pattern) for candidate in candidates):
                return True
            if any(fnmatch.fnmatch(name, pattern) for name in names):
                return True
            if is_dir and any(fnmatch.fnmatch(f"{candidate}/", pattern.rstrip("/") + "/") for candidate in candidates):
                return True
        return False
