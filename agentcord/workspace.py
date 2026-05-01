from __future__ import annotations

import difflib
import fnmatch
import io
import json
import py_compile
import re
import shutil
import time
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
    ".agentcord",
    ".agentcord/*",
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

    def grep_search(
        self,
        user_id: int,
        query: str,
        path: str = ".",
        *,
        is_regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> dict[str, Any]:
        absolute = self._resolve_path(user_id, path)
        if not absolute.exists():
            raise WorkspaceError(f"找不到路徑：{path}")

        needle = query.strip()
        if not needle:
            raise WorkspaceError("grep_search 的 query 不可為空。")

        limit = max(1, min(max_results, 200))
        pattern: re.Pattern[str] | None = None
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                pattern = re.compile(query, flags)
            except re.error as exc:
                raise WorkspaceError(f"grep_search 的 regex 無效：{exc}") from exc

        files = [absolute] if absolute.is_file() else [item for item in sorted(absolute.rglob("*")) if item.is_file()]
        root = self.user_root(user_id)
        matches: list[dict[str, Any]] = []
        truncated = False
        comparison_needle = needle if case_sensitive else needle.lower()

        for file_path in files:
            relative_path = file_path.relative_to(root).as_posix()
            if self._is_internal_review_path(relative_path):
                continue
            try:
                content = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for line_number, line in enumerate(content.splitlines(), start=1):
                if pattern is not None:
                    matched = pattern.search(line) is not None
                else:
                    haystack = line if case_sensitive else line.lower()
                    matched = comparison_needle in haystack
                if not matched:
                    continue
                if len(matches) >= limit:
                    truncated = True
                    break
                matches.append(
                    {
                        "path": relative_path,
                        "line": line_number,
                        "text": line,
                    }
                )
            if truncated:
                break

        return {
            "path": self._to_relative(user_id, absolute),
            "query": query,
            "is_regex": is_regex,
            "case_sensitive": case_sensitive,
            "matches": matches,
            "truncated": truncated,
        }

    def file_exists(self, user_id: int, path: str) -> bool:
        absolute = self._resolve_path(user_id, path)
        return absolute.is_file()

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

    def list_patch_target_paths(self, diff_text: str) -> list[str]:
        file_diffs = self._parse_unified_diff(diff_text)
        paths: list[str] = []
        for file_diff in file_diffs:
            old_path = str(file_diff.get("old_path") or "")
            new_path = str(file_diff.get("new_path") or "")
            target_path = new_path if new_path != "/dev/null" else old_path
            if target_path and target_path != "/dev/null":
                paths.append(target_path)
        return paths

    def list_all_files(self, user_id: int, path: str = ".") -> list[str]:
        absolute = self._resolve_path(user_id, path)
        if not absolute.exists():
            raise WorkspaceError(f"找不到路徑：{path}")
        if absolute.is_file():
            return [self._to_relative(user_id, absolute)]

        files: list[str] = []
        for child in sorted(absolute.rglob("*")):
            if child.is_file():
                files.append(self._to_relative(user_id, child))
        return files

    def stage_task_file_changes(self, user_id: int, task_id: int, paths: list[str]) -> list[str]:
        manifest = self._load_task_review_manifest(user_id, task_id)
        entries = manifest["entries"]
        tracked_paths = {str(entry.get("path") or "") for entry in entries if isinstance(entry, dict)}
        updated_contents: dict[str, str | None] = {}
        added_paths: list[str] = []

        for raw_path in paths:
            normalized_path = self._normalize_relative_path(str(raw_path or ""))
            if normalized_path == "." or self._is_internal_review_path(normalized_path) or normalized_path in tracked_paths:
                continue
            absolute = self._resolve_path(user_id, normalized_path)
            if absolute.exists() and absolute.is_dir():
                raise WorkspaceError(f"只能追蹤檔案變更：{normalized_path}")

            existed = absolute.exists()
            snapshot_path = ""
            if existed:
                try:
                    original_text = absolute.read_text(encoding="utf-8")
                except UnicodeDecodeError as exc:
                    raise WorkspaceError(f"只支援追蹤 UTF-8 文字檔變更：{normalized_path}") from exc
                snapshot_path = self._task_review_snapshot_path(task_id, normalized_path)
                updated_contents[snapshot_path] = original_text

            entries.append(
                {
                    "path": normalized_path,
                    "snapshot_path": snapshot_path,
                    "existed": existed,
                    "updated_at": time.time_ns(),
                }
            )
            tracked_paths.add(normalized_path)
            added_paths.append(normalized_path)

        if not added_paths:
            return []

        updated_contents[self._task_review_manifest_path(task_id)] = self._serialize_task_review_manifest(entries)
        self._write_patch_results(user_id, updated_contents)
        return added_paths

    def discard_task_file_changes(self, user_id: int, task_id: int, paths: list[str]) -> None:
        self._remove_task_file_changes(user_id, task_id, paths)

    def list_task_file_changes(self, user_id: int, task_id: int) -> list[dict[str, Any]]:
        entries = self._load_task_review_manifest(user_id, task_id)["entries"]
        results: list[dict[str, Any]] = []
        for entry in sorted(entries, key=lambda item: (str(item.get("path") or ""))):
            if not isinstance(entry, dict):
                continue
            path = self._normalize_relative_path(str(entry.get("path") or ""))
            if path == ".":
                continue
            absolute = self._resolve_path(user_id, path)
            current_exists = absolute.exists() and absolute.is_file()
            existed = bool(entry.get("existed"))
            status = "modified"
            if existed and not current_exists:
                status = "deleted"
            elif not existed and current_exists:
                status = "added"
            results.append(
                {
                    "path": path,
                    "snapshot_path": str(entry.get("snapshot_path") or ""),
                    "existed": existed,
                    "current_exists": current_exists,
                    "status": status,
                    "updated_at": int(entry.get("updated_at") or 0),
                }
            )
        return results

    def get_task_file_change_diff(
        self,
        user_id: int,
        task_id: int,
        path: str,
        *,
        context_lines: int = 3,
    ) -> dict[str, Any]:
        entry = self._get_task_review_entry(user_id, task_id, path)
        normalized_path = self._normalize_relative_path(str(entry.get("path") or path))
        existed = bool(entry.get("existed"))
        before_text = ""
        snapshot_path = str(entry.get("snapshot_path") or "")
        if existed and snapshot_path:
            before_text = self.read_file(user_id, snapshot_path)

        absolute = self._resolve_path(user_id, normalized_path)
        current_exists = absolute.exists() and absolute.is_file()
        after_text = ""
        if current_exists:
            try:
                after_text = absolute.read_text(encoding="utf-8")
            except UnicodeDecodeError as exc:
                raise WorkspaceError(f"只支援顯示 UTF-8 文字檔 diff：{normalized_path}") from exc

        status = "modified"
        if existed and not current_exists:
            status = "deleted"
        elif not existed and current_exists:
            status = "added"

        diff_text = "".join(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"a/{normalized_path}" if existed else "/dev/null",
                tofile=f"b/{normalized_path}" if current_exists else "/dev/null",
                n=max(0, context_lines),
            )
        )
        if not diff_text.strip():
            diff_text = f"(目前沒有可顯示的文字差異：{normalized_path})"
        return {
            "path": normalized_path,
            "status": status,
            "diff": diff_text,
            "current_exists": current_exists,
            "existed": existed,
        }

    def accept_task_file_change(self, user_id: int, task_id: int, path: str) -> None:
        self._remove_task_file_changes(user_id, task_id, [path])

    def accept_all_task_file_changes(self, user_id: int, task_id: int) -> int:
        entries = self.list_task_file_changes(user_id, task_id)
        if not entries:
            return 0
        self._remove_task_file_changes(user_id, task_id, [str(entry.get("path") or "") for entry in entries])
        return len(entries)

    def clear_task_review_storage(self, user_id: int, task_id: int) -> None:
        review_root = self._resolve_path(user_id, self._task_review_root_path(task_id))
        if not review_root.exists():
            return
        shutil.rmtree(review_root, ignore_errors=True)
        agent_root = review_root.parent
        try:
            if agent_root.exists() and not any(agent_root.iterdir()):
                agent_root.rmdir()
        except OSError:
            return

    def revert_task_file_change(self, user_id: int, task_id: int, path: str) -> None:
        manifest = self._load_task_review_manifest(user_id, task_id)
        entries = manifest["entries"]
        normalized_path = self._normalize_relative_path(path)
        entry = None
        remaining_entries: list[dict[str, Any]] = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            item_path = self._normalize_relative_path(str(item.get("path") or ""))
            if item_path == normalized_path and entry is None:
                entry = item
                continue
            remaining_entries.append(item)
        if entry is None:
            raise WorkspaceError(f"找不到待確認變更：{normalized_path}")

        updated_contents: dict[str, str | None] = {}
        existed = bool(entry.get("existed"))
        snapshot_path = str(entry.get("snapshot_path") or "")
        if existed:
            if not snapshot_path:
                raise WorkspaceError(f"缺少原始快照：{normalized_path}")
            updated_contents[normalized_path] = self.read_file(user_id, snapshot_path)
            updated_contents[snapshot_path] = None
        else:
            updated_contents[normalized_path] = None
        self._apply_task_review_manifest_update(task_id, remaining_entries, updated_contents)
        self._write_patch_results(user_id, updated_contents)
        self._prune_task_review_storage(user_id, task_id)

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
        # 蒐集要打包的檔案，先排序確保 zip 內容穩定可重現，並一律使用 posix 風格
        # 路徑作為 arcname；否則在 Windows 上 zipfile 會寫入 `a\b\c.txt` 這種反斜線
        # 路徑，導致解壓端（Linux/macOS）看到單一壞檔名而非巢狀資料夾。
        entries: list[tuple[str, Path]] = []
        for file_path in sorted(root.rglob("*")):
            if not file_path.is_file():
                continue
            try:
                relative = file_path.relative_to(root)
            except ValueError:
                continue
            relative_posix = relative.as_posix()
            if self._is_internal_review_path(relative_posix):
                continue
            entries.append((relative_posix, file_path))

        # 寫入暫存檔再 rename，避免中途失敗留下半成品 zip。
        tmp_target = target.with_suffix(target.suffix + ".tmp")
        if tmp_target.exists():
            tmp_target.unlink()
        try:
            with zipfile.ZipFile(
                tmp_target,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=True,
            ) as archive:
                for arcname, file_path in entries:
                    archive.write(file_path, arcname=arcname)
        except Exception:
            if tmp_target.exists():
                try:
                    tmp_target.unlink()
                except OSError:
                    pass
            raise
        if target.exists():
            target.unlink()
        tmp_target.replace(target)
        return target

    def import_zip(self, user_id: int, archive_bytes: bytes) -> list[str]:
        if not archive_bytes:
            raise WorkspaceError("上傳的檔案是空的。")
        try:
            archive_buffer = io.BytesIO(archive_bytes)
            with zipfile.ZipFile(archive_buffer) as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise WorkspaceError(
                        f"Zip 壓縮檔內含損毀的檔案：{bad_member}（CRC 不符）。"
                    )

                file_members = [member for member in archive.infolist() if not member.is_dir()]
                if not file_members:
                    raise WorkspaceError("Zip 壓縮檔中沒有可匯入的檔案。")

                normalized_members: list[tuple[str, zipfile.ZipInfo]] = []
                seen_paths: set[str] = set()
                total_size = 0
                skipped_junk = 0
                for member in file_members:
                    raw_name = member.filename or ""
                    if self._is_zip_junk_path(raw_name):
                        skipped_junk += 1
                        continue
                    try:
                        normalized_path = self._normalize_relative_path(raw_name)
                    except WorkspaceError as exc:
                        raise WorkspaceError(
                            f"Zip 壓縮檔包含不允許的路徑：{raw_name}"
                        ) from exc
                    if normalized_path == ".":
                        continue
                    if normalized_path in seen_paths:
                        raise WorkspaceError(f"Zip 壓縮檔包含重複路徑：{normalized_path}")
                    seen_paths.add(normalized_path)
                    if member.file_size < 0:
                        raise WorkspaceError(
                            f"Zip 條目大小不合法：{normalized_path}"
                        )
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

                # 先把所有檔案讀進記憶體並完成 UTF-8 驗證，全部都通過才開始寫入磁碟，
                # 避免一檔失敗後留下半套寫入的工作區。
                staged: list[tuple[str, bytes]] = []
                for normalized_path, member in normalized_members:
                    try:
                        extracted = archive.read(member)
                    except (zipfile.BadZipFile, RuntimeError) as exc:
                        raise WorkspaceError(
                            f"無法讀取 zip 條目 {normalized_path}：{exc}"
                        ) from exc
                    if len(extracted) != member.file_size and member.file_size > 0:
                        # zipfile 已驗 CRC，這通常代表 header 與內容不一致。
                        raise WorkspaceError(
                            f"Zip 條目大小不一致：{normalized_path}"
                        )
                    try:
                        text_content = extracted.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise WorkspaceError(
                            f"只允許匯入 UTF-8 文字檔：{normalized_path}"
                        ) from exc
                    self._assert_text(text_content)
                    staged.append((normalized_path, extracted))

                imported_paths: list[str] = []
                for normalized_path, payload in staged:
                    absolute = self._resolve_path(user_id, normalized_path)
                    absolute.parent.mkdir(parents=True, exist_ok=True)
                    absolute.write_bytes(payload)
                    imported_paths.append(normalized_path)
                return imported_paths
        except zipfile.BadZipFile as exc:
            raise WorkspaceError("上傳的檔案不是有效的 zip 壓縮檔。") from exc

    @staticmethod
    def _is_zip_junk_path(raw_name: str) -> bool:
        cleaned = raw_name.strip().replace("\\", "/")
        if not cleaned:
            return True
        # 一律忽略 macOS Finder 與其他打包器產生的 metadata。
        first_part = cleaned.split("/", 1)[0]
        if first_part in {"__MACOSX", ".DS_Store"}:
            return True
        basename = cleaned.rsplit("/", 1)[-1]
        if basename in {".DS_Store", "Thumbs.db", "desktop.ini"}:
            return True
        if basename.startswith("._"):  # AppleDouble metadata
            return True
        return False

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
                working_lines: list[str] = []
                source_path = new_path
            else:
                original_text = self.read_file(user_id, old_path)
                working_lines = original_text.splitlines(keepends=True)
                source_path = old_path

            if new_path == "/dev/null":
                updated_contents[source_path] = None
                changed_paths.append(source_path)
                continue

            search_start = 0
            for hunk in file_diff["hunks"]:
                source_lines = [line[1:] for line in hunk["lines"] if line and line[0] in {" ", "-"}]
                target_lines = [line[1:] for line in hunk["lines"] if line and line[0] in {" ", "+"}]
                preferred_index = max(0, hunk["old_start"] - 1)
                match_index = self._find_hunk_match(working_lines, source_lines, preferred_index, search_start)
                if match_index is None:
                    raise WorkspaceError(f"Patch 上下文與 {source_path} 不符。")
                working_lines[match_index : match_index + len(source_lines)] = target_lines
                search_start = match_index + len(target_lines)
            updated_contents[new_path] = "".join(working_lines)
            changed_paths.append(new_path)

        self._write_patch_results(user_id, updated_contents)
        return changed_paths

    def _find_hunk_match(
        self,
        lines: list[str],
        source_lines: list[str],
        preferred_index: int,
        minimum_index: int,
    ) -> int | None:
        if not source_lines:
            return max(0, min(max(preferred_index, minimum_index), len(lines)))

        max_start = len(lines) - len(source_lines)
        if max_start < 0:
            return None

        preferred_index = min(max(preferred_index, 0), max_start)
        minimum_index = min(max(minimum_index, 0), max_start)

        if preferred_index >= minimum_index and self._lines_match(lines, preferred_index, source_lines):
            return preferred_index

        candidates = [
            index
            for index in range(minimum_index, max_start + 1)
            if self._lines_match(lines, index, source_lines)
        ]
        if not candidates and minimum_index > 0:
            candidates = [
                index
                for index in range(0, minimum_index)
                if self._lines_match(lines, index, source_lines)
            ]
        if not candidates:
            return None
        return min(candidates, key=lambda index: (abs(index - preferred_index), index))

    def _lines_match(self, lines: list[str], start: int, expected: list[str]) -> bool:
        end = start + len(expected)
        return lines[start:end] == expected

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

    def _task_review_root_path(self, task_id: int) -> str:
        return f".agentcord/task-{task_id}"

    def _task_review_manifest_path(self, task_id: int) -> str:
        return f"{self._task_review_root_path(task_id)}/manifest.json"

    def _task_review_snapshot_path(self, task_id: int, path: str) -> str:
        normalized_path = self._normalize_relative_path(path)
        return f"{self._task_review_root_path(task_id)}/before/{normalized_path}"

    def _load_task_review_manifest(self, user_id: int, task_id: int) -> dict[str, Any]:
        manifest_path = self._task_review_manifest_path(task_id)
        absolute = self._resolve_path(user_id, manifest_path)
        if not absolute.exists():
            return {"version": 1, "entries": []}
        try:
            payload = json.loads(absolute.read_text(encoding="utf-8") or "{}")
        except json.JSONDecodeError as exc:
            raise WorkspaceError(f"task-{task_id} 的變更快照 manifest 格式無效。") from exc
        raw_entries = payload.get("entries") if isinstance(payload, dict) else []
        entries = [entry for entry in raw_entries if isinstance(entry, dict)] if isinstance(raw_entries, list) else []
        return {"version": 1, "entries": entries}

    def _serialize_task_review_manifest(self, entries: list[dict[str, Any]]) -> str:
        normalized_entries: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            path = self._normalize_relative_path(str(entry.get("path") or ""))
            if path == ".":
                continue
            snapshot_path = str(entry.get("snapshot_path") or "").strip()
            normalized_entries.append(
                {
                    "path": path,
                    "snapshot_path": snapshot_path,
                    "existed": bool(entry.get("existed")),
                    "updated_at": int(entry.get("updated_at") or 0),
                }
            )
        return json.dumps({"version": 1, "entries": normalized_entries}, ensure_ascii=False, indent=2)

    def _apply_task_review_manifest_update(
        self,
        task_id: int,
        entries: list[dict[str, Any]],
        updated_contents: dict[str, str | None],
    ) -> None:
        manifest_path = self._task_review_manifest_path(task_id)
        if entries:
            updated_contents[manifest_path] = self._serialize_task_review_manifest(entries)
            return
        updated_contents[manifest_path] = None

    def _get_task_review_entry(self, user_id: int, task_id: int, path: str) -> dict[str, Any]:
        normalized_path = self._normalize_relative_path(path)
        for entry in self._load_task_review_manifest(user_id, task_id)["entries"]:
            if not isinstance(entry, dict):
                continue
            if self._normalize_relative_path(str(entry.get("path") or "")) == normalized_path:
                return entry
        raise WorkspaceError(f"找不到待確認變更：{normalized_path}")

    def _remove_task_file_changes(self, user_id: int, task_id: int, paths: list[str]) -> None:
        normalized_paths = {
            self._normalize_relative_path(str(path or ""))
            for path in paths
            if str(path or "").strip()
        }
        if not normalized_paths:
            return

        manifest = self._load_task_review_manifest(user_id, task_id)
        remaining_entries: list[dict[str, Any]] = []
        removed_entries: list[dict[str, Any]] = []
        for entry in manifest["entries"]:
            if not isinstance(entry, dict):
                continue
            entry_path = self._normalize_relative_path(str(entry.get("path") or ""))
            if entry_path in normalized_paths:
                removed_entries.append(entry)
                continue
            remaining_entries.append(entry)

        if not removed_entries:
            return

        updated_contents: dict[str, str | None] = {}
        for entry in removed_entries:
            snapshot_path = str(entry.get("snapshot_path") or "").strip()
            if snapshot_path:
                updated_contents[snapshot_path] = None
        self._apply_task_review_manifest_update(task_id, remaining_entries, updated_contents)
        self._write_patch_results(user_id, updated_contents)
        self._prune_task_review_storage(user_id, task_id)

    def _prune_task_review_storage(self, user_id: int, task_id: int) -> None:
        review_root = self._resolve_path(user_id, self._task_review_root_path(task_id))
        if not review_root.exists():
            return
        for current in sorted(review_root.rglob("*"), reverse=True):
            if current.is_dir():
                try:
                    current.rmdir()
                except OSError:
                    continue
        try:
            review_root.rmdir()
        except OSError:
            return
        agent_root = review_root.parent
        try:
            if agent_root.exists() and not any(agent_root.iterdir()):
                agent_root.rmdir()
        except OSError:
            return

    def _is_internal_review_path(self, path: str) -> bool:
        normalized = self._normalize_relative_path(path).lower()
        return normalized == ".agentcord" or normalized.startswith(".agentcord/")

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
