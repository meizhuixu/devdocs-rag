"""Per-namespace ingestion state: diff between source tree and Qdrant.

Source-of-truth for "what's already indexed" is Qdrant itself — no SQLite,
no JSON sidecar. The pipeline calls `IngestionState.from_dirs(...)` to build
the diff, then iterates `to_add | to_update` and calls `to_delete` for the
sweep.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

_DOC_SUFFIXES = {".rst", ".md", ".txt"}
_CODE_SUFFIXES = {".py"}


def _git_commit_sha(repo_root: Path, rel_path: str) -> str | None:
    """Last commit that touched `rel_path`. None if untracked / no history."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", rel_path],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    sha = out.stdout.strip()
    return sha or None


def _scan_source(
    source_path: Path,
    repo_root: Path,
    suffixes: Iterable[str],
) -> dict[str, str]:
    """Walk `source_path` for files with the given suffixes, return {rel_path: commit_sha}.

    `rel_path` is relative to `repo_root` so it's stable across machines.
    Files without git history are skipped with a warning.
    """
    result: dict[str, str] = {}
    suffix_set = set(suffixes)
    for p in source_path.rglob("*"):
        if not p.is_file() or p.suffix not in suffix_set:
            continue
        try:
            rel = p.resolve().relative_to(repo_root.resolve())
        except ValueError:
            logger.warning("file %s not under repo_root %s — skipping", p, repo_root)
            continue
        rel_str = str(rel)
        sha = _git_commit_sha(repo_root, rel_str)
        if sha is None:
            logger.warning("no git history for %s — skipping", rel_str)
            continue
        result[rel_str] = sha
    return result


class IngestionState:
    """Diff between current source tree state and what Qdrant already holds."""

    def __init__(
        self,
        current: dict[str, str],
        existing: dict[str, str],
    ) -> None:
        self._current = current
        self._existing = existing

    @classmethod
    def from_dirs(
        cls,
        source_path: Path,
        repo_root: Path,
        existing: dict[str, str],
        suffixes: Iterable[str] = _DOC_SUFFIXES,
    ) -> IngestionState:
        current = _scan_source(source_path, repo_root, suffixes)
        return cls(current=current, existing=existing)

    @property
    def current(self) -> dict[str, str]:
        return self._current

    @property
    def to_add(self) -> set[str]:
        return set(self._current) - set(self._existing)

    @property
    def to_update(self) -> set[str]:
        return {
            p
            for p in self._current
            if p in self._existing and self._current[p] != self._existing[p]
        }

    @property
    def to_delete(self) -> set[str]:
        return set(self._existing) - set(self._current)

    @property
    def to_skip(self) -> set[str]:
        return {
            p
            for p in self._current
            if p in self._existing and self._current[p] == self._existing[p]
        }

    def commit_sha(self, file_path: str) -> str:
        return self._current[file_path]

    def restrict_to(self, file_paths: set[str]) -> IngestionState:
        """Return a new state limited to the given file paths (smoke run)."""
        current = {p: sha for p, sha in self._current.items() if p in file_paths}
        existing = {p: sha for p, sha in self._existing.items() if p in file_paths}
        return IngestionState(current=current, existing=existing)


__all__ = ["_CODE_SUFFIXES", "_DOC_SUFFIXES", "IngestionState"]
