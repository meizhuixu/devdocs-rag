"""Tests for ingestion.state — glob matcher + scan filtering."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devdocs_rag.ingestion.state import (
    _DOC_SUFFIXES,
    _scan_source,
    matches_any_glob,
)

# ---------- matches_any_glob ----------


@pytest.mark.parametrize(
    ("path", "patterns", "expected"),
    [
        # Trailing /** — recursive prefix
        ("docs/source/scripts/build.py", ["docs/source/scripts/**"], True),
        ("docs/source/scripts/sub/x.py", ["docs/source/scripts/**"], True),
        ("docs/source/notes/extending.rst", ["docs/source/scripts/**"], False),
        # ** at start — match anywhere
        ("a/b/node_modules/x.js", ["**/node_modules/**"], True),
        ("a/b/c/x.js", ["**/node_modules/**"], False),
        # Single segment glob
        ("foo.lock", ["*.lock"], True),
        ("foo.lock.bak", ["*.lock"], False),
        # Multiple patterns: any-match
        ("dist/app.js", ["build/**", "dist/**"], True),
        ("src/app.js", ["build/**", "dist/**"], False),
        # ** matches zero segments (so prefix/** matches the prefix dir itself
        # would require trailing slash semantics — we don't support that, and
        # the prefix dir isn't a file anyway, so it's moot for our use)
        # Empty pattern list
        ("anything", [], False),
        # Exact path
        ("README.md", ["README.md"], True),
        ("README.md", ["readme.md"], False),  # case-sensitive
        # ? wildcard within a segment
        ("a/b1.txt", ["a/b?.txt"], True),
        ("a/b12.txt", ["a/b?.txt"], False),
    ],
)
def test_matches_any_glob(path: str, patterns: list[str], expected: bool) -> None:
    assert matches_any_glob(path, patterns) is expected


# ---------- _scan_source filters via ignore_globs ----------


def _git_init(repo_root: Path) -> None:
    """Set up a tiny git repo so _git_commit_sha works for the scan."""
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "config", "commit.gpgsign", "false"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        cwd=repo_root,
        check=True,
    )


def test_scan_source_applies_ignore_globs(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "docs" / "source" / "notes").mkdir(parents=True)
    (repo / "docs" / "source" / "scripts").mkdir(parents=True)
    (repo / "docs" / "source" / "_static").mkdir(parents=True)

    keep = repo / "docs" / "source" / "notes" / "extending.rst"
    keep.write_text("Extending\n=========\nbody\n")

    drop_script = repo / "docs" / "source" / "scripts" / "build.py"
    drop_script.write_text("def build(): pass\n")

    drop_static = repo / "docs" / "source" / "_static" / "logo.txt"
    drop_static.write_text("logo\n")

    _git_init(repo)

    suffixes = _DOC_SUFFIXES | {".py"}
    ignore = ["docs/source/scripts/**", "docs/source/_static/**"]

    out = _scan_source(
        source_path=repo / "docs" / "source",
        repo_root=repo,
        suffixes=suffixes,
        ignore_globs=ignore,
    )

    assert "docs/source/notes/extending.rst" in out
    assert "docs/source/scripts/build.py" not in out
    assert "docs/source/_static/logo.txt" not in out
    assert len(out) == 1


def test_scan_source_no_ignore_returns_all(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "scripts").mkdir()
    (repo / "scripts" / "x.py").write_text("x = 1\n")
    (repo / "notes.rst").write_text("Note\n====\n")

    _git_init(repo)

    out = _scan_source(
        source_path=repo,
        repo_root=repo,
        suffixes=_DOC_SUFFIXES | {".py"},
    )
    assert "scripts/x.py" in out
    assert "notes.rst" in out
