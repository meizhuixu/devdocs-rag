"""Markdown / reStructuredText / prose chunker.

Chunks on heading boundaries and tracks the full heading hierarchy
(`heading_path`) so retrieval can show a breadcrumb.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")

# RST adornment characters in conventional priority order. The actual mapping
# from char to level is built per-document — first char seen = h1, second = h2,
# and so on. Spec: https://devguide.python.org/documentation/markup/#sections
_RST_ADORNMENT_CHARS = set('=-~`#"*^+:_')


class DocChunk(BaseModel):
    """A retrieval unit derived from a doc / markdown / rst file."""

    path: str
    symbol: str  # immediate heading text (or filename if no headings)
    chunk_type: str = Field(default="doc")
    heading_path: str = Field(default="")  # "A > B > C"
    level: int = Field(default=1)
    start_line: int
    end_line: int
    text: str


def _md_headings(source: str) -> list[tuple[int, int, str]]:
    """Return [(line_no, level, title)] for markdown ATX headings."""
    out: list[tuple[int, int, str]] = []
    for i, line in enumerate(source.splitlines(), start=1):
        m = _MD_HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            out.append((i, level, m.group(2).strip()))
    return out


def _rst_headings(source: str) -> list[tuple[int, int, str]]:
    """Return [(line_no, level, title)] for RST underline-style headings.

    A heading is a non-empty title line followed by a line of a single
    adornment character repeated at least as many times as the title length
    (allowing 1-char tolerance for trailing whitespace).
    """
    lines = source.splitlines()
    char_to_level: dict[str, int] = {}
    next_level = 1
    out: list[tuple[int, int, str]] = []

    for i in range(len(lines) - 1):
        title = lines[i].strip()
        if not title:
            continue
        underline = lines[i + 1].rstrip()
        if len(underline) < 3:
            continue
        if len(set(underline)) != 1:
            continue
        ch = underline[0]
        if ch not in _RST_ADORNMENT_CHARS:
            continue
        if len(underline) < len(title) - 1:
            continue
        # Skip if the "title" itself looks like an adornment (overline case).
        if title and title[0] in _RST_ADORNMENT_CHARS and len(set(title)) == 1:
            continue
        level = char_to_level.get(ch)
        if level is None:
            level = next_level
            char_to_level[ch] = level
            next_level += 1
        out.append((i + 1, level, title))

    return out


def _extract_headings(path: Path, source: str) -> list[tuple[int, int, str]]:
    if path.suffix == ".md":
        return _md_headings(source)
    if path.suffix == ".rst":
        return _rst_headings(source)
    return []


def _build_heading_path(stack: list[str], level: int, title: str) -> list[str]:
    """Truncate stack to `level - 1` parents, then push `title`."""
    parents = stack[: level - 1]
    # Pad with empty strings if the doc skips levels (e.g. h1 then h3).
    while len(parents) < level - 1:
        parents.append("")
    return [*parents, title]


def load_doc_file(path: Path) -> list[DocChunk]:
    """Read a doc file, split on headings, track heading_path."""
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()
    headings = _extract_headings(path, source)

    if not headings:
        return [
            DocChunk(
                path=str(path),
                symbol=path.stem,
                start_line=1,
                end_line=max(1, len(lines)),
                text=source,
                heading_path=path.stem,
                level=1,
            )
        ]

    chunks: list[DocChunk] = []
    stack: list[str] = []

    # Headings produce one chunk each, spanning from this heading to the next.
    for idx, (line_no, level, title) in enumerate(headings):
        end = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
        # For RST, the underline lives on the line *after* the title — include it.
        text = "\n".join(lines[line_no - 1 : end])
        stack = _build_heading_path(stack, level, title)
        chunks.append(
            DocChunk(
                path=str(path),
                symbol=title,
                heading_path=" > ".join(s for s in stack if s),
                level=level,
                start_line=line_no,
                end_line=end,
                text=text,
            )
        )

    return chunks
