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

# Sphinx text roles (and `py:` domain variants) we strip from chunk text before
# tokenization / embedding. The marker syntax pollutes BM25 token sets and
# muddies dense embeddings without contributing to retrieval. See D19 in
# ARCHITECTURE.md for the design choice (regex over a fixed role list, not a
# full docutils parse).
_RST_ROLES = (
    "ref",
    "class",
    "mod",
    "func",
    "meth",
    "attr",
    "exc",
    "obj",
    "data",
    "const",
    "doc",
    "term",
    "envvar",
    "option",
    "file",
    "samp",
    "guilabel",
    "kbd",
    "menuselection",
)
_RST_ROLE_RE = re.compile(r":(?:py:)?(?:" + "|".join(_RST_ROLES) + r"):`([^`]+)`")


def _replace_rst_role(match: re.Match[str]) -> str:
    """Replacement function for `_RST_ROLE_RE.sub`.

    Returns just the displayable text:
    - `:class:`Foo``                 → "Foo"
    - `:class:`~torch.nn.Linear``    → "torch.nn.Linear"
      (we strip only the leading `~` and KEEP the full module path; better
      BM25 recall on dotted symbol queries than the Sphinx-display "Linear")
    - `:ref:`hooks <my-hooks>``      → "hooks"  (display label, drop target)
    - `:py:class:`Foo``              → "Foo"
    """
    content = match.group(1)
    if "<" in content and content.rstrip().endswith(">"):
        display, _, _ = content.partition("<")
        if display.strip():
            return display.strip()
    if content.startswith("~"):
        content = content[1:]
    return content


def _clean_rst_text(source: str) -> str:
    """Strip Sphinx role markup from raw source text.

    Single-pass regex over the full file; no code-block awareness (the few
    meta-doc cases where a `:role:` literal appears inside a code fence are
    rare in practice and read fine post-strip anyway). Other RST markup
    (single backtick literals, `*emphasis*`, `**strong**`, double-backtick
    inline code) is intentionally left alone.
    """
    return _RST_ROLE_RE.sub(_replace_rst_role, source)


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
    source = _clean_rst_text(source)
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
