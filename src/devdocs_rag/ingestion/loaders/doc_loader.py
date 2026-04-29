"""Markdown / prose chunker.

Phase 1: split on top-level headings; each section becomes one chunk.
Phase 2 will swap to a semantic splitter via langchain-text-splitters.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


class DocChunk(BaseModel):
    """A retrieval unit derived from a doc / markdown file."""

    path: str
    symbol: str  # heading text, or filename if no headings
    chunk_type: str = Field(default="doc")
    start_line: int
    end_line: int
    text: str


def load_doc_file(path: Path) -> list[DocChunk]:
    """Read a markdown/text file, split on headings."""
    source = path.read_text(encoding="utf-8", errors="replace")
    lines = source.splitlines()

    headings: list[tuple[int, str]] = []
    for i, line in enumerate(lines, start=1):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, m.group(2).strip()))

    if not headings:
        return [
            DocChunk(
                path=str(path),
                symbol=path.stem,
                start_line=1,
                end_line=max(1, len(lines)),
                text=source,
            )
        ]

    chunks: list[DocChunk] = []
    for idx, (line_no, title) in enumerate(headings):
        end = headings[idx + 1][0] - 1 if idx + 1 < len(headings) else len(lines)
        text = "\n".join(lines[line_no - 1 : end])
        chunks.append(
            DocChunk(
                path=str(path),
                symbol=title,
                start_line=line_no,
                end_line=end,
                text=text,
            )
        )
    return chunks
