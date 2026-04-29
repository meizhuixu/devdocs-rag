"""AST-based code chunker.

Phase 1: returns one chunk per top-level def/class for `.py`. Other languages
return a single whole-file chunk as a placeholder until tree-sitter lands.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CodeChunk(BaseModel):
    """A retrieval unit derived from a source file."""

    path: str
    symbol: str
    chunk_type: str = Field(default="code")
    start_line: int
    end_line: int
    text: str
    language: str


def _python_chunks(path: Path, source: str) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        logger.warning("python parse failed — emitting whole-file chunk", extra={"path": str(path)})
        return [_whole_file_chunk(path, source, "python")]

    lines = source.splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            start = node.lineno
            end = node.end_lineno or start
            text = "\n".join(lines[start - 1 : end])
            chunks.append(
                CodeChunk(
                    path=str(path),
                    symbol=node.name,
                    start_line=start,
                    end_line=end,
                    text=text,
                    language="python",
                )
            )

    if not chunks:
        chunks.append(_whole_file_chunk(path, source, "python"))
    return chunks


def _whole_file_chunk(path: Path, source: str, language: str) -> CodeChunk:
    return CodeChunk(
        path=str(path),
        symbol=path.stem,
        start_line=1,
        end_line=max(1, source.count("\n") + 1),
        text=source,
        language=language,
    )


def load_code_file(path: Path) -> list[CodeChunk]:
    """Read a code file and split into AST-aware chunks."""
    source = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".py":
        return _python_chunks(path, source)
    return [_whole_file_chunk(path, source, path.suffix.lstrip("."))]
