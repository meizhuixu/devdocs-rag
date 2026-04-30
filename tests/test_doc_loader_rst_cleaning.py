"""Unit tests for RST role stripping in doc_loader.

Covers `_clean_rst_text` directly + an integration check that
`load_doc_file` applies it before chunking. No Qdrant / no embedder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from devdocs_rag.ingestion.loaders.doc_loader import (
    _clean_rst_text,
    load_doc_file,
)

# ---------- direct regex behavior ----------


@pytest.mark.parametrize(
    ("text_in", "expected"),
    [
        # plain role
        (":class:`Foo`", "Foo"),
        # py: domain prefix
        (":py:class:`Foo`", "Foo"),
        # ~ prefix: leading tilde stripped, full path KEPT (better BM25 recall)
        (":class:`~torch.nn.Linear`", "torch.nn.Linear"),
        (":class:`~Foo`", "Foo"),
        # display <target>: keep display label
        (":ref:`hooks <my-hooks>`", "hooks"),
        (":doc:`Tutorial <path/to/tutorial>`", "Tutorial"),
        # multiple roles in one line
        ("see :class:`Foo` and :func:`bar`", "see Foo and bar"),
        # role at start of line
        (":class:`Foo` is a class.", "Foo is a class."),
        # all role types we cover
        (":mod:`os`", "os"),
        (":func:`open`", "open"),
        (":meth:`Foo.bar`", "Foo.bar"),
        (":attr:`x.y`", "x.y"),
        (":exc:`ValueError`", "ValueError"),
        (":obj:`None`", "None"),
        (":data:`PI`", "PI"),
        (":const:`MAX`", "MAX"),
        (":term:`tensor`", "tensor"),
        (":envvar:`HOME`", "HOME"),
        (":file:`README.md`", "README.md"),
        # unknown role: left alone
        (":newrole:`X`", ":newrole:`X`"),
        # unclosed (no trailing backtick): left alone
        (":class:`Foo (no close", ":class:`Foo (no close"),
        # empty content: regex requires at least one non-backtick char,
        # so `:class:``` won't match — left alone.
        (":class:``", ":class:``"),
        # plain text untouched
        ("the quick brown fox", "the quick brown fox"),
        ("", ""),
        # markdown emphasis / strong / inline code: NOT stripped (deliberate)
        ("*emphasis* and **strong** and `literal`", "*emphasis* and **strong** and `literal`"),
        # double-backtick inline RST: NOT touched (only `:role:` syntax)
        ("``code literal``", "``code literal``"),
    ],
    ids=lambda v: repr(v)[:48] if isinstance(v, str) else "",
)
def test_clean_rst_text_cases(text_in: str, expected: str) -> None:
    assert _clean_rst_text(text_in) == expected


def test_clean_rst_real_pytorch_snippet() -> None:
    """A representative paragraph from docs/source/notes/extending.rst."""
    src = (
        "If your network uses :ref:`custom autograd functions"
        "<extending-autograd>` (subclasses of :class:`torch.autograd.Function`),"
        " changes are required for autocast compatibility."
    )
    out = _clean_rst_text(src)
    assert ":ref:" not in out
    assert ":class:" not in out
    # No tilde was used → full path preserved for retrieval.
    assert "torch.autograd.Function" in out
    # Display label "custom autograd functions" preserved (target dropped).
    assert "custom autograd functions" in out
    assert "<extending-autograd>" not in out


def test_clean_rst_preserves_plain_text_around_roles() -> None:
    src = "Use the :class:`~torch.nn.Linear` layer for fully-connected nets."
    out = _clean_rst_text(src)
    assert out == "Use the torch.nn.Linear layer for fully-connected nets."


def test_clean_rst_idempotent() -> None:
    """Cleaning twice gives the same result as cleaning once."""
    src = "see :class:`A` and :func:`b` and :ref:`label <target>`"
    once = _clean_rst_text(src)
    twice = _clean_rst_text(once)
    assert once == twice


# ---------- integration with load_doc_file ----------


def test_load_doc_file_applies_rst_cleaning(tmp_path: Path) -> None:
    """End-to-end: `:class:`Foo`` in source becomes `Foo` in chunk text."""
    f = tmp_path / "sample.rst"
    f.write_text(
        "Title\n=====\n\nUse :class:`~torch.nn.Linear` and see :ref:`docs <some-target>`.\n"
    )
    chunks = load_doc_file(f)
    assert len(chunks) == 1
    text = chunks[0].text
    assert ":class:" not in text
    assert ":ref:" not in text
    assert "torch.nn.Linear" in text
    assert "docs" in text
    assert "some-target" not in text


def test_load_doc_file_md_with_rst_roles(tmp_path: Path) -> None:
    """PyTorch's myst-parser .md files contain RST roles too — clean them."""
    f = tmp_path / "sample.md"
    f.write_text("# Heading\n\nThe :class:`MyClass` is built on :func:`make_x`.\n")
    chunks = load_doc_file(f)
    assert len(chunks) == 1
    text = chunks[0].text
    assert ":class:" not in text
    assert ":func:" not in text
    assert "MyClass" in text
    assert "make_x" in text
