"""Unit tests for the BM25 sparse retriever.

These don't touch Qdrant — they exercise `BM25Index.from_documents` directly.
The `from_qdrant` path is covered by the M1 probe script (live data) and
indirectly by `tests/test_retrieval_sanity.py` once that lands in M3.
"""

from __future__ import annotations

import pytest

from devdocs_rag.retrieval.bm25 import BM25Index, tokenize

# ---------- tokenize ----------


def test_tokenize_lowercases() -> None:
    assert tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_keeps_underscored_symbols_whole() -> None:
    # `register_buffer` is the canonical exact-symbol query — must not split.
    assert tokenize("register_buffer") == ["register_buffer"]
    assert tokenize("How do I use register_buffer in nn.Module?") == [
        "how",
        "do",
        "i",
        "use",
        "register_buffer",
        "in",
        "nn",
        "module",
    ]


def test_tokenize_splits_dotted_names() -> None:
    # `nn.Linear` → ["nn", "linear"]. Symmetric with query tokenization, so
    # this is a documented choice, not a bug.
    assert tokenize("nn.Linear") == ["nn", "linear"]
    assert tokenize("torch.compile") == ["torch", "compile"]


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_only_punctuation() -> None:
    assert tokenize("!!! ??? ...") == []


# ---------- BM25Index ----------


def _corpus() -> tuple[list[str], list[str]]:
    """Tiny corpus that lets us assert ranking behavior."""
    doc_ids = ["doc-a", "doc-b", "doc-c", "doc-d"]
    texts = [
        "extending pytorch with custom autograd functions",
        "cuda stream synchronization and event recording",
        "register_buffer attaches a tensor to an nn.Module without making it a parameter",
        "the quick brown fox jumps over the lazy dog",
    ]
    return doc_ids, texts


def test_index_size_matches_input() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    assert idx.size == 4


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="must be the same length"):
        BM25Index(["a"], ["a", "b"])


def test_search_returns_doc_ids_not_indices() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("autograd", top_k=4)
    assert all(isinstance(d, str) for d, _ in out)
    assert "doc-a" in {d for d, _ in out}


def test_search_ranks_relevant_first() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("custom autograd functions", top_k=4)
    # doc-a has all three terms; should rank first.
    assert out[0][0] == "doc-a"


def test_search_finds_exact_symbol() -> None:
    """The whole point of BM25 — recall on `register_buffer` style queries."""
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("register_buffer", top_k=4)
    assert out, "BM25 should find at least one match for the exact symbol"
    assert out[0][0] == "doc-c"


def test_search_filters_zero_scores() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("autograd", top_k=4)
    # doc-d ("quick brown fox...") has no relevant terms — should be excluded.
    assert "doc-d" not in {d for d, _ in out}


def test_search_respects_top_k() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("the", top_k=2)
    assert len(out) <= 2


def test_search_empty_corpus_returns_empty() -> None:
    idx = BM25Index.from_documents([], [])
    assert idx.search("anything", top_k=10) == []


def test_search_empty_query_returns_empty() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    assert idx.search("", top_k=10) == []


def test_search_query_with_no_word_chars_returns_empty() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    assert idx.search("!!!", top_k=10) == []


def test_search_scores_are_floats_descending() -> None:
    doc_ids, texts = _corpus()
    idx = BM25Index.from_documents(doc_ids, texts)
    out = idx.search("autograd cuda", top_k=4)
    scores = [s for _, s in out]
    assert all(isinstance(s, float) for s in scores)
    assert scores == sorted(scores, reverse=True)
