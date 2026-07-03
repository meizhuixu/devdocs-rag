"""Typed exceptions for the generation module.

CLAUDE.md error convention: each module raises its own typed exception;
the API boundary catches and translates to HTTP responses.
"""

from __future__ import annotations


class GenerationError(Exception):
    """Raised when LLM generation fails or the client is misconfigured."""
