"""Deterministic text preprocessing for lexical retrieval."""

from __future__ import annotations

import re


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenise(text: str) -> list[str]:
    """Lowercase and tokenise alphanumeric terms."""
    return TOKEN_PATTERN.findall(text.lower())
