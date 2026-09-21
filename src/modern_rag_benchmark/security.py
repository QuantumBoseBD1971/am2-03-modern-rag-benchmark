"""Prompt-injection and context-safety helpers."""

from __future__ import annotations

import re

INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|the) previous instructions", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"reveal .*instructions", re.IGNORECASE),
    re.compile(r"do not follow .*instructions", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """Return True when retrieved text contains common instruction-override language."""
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)
