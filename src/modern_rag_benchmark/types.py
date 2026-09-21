"""Core retrieval data structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    document_id: str
    text: str
    title: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    query_id: str
    document_id: str
    score: float
    rank: int
