"""Information-retrieval evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable

from modern_rag_benchmark.types import SearchResult


def relevant_documents(
    qrels: dict[str, set[str]],
    query_id: str,
) -> set[str]:
    """Return known relevant document ids for one query."""
    return qrels.get(query_id, set())


def recall_at_k(
    results: list[SearchResult],
    relevant: set[str],
    k: int,
) -> float:
    """Compute Recall@K."""
    if not relevant:
        return 0.0
    retrieved = {result.document_id for result in results[:k]}
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(
    results: list[SearchResult],
    relevant: set[str],
    k: int,
) -> float:
    """Compute Precision@K."""
    if k <= 0:
        raise ValueError("k must be positive.")
    retrieved = [result.document_id for result in results[:k]]
    return sum(document_id in relevant for document_id in retrieved) / k


def reciprocal_rank(
    results: list[SearchResult],
    relevant: set[str],
) -> float:
    """Return reciprocal rank of the first relevant result."""
    for result in results:
        if result.document_id in relevant:
            return 1.0 / result.rank
    return 0.0


def mean_reciprocal_rank(
    ranked_results: Iterable[tuple[list[SearchResult], set[str]]],
) -> float:
    """Compute mean reciprocal rank across multiple queries."""
    values = [reciprocal_rank(results, relevant) for results, relevant in ranked_results]
    if not values:
        return 0.0
    return sum(values) / len(values)
