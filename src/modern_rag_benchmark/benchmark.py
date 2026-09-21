"""Shared retrieval benchmark utilities."""

from __future__ import annotations

import pandas as pd

from modern_rag_benchmark.metrics import precision_at_k, recall_at_k, reciprocal_rank
from modern_rag_benchmark.types import Query


def evaluate_retriever(
    retriever,
    queries: list[Query],
    qrels: dict[str, set[str]],
    top_k: int = 10,
) -> pd.DataFrame:
    """Evaluate one retriever query-by-query."""
    rows = []

    for query in queries:
        results = retriever.search(query, top_k=top_k)
        relevant = qrels.get(query.query_id, set())

        rows.append(
            {
                "query_id": query.query_id,
                "recall_at_1": recall_at_k(results, relevant, 1),
                "recall_at_3": recall_at_k(results, relevant, 3),
                "recall_at_10": recall_at_k(results, relevant, 10),
                "precision_at_1": precision_at_k(results, relevant, 1),
                "precision_at_3": precision_at_k(results, relevant, 3),
                "reciprocal_rank": reciprocal_rank(results, relevant),
            }
        )

    return pd.DataFrame(rows)


def summarise_benchmark(
    results: pd.DataFrame,
    retriever_name: str,
) -> dict[str, float | str]:
    """Return mean retrieval metrics for one retriever."""
    metric_columns = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_10",
        "precision_at_1",
        "precision_at_3",
        "reciprocal_rank",
    ]
    summary: dict[str, float | str] = {"retriever": retriever_name}
    for column in metric_columns:
        summary[column] = float(results[column].mean())
    return summary
