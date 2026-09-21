"""Run the Phase 1 BM25 benchmark on the local fixture corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.fixtures import (
    demo_documents,
    demo_qrels,
    demo_queries,
)
from modern_rag_benchmark.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RESULTS_PATH = Path("results/tables/phase1_bm25_demo.csv")


def main() -> None:
    documents = demo_documents()
    queries = demo_queries()
    qrels = demo_qrels()

    retriever = BM25Retriever(documents)

    rows = []
    for query in queries:
        results = retriever.search(query, top_k=3)
        relevant = qrels[query.query_id]

        rows.append(
            {
                "query_id": query.query_id,
                "recall_at_1": recall_at_k(results, relevant, 1),
                "recall_at_3": recall_at_k(results, relevant, 3),
                "precision_at_1": precision_at_k(results, relevant, 1),
                "reciprocal_rank": reciprocal_rank(results, relevant),
            }
        )

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    print(table.to_string(index=False))
    print()
    print(f"Mean reciprocal rank: {table['reciprocal_rank'].mean():.3f}")


if __name__ == "__main__":
    main()
