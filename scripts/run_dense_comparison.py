"""Compare BM25 and dense semantic retrieval on a benchmark corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_rag_benchmark.benchmark import evaluate_retriever, summarise_benchmark
from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.dense import DenseRetriever, SentenceTransformerEncoder
from modern_rag_benchmark.fixtures import demo_documents, demo_qrels, demo_queries

RESULTS_PATH = Path("results/tables/phase2_retrieval_comparison.csv")


def main() -> None:
    documents = demo_documents()
    queries = demo_queries()
    qrels = demo_qrels()

    bm25 = BM25Retriever(documents)
    dense = DenseRetriever(
        documents,
        SentenceTransformerEncoder(),
    )

    rows = []

    bm25_results = evaluate_retriever(bm25, queries, qrels, top_k=3)
    rows.append(summarise_benchmark(bm25_results, "bm25"))

    dense_results = evaluate_retriever(dense, queries, qrels, top_k=3)
    rows.append(summarise_benchmark(dense_results, "dense_minilm"))

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
