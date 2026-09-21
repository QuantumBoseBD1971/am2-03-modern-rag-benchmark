"""Run BM25-vs-dense comparison on the optional SciFact benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_rag_benchmark.benchmark import evaluate_retriever, summarise_benchmark
from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.dense import DenseRetriever, SentenceTransformerEncoder
from modern_rag_benchmark.scifact import load_scifact

RESULTS_PATH = Path("results/tables/phase2_scifact_comparison.csv")


def main() -> None:
    documents, queries, qrels = load_scifact()

    bm25 = BM25Retriever(documents)
    dense = DenseRetriever(documents, SentenceTransformerEncoder())

    rows = []
    for name, retriever in [("bm25", bm25), ("dense_minilm", dense)]:
        results = evaluate_retriever(retriever, queries, qrels, top_k=10)
        rows.append(summarise_benchmark(results, name))

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
