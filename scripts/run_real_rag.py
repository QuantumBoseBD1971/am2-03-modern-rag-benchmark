"""Run real cross-encoder reranking and local Hugging Face generation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.context import build_evidence
from modern_rag_benchmark.evaluation import evaluate_grounded_answer
from modern_rag_benchmark.fixtures import demo_documents, demo_qrels, demo_queries
from modern_rag_benchmark.metrics import precision_at_k, recall_at_k, reciprocal_rank
from modern_rag_benchmark.rag import TransformersGenerator, answer_with_evidence
from modern_rag_benchmark.rerank import CrossEncoderScorer, Reranker

RESULTS_PATH = Path("results/tables/phase3_real_rag.csv")


def main() -> None:
    documents = demo_documents()
    queries = demo_queries()
    qrels = demo_qrels()

    retriever = BM25Retriever(documents)
    reranker = Reranker(documents, CrossEncoderScorer())
    generator = TransformersGenerator()

    rows = []

    for query in queries:
        candidates = retriever.search(query, top_k=3)
        reranked = reranker.rerank(query, candidates, top_k=3)
        evidence = build_evidence(documents, reranked, top_k=2)
        answer = answer_with_evidence(query.text, evidence, generator)
        grounding = evaluate_grounded_answer(answer, evidence)
        relevant = qrels.get(query.query_id, set())

        rows.append(
            {
                "query_id": query.query_id,
                "answer": answer,
                "recall_at_1": recall_at_k(reranked, relevant, 1),
                "recall_at_3": recall_at_k(reranked, relevant, 3),
                "precision_at_1": precision_at_k(reranked, relevant, 1),
                "reciprocal_rank": reciprocal_rank(reranked, relevant),
                **asdict(grounding),
            }
        )

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    print(table.to_string(index=False))
    print()
    print("Mean retrieval/grounding metrics:")
    print(
        table[
            [
                "recall_at_1",
                "recall_at_3",
                "precision_at_1",
                "reciprocal_rank",
                "citation_precision",
            ]
        ]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
