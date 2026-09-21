"""Run reranking ablation and a grounded RAG demonstration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from modern_rag_benchmark.benchmark import evaluate_retriever, summarise_benchmark
from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.context import build_evidence
from modern_rag_benchmark.fixtures import demo_documents, demo_qrels, demo_queries
from modern_rag_benchmark.rag import answer_with_evidence
from modern_rag_benchmark.rerank import Reranker

RESULTS_PATH = Path("results/tables/phase3_reranking_demo.csv")


class KeywordPairScorer:
    """Deterministic scorer used for the local demo and CI-safe execution."""

    def score(self, query: str, documents):
        query_terms = set(query.lower().split())
        return [
            float(len(query_terms & set(document.lower().split())))
            for document in documents
        ]


class EvidenceEchoGenerator:
    """Deterministic local generator for demonstrating citation plumbing."""

    def generate(self, prompt: str) -> str:
        if "[E1]" in prompt:
            return "The supplied evidence supports this answer [E1]."
        return "The answer is not supported by the supplied evidence."


class RerankedRetriever:
    def __init__(self, base, reranker, query_lookup, candidate_k: int = 3):
        self.base = base
        self.reranker = reranker
        self.query_lookup = query_lookup
        self.candidate_k = candidate_k

    def search(self, query, top_k: int = 3):
        candidates = self.base.search(query, top_k=self.candidate_k)
        return self.reranker.rerank(query, candidates, top_k=top_k)


def main() -> None:
    documents = demo_documents()
    queries = demo_queries()
    qrels = demo_qrels()

    bm25 = BM25Retriever(documents)
    reranker = Reranker(documents, KeywordPairScorer())
    reranked = RerankedRetriever(
        bm25,
        reranker,
        {query.query_id: query for query in queries},
    )

    rows = []
    for name, retriever in [("bm25", bm25), ("bm25_reranked", reranked)]:
        results = evaluate_retriever(retriever, queries, qrels, top_k=3)
        rows.append(summarise_benchmark(results, name))

    table = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_PATH, index=False)

    query = queries[0]
    evidence = build_evidence(documents, reranked.search(query, top_k=2), top_k=2)
    answer = answer_with_evidence(
        query.text,
        evidence,
        EvidenceEchoGenerator(),
    )

    print(table.to_string(index=False))
    print()
    print(f"Grounded demo answer: {answer}")


if __name__ == "__main__":
    main()
