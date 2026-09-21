from modern_rag_benchmark.benchmark import evaluate_retriever, summarise_benchmark
from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.fixtures import demo_documents, demo_qrels, demo_queries


def test_shared_benchmark_returns_expected_columns() -> None:
    retriever = BM25Retriever(demo_documents())
    results = evaluate_retriever(
        retriever,
        demo_queries(),
        demo_qrels(),
        top_k=3,
    )

    expected = {
        "query_id",
        "recall_at_1",
        "recall_at_3",
        "recall_at_10",
        "precision_at_1",
        "precision_at_3",
        "reciprocal_rank",
    }
    assert expected.issubset(results.columns)

    summary = summarise_benchmark(results, "bm25")
    assert summary["retriever"] == "bm25"
