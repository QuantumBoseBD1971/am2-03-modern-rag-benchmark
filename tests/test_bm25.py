from modern_rag_benchmark.bm25 import BM25Retriever
from modern_rag_benchmark.fixtures import demo_documents, demo_qrels, demo_queries
from modern_rag_benchmark.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_bm25_retrieves_expected_document() -> None:
    retriever = BM25Retriever(demo_documents())
    query = demo_queries()[0]

    results = retriever.search(query, top_k=2)

    assert results[0].document_id == "d1"
    assert results[0].rank == 1


def test_retrieval_metrics() -> None:
    retriever = BM25Retriever(demo_documents())
    query = demo_queries()[0]
    relevant = demo_qrels()[query.query_id]
    results = retriever.search(query, top_k=3)

    assert recall_at_k(results, relevant, 1) == 1.0
    assert precision_at_k(results, relevant, 1) == 1.0
    assert reciprocal_rank(results, relevant) == 1.0
