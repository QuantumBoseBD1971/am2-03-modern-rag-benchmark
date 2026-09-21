from modern_rag_benchmark.rerank import Reranker
from modern_rag_benchmark.types import Document, Query, SearchResult


class FakeScorer:
    def score(self, query, documents):
        return [0.1, 0.9]


def test_reranker_changes_candidate_order() -> None:
    documents = [
        Document(document_id="d1", text="first"),
        Document(document_id="d2", text="second"),
    ]
    candidates = [
        SearchResult(query_id="q1", document_id="d1", score=2.0, rank=1),
        SearchResult(query_id="q1", document_id="d2", score=1.0, rank=2),
    ]

    results = Reranker(documents, FakeScorer()).rerank(
        Query(query_id="q1", text="query"),
        candidates,
    )

    assert [result.document_id for result in results] == ["d2", "d1"]
    assert [result.rank for result in results] == [1, 2]
