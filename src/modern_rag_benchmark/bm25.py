"""BM25 lexical retrieval implementation."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from modern_rag_benchmark.text import tokenise
from modern_rag_benchmark.types import Document, Query, SearchResult


class BM25Retriever:
    """Simple in-memory BM25 retriever."""

    def __init__(self, documents: list[Document]) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        self.documents = documents
        tokenised_corpus = [tokenise(document.text) for document in documents]
        self.index = BM25Okapi(tokenised_corpus)

    def search(self, query: Query, top_k: int = 5) -> list[SearchResult]:
        """Return the top-k lexical matches for one query."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        scores = self.index.get_scores(tokenise(query.text))
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                query_id=query.query_id,
                document_id=self.documents[index].document_id,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
