"""Cross-encoder reranking with an optional production backend."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from modern_rag_benchmark.types import Document, Query, SearchResult


class PairScorer(Protocol):
    """Minimal query-document scoring contract."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Return one relevance score per document."""


class CrossEncoderScorer:
    """Sentence-Transformers cross-encoder adapter."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                'Reranking is optional. Install with: pip install -e ".[rerank]"'
            ) from exc

        self.model = CrossEncoder(model_name)

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        pairs = [(query, document) for document in documents]
        values = self.model.predict(pairs, show_progress_bar=False)
        return [float(value) for value in values]


class Reranker:
    """Rerank an existing candidate list with a pairwise scorer."""

    def __init__(
        self,
        documents: list[Document],
        scorer: PairScorer,
    ) -> None:
        self.documents = {document.document_id: document for document in documents}
        self.scorer = scorer

    def rerank(
        self,
        query: Query,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Rerank pre-retrieved candidates."""
        if not candidates:
            return []

        documents = [self.documents[result.document_id] for result in candidates]
        scores = self.scorer.score(query.text, [document.text for document in documents])

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive.")
            ranked = ranked[:top_k]

        return [
            SearchResult(
                query_id=query.query_id,
                document_id=result.document_id,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(ranked, start=1)
        ]
