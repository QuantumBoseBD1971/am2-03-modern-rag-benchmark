"""Dense semantic retrieval with pluggable embedding backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from modern_rag_benchmark.types import Document, Query, SearchResult


class Encoder(Protocol):
    """Minimal encoder contract used by the dense retriever."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode texts into a 2D float array."""


class SentenceTransformerEncoder:
    """Sentence-transformer adapter loaded lazily to keep CI lightweight."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                'Dense retrieval is optional. Install with: pip install -e ".[dense]"'
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return L2-normalised embeddings."""
        embeddings = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=float)


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Embedding matrix must be two-dimensional.")

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return values / norms


class DenseRetriever:
    """In-memory cosine-similarity retriever."""

    def __init__(self, documents: list[Document], encoder: Encoder) -> None:
        if not documents:
            raise ValueError("At least one document is required.")

        self.documents = documents
        self.encoder = encoder
        corpus = [document.text for document in documents]
        self.document_embeddings = l2_normalise(encoder.encode(corpus))

        if self.document_embeddings.shape[0] != len(documents):
            raise ValueError("Encoder returned an unexpected number of document embeddings.")

    def search(self, query: Query, top_k: int = 5) -> list[SearchResult]:
        """Return top-k documents ranked by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = l2_normalise(self.encoder.encode([query.text]))[0]
        scores = self.document_embeddings @ query_embedding

        ranked_indices = np.argsort(-scores)[:top_k]

        return [
            SearchResult(
                query_id=query.query_id,
                document_id=self.documents[int(index)].document_id,
                score=float(scores[int(index)]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]
