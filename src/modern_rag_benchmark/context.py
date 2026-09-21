"""Grounded context construction for retrieval-augmented generation."""

from __future__ import annotations

from dataclasses import dataclass

from modern_rag_benchmark.types import Document, SearchResult


@dataclass(frozen=True)
class EvidenceChunk:
    citation_id: str
    document_id: str
    text: str
    rank: int


def build_evidence(
    documents: list[Document],
    results: list[SearchResult],
    top_k: int = 3,
) -> list[EvidenceChunk]:
    """Convert ranked results into explicitly citable evidence chunks."""
    document_map = {document.document_id: document for document in documents}
    evidence = []

    for result in results[:top_k]:
        document = document_map[result.document_id]
        evidence.append(
            EvidenceChunk(
                citation_id=f"E{len(evidence) + 1}",
                document_id=document.document_id,
                text=document.text,
                rank=result.rank,
            )
        )

    return evidence


def render_context(evidence: list[EvidenceChunk]) -> str:
    """Render evidence for inclusion in a grounded prompt."""
    return "\n\n".join(
        f"[{item.citation_id}] {item.text}"
        for item in evidence
    )
