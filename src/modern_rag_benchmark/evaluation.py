"""Groundedness and failure-analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

from modern_rag_benchmark.context import EvidenceChunk
from modern_rag_benchmark.grounding import (
    citation_precision,
    cited_evidence_ids,
    has_unsupported_citation,
)


@dataclass(frozen=True)
class GroundingEvaluation:
    citation_precision: float
    unsupported_citation: bool
    citation_count: int
    abstained: bool


def evaluate_grounded_answer(
    answer: str,
    evidence: list[EvidenceChunk],
) -> GroundingEvaluation:
    """Evaluate citation integrity and simple abstention behaviour."""
    lower = answer.lower()
    abstained = "not supported" in lower or "insufficient evidence" in lower

    return GroundingEvaluation(
        citation_precision=citation_precision(answer, evidence),
        unsupported_citation=has_unsupported_citation(answer, evidence),
        citation_count=len(cited_evidence_ids(answer)),
        abstained=abstained,
    )


def classify_failure(
    retrieved_relevant: bool,
    answer_supported: bool,
    abstained: bool,
) -> str:
    """Assign a simple retrieval/generation failure category."""
    if not retrieved_relevant:
        return "retrieval_failure"
    if abstained:
        return "over_abstention"
    if not answer_supported:
        return "generation_grounding_failure"
    return "success"
