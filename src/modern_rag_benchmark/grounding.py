"""Simple deterministic groundedness checks for generated answers."""

from __future__ import annotations

import re

from modern_rag_benchmark.context import EvidenceChunk


CITATION_PATTERN = re.compile(r"\[(E\d+)\]")


def cited_evidence_ids(answer: str) -> set[str]:
    """Extract evidence citation labels from an answer."""
    return set(CITATION_PATTERN.findall(answer))


def citation_precision(answer: str, evidence: list[EvidenceChunk]) -> float:
    """Return fraction of cited labels that correspond to supplied evidence."""
    cited = cited_evidence_ids(answer)
    if not cited:
        return 0.0

    valid = {item.citation_id for item in evidence}
    return len(cited & valid) / len(cited)


def has_unsupported_citation(answer: str, evidence: list[EvidenceChunk]) -> bool:
    """Return True when the answer cites evidence not supplied in context."""
    cited = cited_evidence_ids(answer)
    valid = {item.citation_id for item in evidence}
    return bool(cited - valid)
