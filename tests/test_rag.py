from modern_rag_benchmark.context import build_evidence, render_context
from modern_rag_benchmark.grounding import (
    citation_precision,
    has_unsupported_citation,
)
from modern_rag_benchmark.rag import build_grounded_prompt
from modern_rag_benchmark.types import Document, SearchResult


def _evidence():
    documents = [Document(document_id="d1", text="Batteries should be kept cool.")]
    results = [SearchResult(query_id="q1", document_id="d1", score=1.0, rank=1)]
    return build_evidence(documents, results)


def test_context_contains_evidence_label() -> None:
    evidence = _evidence()
    assert render_context(evidence) == "[E1] Batteries should be kept cool."


def test_grounded_prompt_contains_question_and_evidence() -> None:
    prompt = build_grounded_prompt("How should batteries be stored?", _evidence())
    assert "[E1]" in prompt
    assert "How should batteries be stored?" in prompt


def test_citation_validation() -> None:
    evidence = _evidence()
    assert citation_precision("Keep them cool [E1].", evidence) == 1.0
    assert citation_precision("Keep them cool [E2].", evidence) == 0.0
    assert has_unsupported_citation("Claim [E2].", evidence) is True
