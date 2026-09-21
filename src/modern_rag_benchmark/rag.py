"""Grounded RAG prompting and generation interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from modern_rag_benchmark.context import EvidenceChunk, render_context


SYSTEM_INSTRUCTION = (
    "Answer only from the supplied evidence. "
    "If the evidence is insufficient, say that the answer is not supported. "
    "Cite evidence using labels such as [E1] and [E2]."
)


class Generator(Protocol):
    """Minimal text-generation contract."""

    def generate(self, prompt: str) -> str:
        """Generate one answer from a prompt."""


class TransformersGenerator:
    """Optional Hugging Face text-generation adapter."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        max_new_tokens: int = 128,
    ) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                'Generation is optional. Install with: pip install -e ".[rag]"'
            ) from exc

        self.pipeline = pipeline(
            "text2text-generation",
            model=model_name,
        )
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        output = self.pipeline(
            prompt,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )[0]["generated_text"]
        return str(output)


def build_grounded_prompt(
    question: str,
    evidence: Sequence[EvidenceChunk],
) -> str:
    """Build a prompt that preserves explicit evidence labels."""
    context = render_context(list(evidence))
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Evidence:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def answer_with_evidence(
    question: str,
    evidence: Sequence[EvidenceChunk],
    generator: Generator,
) -> str:
    """Generate an answer using only rendered retrieval evidence."""
    return generator.generate(build_grounded_prompt(question, evidence))
