"""Small local benchmark corpus used for CI and smoke tests."""

from modern_rag_benchmark.types import Document, Query


def demo_documents() -> list[Document]:
    return [
        Document(
            document_id="d1",
            title="Battery storage",
            text="Lithium batteries should be stored in a cool dry place away from heat.",
        ),
        Document(
            document_id="d2",
            title="Solar panels",
            text="Solar photovoltaic panels convert sunlight into electrical energy.",
        ),
        Document(
            document_id="d3",
            title="Battery charging",
            text="Battery charging systems should monitor voltage and temperature.",
        ),
        Document(
            document_id="d4",
            title="Wind turbines",
            text="Wind turbines convert kinetic energy from wind into electricity.",
        ),
    ]


def demo_queries() -> list[Query]:
    return [
        Query(query_id="q1", text="how should lithium batteries be stored"),
        Query(query_id="q2", text="what converts sunlight to electricity"),
    ]


def demo_qrels() -> dict[str, set[str]]:
    return {
        "q1": {"d1"},
        "q2": {"d2"},
    }
