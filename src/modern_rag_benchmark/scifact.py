"""Optional loader for the public BEIR SciFact benchmark."""

from __future__ import annotations

from modern_rag_benchmark.types import Document, Query


def load_scifact(
    dataset_id: str = "beir/scifact/test",
) -> tuple[list[Document], list[Query], dict[str, set[str]]]:
    """Load SciFact from ir_datasets.

    The dependency is optional so normal CI remains lightweight.
    """
    try:
        import ir_datasets
    except ImportError as exc:
        raise ImportError(
            'SciFact loading is optional. Install with: pip install -e ".[data]"'
        ) from exc

    dataset = ir_datasets.load(dataset_id)

    documents = [
        Document(
            document_id=str(document.doc_id),
            title=getattr(document, "title", "") or "",
            text=getattr(document, "text", "") or "",
        )
        for document in dataset.docs_iter()
    ]

    queries = [
        Query(
            query_id=str(query.query_id),
            text=str(query.text),
        )
        for query in dataset.queries_iter()
    ]

    qrels: dict[str, set[str]] = {}
    for qrel in dataset.qrels_iter():
        if int(qrel.relevance) > 0:
            qrels.setdefault(str(qrel.query_id), set()).add(str(qrel.doc_id))

    return documents, queries, qrels
