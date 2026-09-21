import numpy as np

from modern_rag_benchmark.dense import DenseRetriever, l2_normalise
from modern_rag_benchmark.types import Document, Query


class FakeEncoder:
    def encode(self, texts):
        mapping = {
            "cats purr": [1.0, 0.0],
            "dogs bark": [0.0, 1.0],
            "feline animal": [1.0, 0.1],
        }
        return np.asarray([mapping[text] for text in texts], dtype=float)


def test_l2_normalise_rows() -> None:
    values = l2_normalise(np.array([[3.0, 4.0]]))
    assert np.allclose(values, np.array([[0.6, 0.8]]))


def test_dense_retriever_ranks_semantic_match_first() -> None:
    documents = [
        Document(document_id="d1", text="cats purr"),
        Document(document_id="d2", text="dogs bark"),
    ]
    retriever = DenseRetriever(documents, FakeEncoder())

    results = retriever.search(
        Query(query_id="q1", text="feline animal"),
        top_k=2,
    )

    assert results[0].document_id == "d1"
    assert results[0].rank == 1
