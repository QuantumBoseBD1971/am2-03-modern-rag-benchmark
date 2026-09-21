# Phase 2 — Dense Semantic Retrieval

## Objective

Phase 2 compares classical lexical retrieval against neural semantic retrieval under the same relevance judgements.

The key question is:

> Does semantic representation retrieve relevant evidence when the wording of the query differs from the wording of the document?

## Dense retrieval

Each document and query is mapped to an embedding vector.

For a query vector (q) and document vector (d), cosine similarity is:

[
cos(q,d)=\frac{q \cdot d}{\|q\|\|d\|}
]

The implementation L2-normalises embeddings so cosine similarity reduces to a dot product.

## Sentence-transformer backend

The default optional model is:

`sentence-transformers/all-MiniLM-L6-v2`

The dependency is isolated behind the `dense` optional install:

```bash
pip install -e ".[dense]"
```

Normal CI uses a deterministic fake encoder, so pull requests do not need to download model weights.

## Comparison design

BM25 and dense retrieval are evaluated with the same:

- queries
- relevance judgements
- Recall@K
- Precision@K
- reciprocal rank

This creates an explicit ablation rather than presenting semantic retrieval in isolation.

## Why vector search is kept in memory

At this stage the goal is to isolate retrieval quality, not database infrastructure.

The in-memory vector matrix makes the similarity calculation transparent and testable.

A production implementation could later replace it with FAISS, pgvector, OpenSearch, Qdrant, Pinecone or another vector index without changing the benchmark contract.

## Public benchmark

The optional SciFact loader allows the same BM25-vs-dense comparison to run against a recognised public IR benchmark.

## Next phase

Phase 3 will add:

- cross-encoder reranking
- retrieval-stage ablation
- grounded context construction
- RAG generation with evidence references
