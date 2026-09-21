# Phase 3 — Reranking and Grounded RAG

## Objective

Phase 3 adds two layers on top of retrieval:

1. **cross-encoder reranking**
2. **evidence-constrained answer generation**

The aim is to improve ranking quality while preserving traceability from answer back to retrieved evidence.

## Reranking

Bi-encoder dense retrieval is efficient because documents and queries are encoded independently.

A cross-encoder instead scores a query and candidate document jointly.

Conceptually:

```text
query
  ↓
first-stage retriever
  ↓
top-N candidates
  ↓
cross-encoder(query, candidate)
  ↓
reranked top-K
```

This is more computationally expensive, so it is normally applied only to a small candidate set.

The production adapter uses:

`cross-encoder/ms-marco-MiniLM-L-6-v2`

through the optional `rerank` dependency group.

## Retrieval ablation

The project compares retrieval stages independently:

- BM25
- dense retrieval
- reranked candidates

This allows a later assessor discussion around whether each layer adds measurable value.

## Grounded context construction

Retrieved documents are converted into explicit evidence chunks:

```text
[E1] first retrieved passage
[E2] second retrieved passage
[E3] third retrieved passage
```

These labels are then preserved inside the prompt.

## RAG generation

The prompt instructs the generator to:

- answer only from supplied evidence
- say when evidence is insufficient
- cite evidence labels

The optional production generator uses Hugging Face Transformers.

CI does not download an LLM; deterministic generator doubles test the prompt/citation plumbing.

## Grounding checks

Phase 3 introduces deterministic checks for:

- cited evidence ids
- citation precision
- unsupported citation detection

These are not a full semantic groundedness metric. They test citation integrity and prepare the repository for deeper answer evaluation in Phase 4.

## Next phase

Phase 4 will add:

- groundedness/failure analysis
- experiment tracking
- model card
- deployment/MLOps design
- final AM2 evidence synthesis
