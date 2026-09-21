# Final Project Summary

## Research question

How much retrieval and answer quality is gained as a system evolves from lexical search to dense retrieval, reranking and grounded RAG?

## Technical progression

```text
BM25
  ↓
dense embeddings
  ↓
cross-encoder reranking
  ↓
labelled evidence context
  ↓
grounded generation
  ↓
citation / failure analysis
```

## Key engineering decisions

- retrieval and generation evaluated separately
- same relevance judgements reused across retrievers
- optional heavy dependencies kept out of normal CI
- model backends hidden behind small interfaces
- evidence labels preserved end-to-end
- retrieved text treated as untrusted input
- failure taxonomy separates retrieval from generation issues

## Interpretation

A better LLM cannot compensate for missing evidence, and better retrieval does not guarantee a grounded answer.

A production RAG system therefore needs evaluation at each stage rather than one end-to-end score.
