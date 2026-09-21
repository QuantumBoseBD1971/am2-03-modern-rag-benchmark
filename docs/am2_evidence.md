# AM2 Evidence Notes

This document evolves with the retrieval/RAG project.

## Problem framing

Question answering is decomposed into retrieval, reranking and generation so failure modes can be analysed independently.

## Classical and dense retrieval

Phase 1 establishes BM25.

Phase 2 adds sentence-transformer embeddings, cosine similarity and direct lexical-vs-semantic benchmarking.

## Reranking

Phase 3 introduces a pluggable pairwise scoring interface and an optional cross-encoder implementation.

This demonstrates the difference between:

- efficient first-stage retrieval
- more expensive joint query-document scoring

## Grounded RAG

Retrieved passages are converted into explicitly labelled evidence chunks such as `[E1]` and `[E2]`.

The generation prompt requires answers to use only supplied evidence and cite these labels.

## Evaluation

Retrieval remains evaluated with:

- Recall@K
- Precision@K
- Reciprocal Rank

Phase 3 additionally introduces citation-integrity checks:

- valid cited evidence ids
- citation precision
- unsupported citation detection

## Reproducibility and dependency isolation

Cross-encoder and LLM dependencies are optional.

CI uses deterministic scorer/generator doubles, allowing the orchestration and evidence plumbing to be tested without downloading large models.

## Engineering design

Retriever, encoder, reranker and generator components use small interfaces so model implementations can be changed without rewriting the evaluation pipeline.

## Evidence still to add

- semantic groundedness evaluation
- retrieval/generation failure taxonomy
- experiment tracking
- model card
- deployment/MLOps design
- monitoring and prompt-injection considerations
- final reflection
