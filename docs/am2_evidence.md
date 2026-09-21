# AM2 Evidence Notes

This document evolves with the retrieval/RAG project.

## Problem framing

Question answering is decomposed into retrieval and generation so failures can be attributed correctly.

## Classical information retrieval

Phase 1 implements BM25 as the lexical baseline.

## Dense semantic retrieval

Phase 2 adds:

- sentence-transformer embeddings
- L2 vector normalisation
- cosine-similarity ranking
- a pluggable encoder interface
- direct lexical-vs-semantic benchmarking

The production model dependency is optional, while deterministic fake embeddings are used in CI tests.

## Evaluation

Retrieval quality is measured consistently with:

- Recall@1 / Recall@3 / Recall@10
- Precision@1 / Precision@3
- Reciprocal Rank

A common benchmark helper ensures BM25 and dense retrieval are compared under the same qrels.

## Reproducibility

The repository separates:

- deterministic local fixtures for CI
- optional public SciFact experiments
- optional sentence-transformer dependencies

This makes the core package testable without external model downloads.

## Engineering design

The dense retriever depends on a minimal encoder protocol rather than a specific framework.

This makes the vector-search layer replaceable and supports future use of different embedding models or vector databases.

## Evidence still to add

- cross-encoder reranking
- retrieval-stage ablation
- grounded generation
- citation/evidence tracking
- groundedness/hallucination evaluation
- experiment tracking
- model card
- deployment/MLOps design
- final reflection
