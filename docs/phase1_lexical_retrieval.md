# Phase 1 — Lexical Retrieval Benchmark

## Objective

Establish a classical information-retrieval baseline before introducing neural embeddings or generative models.

## Why start with BM25

A modern RAG system should not assume semantic embeddings are automatically superior.

BM25 remains a strong lexical baseline because it rewards term overlap while normalising for document length and term frequency.

Starting with BM25 provides a transparent answer to:

> How much value do later neural retrieval layers actually add?

## Evaluation design

The project separates retrieval evaluation from answer generation.

Phase 1 measures:

- Recall@K
- Precision@K
- Reciprocal Rank
- Mean Reciprocal Rank

### Recall@K

The fraction of all relevant documents found in the first K results.

### Precision@K

The fraction of the first K results that are relevant.

### Reciprocal Rank

The reciprocal of the rank position of the first relevant result.

For example:

```text
rank 1 → 1.00
rank 2 → 0.50
rank 4 → 0.25
```

## Local fixture vs external benchmark

The repository contains a tiny deterministic fixture corpus so:

- CI is fast
- tests are reproducible
- no external download is required for every pull request

For larger experiments the optional `ir-datasets` dependency can load the public BEIR SciFact benchmark.

This keeps infrastructure concerns separate from retrieval logic.

## Next phase

Phase 2 will compare BM25 against dense semantic retrieval using sentence-transformer embeddings and cosine similarity.
