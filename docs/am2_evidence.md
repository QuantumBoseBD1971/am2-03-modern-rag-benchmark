# AM2 Evidence Notes

This document evolves with the retrieval/RAG project.

## Current evidence

### Problem framing
Question answering is decomposed into evidence retrieval and answer generation.

### Classical information retrieval
Phase 1 implements BM25 as an explicit lexical baseline.

### Evaluation
The project measures retrieval quality with:

- Recall@K
- Precision@K
- Reciprocal Rank
- Mean Reciprocal Rank

### Software engineering
The repository uses typed data structures, packaged modules, unit tests and GitHub Actions CI.

### Reproducibility
A local fixture corpus supports deterministic CI, while an optional public benchmark loader supports larger offline experiments.

### Responsible design
Retrieval quality is evaluated separately from generation quality so later RAG failures can be attributed to retrieval vs generation.

## Evidence still to add

- dense semantic embeddings
- vector similarity search
- lexical vs semantic benchmark
- cross-encoder reranking
- retrieval-stage ablation
- grounded generation
- citation/evidence tracking
- groundedness and hallucination evaluation
- experiment tracking
- model card
- deployment/MLOps design
- final reflection
