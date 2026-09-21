# AM2-03 — Modern Retrieval and RAG Benchmark

A comparative information-retrieval and retrieval-augmented generation project.

The project follows:

**BM25 → dense semantic retrieval → cross-encoder reranking → grounded RAG**

## Current architecture

```text
Question
   ↓
First-stage retrieval
   ↓
Candidate evidence
   ↓
Cross-encoder reranking
   ↓
[E1] [E2] [E3] context
   ↓
Grounded generator
   ↓
Answer with evidence references
```

## Implemented layers

- BM25 lexical retrieval
- sentence-transformer dense retrieval
- cosine similarity
- optional cross-encoder reranking
- evidence-labelled context construction
- optional Transformers generator
- citation-integrity checks
- Recall@K / Precision@K / Reciprocal Rank

## Quick start

CI-safe core:

```bash
pip install -e ".[dev]"
python scripts/run_bm25_demo.py
python scripts/run_reranking_rag_demo.py
pytest
```

Dense retrieval:

```bash
pip install -e ".[dense]"
python scripts/run_dense_comparison.py
```

Production reranker / generator dependencies:

```bash
pip install -e ".[rerank,rag]"
```

## Development status

- **Phase 1 — complete:** lexical retrieval.
- **Phase 2 — complete:** dense semantic retrieval.
- **Phase 3 — in progress:** reranking and grounded RAG.
- **Phase 4 — planned:** groundedness/failure analysis, experiment tracking, model card and MLOps.

## Responsible use

Citation presence does not prove factual correctness. Generated answers require evidence-quality checks, semantic groundedness evaluation and domain-appropriate human oversight.

## Licence

Code: MIT. External datasets/models retain their original licences.
