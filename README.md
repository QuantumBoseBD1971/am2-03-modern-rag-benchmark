# AM2-03 — Modern Retrieval and RAG Benchmark

A comparative information-retrieval and retrieval-augmented generation project.

The project follows:

**BM25 lexical retrieval → dense semantic retrieval → reranking → grounded RAG**

## Research question

> How much retrieval and answer quality is gained as we move from classical lexical search to dense retrieval, reranking and retrieval-augmented generation?

## Current retrieval layers

### Lexical
- BM25
- deterministic tokenisation

### Dense semantic
- sentence-transformer embeddings
- cosine similarity
- in-memory vector index
- pluggable encoder backend

### Evaluation
- Recall@1 / @3 / @10
- Precision@1 / @3
- Reciprocal Rank

The same relevance judgements are used for both retrievers.

## Quick start

Core CI-safe setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/run_bm25_demo.py
pytest
```

Dense semantic benchmark:

```bash
pip install -e ".[dense]"
python scripts/run_dense_comparison.py
```

Public SciFact benchmark:

```bash
pip install -e ".[data,dense]"
python scripts/run_scifact_comparison.py
```

## Development status

- **Phase 1 — complete:** BM25 and lexical IR metrics.
- **Phase 2 — in progress:** dense embeddings and BM25-vs-semantic comparison.
- **Phase 3 — planned:** cross-encoder reranking and grounded RAG.
- **Phase 4 — planned:** groundedness, failure analysis, experiment tracking and MLOps.

## Responsible use

Retrieval quality and RAG quality are domain-dependent. This benchmark does not establish suitability for high-stakes question answering.

## Licence

Code: MIT. External datasets/models retain their original licences.
