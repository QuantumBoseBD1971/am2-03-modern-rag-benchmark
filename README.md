# AM2-03 — Modern Retrieval and RAG Benchmark

A comparative information-retrieval and retrieval-augmented generation project.

The project is designed to compare the evolution of retrieval systems:

**lexical retrieval → dense semantic retrieval → reranking → grounded RAG**

The first phase establishes a measurable lexical-retrieval benchmark before introducing embeddings or language models.

## Research question

> How much retrieval and answer quality is gained as we move from classical lexical search to dense retrieval, reranking and retrieval-augmented generation?

## Benchmark design

The project separates two evaluation problems:

1. **Retrieval quality** — did the system retrieve the relevant evidence?
2. **Answer quality** — did the generated answer stay grounded in that evidence?

This separation is important because a RAG system can fail either because retrieval is poor or because generation is poorly grounded.

## Phase 1 — lexical retrieval

The first layer includes:

- document/query abstractions
- deterministic text normalisation
- BM25 retrieval
- Recall@K
- Precision@K
- Mean Reciprocal Rank
- small local fixture corpus for CI
- optional public SciFact/BEIR loader for larger experiments
- tests and GitHub Actions CI

## Later phases

### Phase 2 — dense retrieval
- sentence-transformer embeddings
- cosine similarity
- vector indexing
- lexical vs semantic comparison

### Phase 3 — reranking and RAG
- cross-encoder reranking
- retrieval-stage ablation
- grounded prompting
- citation/evidence tracking

### Phase 4 — evaluation and productionisation
- answer groundedness
- retrieval failure analysis
- hallucination checks
- experiment tracking
- model card
- deployment/MLOps design
- final AM2 evidence synthesis

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts/run_bm25_demo.py
pytest
```

To use the optional public IR dataset loader:

```bash
pip install -e ".[data]"
python scripts/load_scifact.py
```

## Responsible use

This project is educational. Retrieval and RAG quality is dataset- and domain-dependent; benchmark results do not establish safe performance for high-stakes question answering.

## Licence

Code: MIT. External datasets retain their original licences and citation requirements.
