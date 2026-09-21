# AM2-03 — Modern Retrieval and RAG Benchmark

A comparative project covering:

**BM25 → dense retrieval → cross-encoder reranking → grounded RAG**

## Architecture

```text
Question
  ↓
BM25 / dense retrieval
  ↓
top-N candidates
  ↓
cross-encoder reranking
  ↓
[E1] [E2] [E3]
  ↓
grounded generator
  ↓
citation + failure analysis
```

## Evaluation

Retrieval:
- Recall@1 / @3 / @10
- Precision@1 / @3
- Reciprocal Rank

Grounding:
- citation precision
- unsupported-citation detection
- abstention detection
- failure taxonomy

Security:
- prompt-injection pattern detection
- retrieved-context trust boundary documented

## Quick start

```bash
pip install -e ".[dev]"
python scripts/run_bm25_demo.py
python scripts/run_reranking_rag_demo.py
pytest
```

Optional model stacks:

```bash
pip install -e ".[dense]"
pip install -e ".[rerank,rag]"
```

## Development status

- **Phase 1 — complete:** BM25 lexical retrieval
- **Phase 2 — complete:** dense semantic retrieval
- **Phase 3 — complete:** reranking and grounded RAG
- **Phase 4 — complete:** failure analysis, security, experiment tracking, model card and MLOps

## Documentation

See `docs/` for methodology, model card, deployment/MLOps, AM2 evidence and final reflection.

## Responsible use

This is an educational benchmark. Citation presence does not prove correctness, and retrieved content must be treated as untrusted input.

## Licence

Code: MIT. External datasets and models retain their original licences.
