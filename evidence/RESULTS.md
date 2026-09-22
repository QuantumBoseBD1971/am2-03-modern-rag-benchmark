# Real Experiment Results

This evidence pack was generated from executed retrieval/RAG workflows.

## BM25 lexical retrieval

- Mean Recall@1: **1.0000**
- Mean Recall@3: **1.0000**
- Mean Precision@1: **1.0000**
- Mean reciprocal rank: **1.0000**

## BM25 vs dense semantic retrieval

- bm25: Recall@1=1.0000, Recall@3=1.0000, MRR=1.0000
- dense_minilm: Recall@1=1.0000, Recall@3=1.0000, MRR=1.0000

## Evidence files

- tables/phase1_bm25_demo.csv
- tables/phase2_retrieval_comparison.csv
- tables/phase2_scifact_comparison.csv when enabled
- tables/phase3_reranking_demo.csv
- tables/phase3_real_rag.csv when full models are enabled
- final_project_summary.json

Generated automatically by scripts/build_evidence_pack.py.
