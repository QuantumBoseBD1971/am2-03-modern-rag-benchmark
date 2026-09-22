# Real Experiment Results

This evidence pack consolidates outputs from all completed retrieval/RAG workflow runs.
Current-run outputs take precedence, while previously committed evidence is retained across specialised runs.

## BM25 lexical retrieval

- Mean Recall@1: **1.0000**
- Mean Recall@3: **1.0000**
- Mean Precision@1: **1.0000**
- Mean reciprocal rank: **1.0000**

## BM25 vs dense semantic retrieval

- bm25: Recall@1=1.0000, Recall@3=1.0000, MRR=1.0000
- dense_minilm: Recall@1=1.0000, Recall@3=1.0000, MRR=1.0000

## Public SciFact benchmark

- bm25: Recall@10=0.7590, Precision@3=0.2367, MRR=0.6041
- dense_minilm: Recall@10=0.7867, Precision@3=0.2344, MRR=0.5979

## Real cross-encoder + generator run

- Mean Recall@1: **1.0000**
- Mean MRR: **1.0000**
- Mean citation precision: **1.0000**
- Unsupported-citation rate: **0.0000**
- Abstention rate: **0.0000**

### Generated answers

- q1: [E1]
- q2: [E2]

## Evidence files

- tables/phase1_bm25_demo.csv
- tables/phase2_retrieval_comparison.csv
- tables/phase2_scifact_comparison.csv
- tables/phase3_reranking_demo.csv
- tables/phase3_real_rag.csv
- final_project_summary.json

Generated automatically by scripts/build_evidence_pack.py.
