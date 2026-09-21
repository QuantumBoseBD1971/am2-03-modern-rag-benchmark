# AM2 Evidence Notes

This repository demonstrates a modular retrieval-augmented generation lifecycle.

## Information retrieval
- BM25 lexical baseline
- dense semantic retrieval
- cosine similarity
- common qrel-based evaluation

## Reranking
- pairwise query-document scoring
- optional cross-encoder
- candidate-stage ablation

## RAG
- labelled evidence chunks
- grounded prompt construction
- optional generator backend
- citation integrity checks

## Evaluation and failure analysis
- Recall@K
- Precision@K
- Reciprocal Rank
- citation precision
- unsupported-citation detection
- abstention detection
- retrieval-vs-generation failure taxonomy

## Security
Retrieved content is explicitly treated as untrusted input. The project includes simple prompt-injection detection and documents stronger production controls.

## Engineering and MLOps
- modular interfaces
- optional model dependencies
- deterministic CI doubles
- experiment tracking
- independent component versioning
- monitoring and rollback design

## Responsible AI
Citation presence is not treated as proof of factual correctness. High-stakes use requires stronger semantic groundedness evaluation and human oversight.
