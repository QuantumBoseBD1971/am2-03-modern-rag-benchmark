# Model Card

## Scope

This repository benchmarks a modular RAG pipeline:

- BM25 lexical retrieval
- dense embedding retrieval
- cross-encoder reranking
- grounded generation

## Intended use

Educational benchmarking and AM2 portfolio evidence for retrieval and grounded question-answering systems.

## Out-of-scope use

The system must not be treated as a trusted source for medical, legal, financial, safety-critical or other high-stakes decisions.

## Evaluation

Retrieval:
- Recall@K
- Precision@K
- Reciprocal Rank

Generation/grounding:
- citation precision
- unsupported-citation detection
- abstention behaviour
- retrieval-vs-generation failure taxonomy

## Risks

- relevant evidence may not be retrieved
- retrieved evidence may itself be wrong or malicious
- citation presence does not guarantee semantic support
- prompt injection may be present in retrieved text
- embedding/reranker behaviour can shift with model changes
- generator models may hallucinate or over-abstain

## Human oversight

Human review remains necessary for high-impact use cases and for benchmark claims that depend on external datasets or model behaviour.
