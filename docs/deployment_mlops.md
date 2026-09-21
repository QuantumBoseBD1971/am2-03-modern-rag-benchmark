# Deployment, Security and MLOps

## Production architecture

A production RAG service can separate:

1. ingestion/chunking
2. embedding/indexing
3. lexical/dense retrieval
4. reranking
5. context assembly
6. generation
7. groundedness checks
8. logging/monitoring

## Versioning

Version independently:

- corpus snapshot
- chunking configuration
- embedding model
- vector index
- reranker
- prompt template
- generator
- evaluation set

## Monitoring

Track:

- retrieval Recall@K where labels exist
- no-result / low-score rate
- unsupported citation rate
- abstention rate
- latency by stage
- index freshness
- embedding drift
- prompt-injection detections

## Prompt injection

Retrieved text is untrusted input.

Recommended controls include:

- treat retrieved content as data, not instructions
- isolate system instructions from retrieved context
- detect common injection patterns
- allow-list downstream tools/actions
- avoid executing instructions found in documents
- log suspicious retrieved passages

## Rollback

Never overwrite prior model/index artefacts. Promote versions through an alias such as `production` and roll back by moving the alias to a previously validated version.
