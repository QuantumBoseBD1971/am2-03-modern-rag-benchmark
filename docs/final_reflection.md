# Final Reflection

The main lesson from this project is that RAG is not one model. It is a pipeline of retrieval, ranking, context construction and generation components.

Starting with BM25 made the neural layers measurable rather than fashionable additions. Dense retrieval tests semantic matching, reranking spends more compute on a smaller candidate set, and grounded generation introduces a new class of failure that retrieval metrics alone cannot detect.

I also learned that citation syntax is only a first control. A valid citation label does not prove the generated claim is semantically entailed by the cited text.

With more time I would add:
- semantic entailment / LLM-as-judge evaluation with human calibration
- adversarial prompt-injection benchmark
- hybrid BM25+dense fusion
- production vector database comparison
- chunk-size experiments
- multi-vector retrieval
- latency/cost benchmarking across models
