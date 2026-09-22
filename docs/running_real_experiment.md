# Running the real RAG experiment

The manual **Run real RAG experiment** workflow executes Project 03 and stores real outputs.

## Standard run

Leave both optional inputs disabled to run:

1. BM25 lexical retrieval
2. real SentenceTransformer dense retrieval
3. deterministic reranking/RAG orchestration
4. evidence-pack generation

This is the recommended first run.

## Full model run

Enable **Run cross-encoder reranking and Hugging Face generation** to additionally download and execute:

- cross-encoder/ms-marco-MiniLM-L-6-v2
- google/flan-t5-small

The resulting generated answers and citation-integrity metrics are retained in the evidence pack.

## Public benchmark

Enable **Run the larger public SciFact BM25-vs-dense benchmark** to run retrieval on the BEIR SciFact benchmark through ir-datasets.

## Evidence

The workflow uploads the full results directory as an Actions artifact for 90 days and can commit a compact evidence directory back into Git.
