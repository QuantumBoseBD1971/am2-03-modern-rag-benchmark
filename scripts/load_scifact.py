"""Load the optional SciFact benchmark and print corpus statistics."""

from modern_rag_benchmark.scifact import load_scifact

if __name__ == "__main__":
    documents, queries, qrels = load_scifact()
    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgements: {len(qrels)}")
