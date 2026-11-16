import json
import sys
import time
from typing import Any

from ingest import DEFAULT_STRATEGY, to_record


class DenseRetriever:
    name = "dense"

    def __init__(self, strategy: str = DEFAULT_STRATEGY, *, embedder=None, store=None) -> None:
        if embedder is None or store is None:
            from store import OllamaEmbedder, PgVectorStore
            embedder = embedder or OllamaEmbedder()
            store = store or PgVectorStore(embedder.dimensions)
        if store.dimensions != embedder.dimensions:
            raise RuntimeError(f"store {store.dimensions} chiều, embedder {embedder.dimensions} chiều")
        self.embedder, self.store, self.strategy = embedder, store, strategy

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query phải là chuỗi không rỗng")
        if top_k < 1:
            raise ValueError("top_k phải >= 1")
        rows = self.store.search(self.strategy, self.embedder.embed_query(query), top_k, filters)
        return [{**to_record(r), "rank": i, "strategy": self.name} for i, r in enumerate(rows, 1)]

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "index_strategy": self.strategy,
                "embedding_model": self.embedder.model_name, "dimensions": self.embedder.dimensions,
                "metric": "cosine, score = 1 - distance", "search": "exact (pgvector, no ANN index)"}

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()


def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "How does the book formally define a large language model?"
    retriever = DenseRetriever()
    try:
        count = retriever.store.count(retriever.strategy)
        if count == 0:
            raise SystemExit("Index rỗng: chạy python starter/ingest.py --reindex")
        t = time.perf_counter()
        results = retriever.search(query, top_k=3)
        ms = (time.perf_counter() - t) * 1000
        print(json.dumps({**retriever.info(), **retriever.store.versions(), "chunks": count,
                          "query": query, "latency_ms": round(ms, 1)}, indent=2))
        for r in results:
            print(f"#{r['rank']} {r['score']:.3f} p{r['page']} [{r['id']}] {r['text'][:120]}...")
    finally:
        retriever.close()


if __name__ == "__main__":
    main()
