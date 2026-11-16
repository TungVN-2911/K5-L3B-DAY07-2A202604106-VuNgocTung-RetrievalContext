import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

from baseline import TfidfIndex, tokenize
from ingest import DEFAULT_STRATEGY, to_record
from store import STARTER

CHUNKS = STARTER / "data" / "chunks.json"
FILTER_KEYS = {"chapter", "document_id", "page"}
DEFAULT_RERANKER = "Xenova/ms-marco-MiniLM-L-12-v2"


def check_request(query: str, top_k: int, filters: dict[str, Any] | None) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if top_k < 1:
        raise ValueError("top_k phải >= 1")
    unknown = set(filters or {}) - FILTER_KEYS
    if unknown:
        raise ValueError(f"filter không được hỗ trợ: {sorted(unknown)}")


def matches(chunk: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    for key, value in (filters or {}).items():
        actual = {"page": chunk["page"], "document_id": chunk["source_id"]}.get(key, chunk["metadata"].get(key))
        if actual != value:
            return False
    return True


def ranked(results: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [{**r, "rank": i, "strategy": name} for i, r in enumerate(results, 1)]


def load_chunks(path: Path = CHUNKS) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"Thiếu {path}: chạy python starter/ingest.py")
    return json.loads(path.read_text(encoding="utf-8"))


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
        check_request(query, top_k, filters)
        rows = self.store.search(self.strategy, self.embedder.embed_query(query), top_k, filters)
        return ranked([to_record(r) for r in rows], self.name)

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "index_strategy": self.strategy,
                "embedding_model": self.embedder.model_name, "dimensions": self.embedder.dimensions,
                "metric": "cosine, score = 1 - distance", "search": "exact (pgvector, no ANN index)"}

    def close(self) -> None:
        if hasattr(self.store, "close"):
            self.store.close()


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25 cần ít nhất một chunk")
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf = [Counter(tokenize(c["text"])) for c in chunks]
        self.length = [sum(tf.values()) for tf in self.tf]
        self.avg_length = sum(self.length) / len(chunks)
        df = Counter(term for tf in self.tf for term in tf)
        n = len(chunks)
        # Lucene/BM25+ style idf: always positive, so very common terms still count a little.
        self.idf = {term: math.log(1 + (n - d + 0.5) / (d + 0.5)) for term, d in df.items()}

    def score(self, terms: list[str], i: int) -> float:
        tf, norm = self.tf[i], self.k1 * (1 - self.b + self.b * self.length[i] / self.avg_length)
        return sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        check_request(query, top_k, filters)
        terms = sorted(set(t for t in tokenize(query) if t in self.idf))
        scored = [(self.score(terms, i), c) for i, c in enumerate(self.chunks) if matches(c, filters)]
        scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]["id"]))[:top_k]
        return ranked([{**c, "score": s} for s, c in scored], self.name)

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "k1": self.k1, "b": self.b, "chunks": len(self.chunks),
                "tokenizer": "regex \\w+, NFC, casefold, không stopword/stemming"}


class TfidfRetriever:
    name = "tfidf"

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        check_request(query, top_k, filters)
        hits = [r for r in self.index.search(query, top_k=len(self.index.chunks)) if matches(r, filters)]
        return ranked([{k: v for k, v in r.items() if k != "strategy"} for r in hits[:top_k]], self.name)

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "source": "starter/baseline.py TfidfIndex"}


class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list, *, candidates: int = 20, rrf_k: int = 60) -> None:
        if len(retrievers) < 2 or candidates < 1 or rrf_k < 0:
            raise ValueError("hybrid cần >= 2 retriever, candidates >= 1, rrf_k >= 0")
        self.retrievers, self.candidates, self.rrf_k = retrievers, candidates, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        check_request(query, top_k, filters)
        fused: dict[str, dict[str, Any]] = {}
        for retriever in self.retrievers:
            for r in retriever.search(query, filters=filters, top_k=max(self.candidates, top_k)):
                item = fused.setdefault(r["id"], {**{k: v for k, v in r.items() if k not in ("score", "rank", "strategy")},
                                                  "score": 0.0, "components": {}})
                item["score"] += 1 / (self.rrf_k + r["rank"])
                item["components"][retriever.name] = {"rank": r["rank"], "score": r["score"]}
        results = sorted(fused.values(), key=lambda r: (-r["score"], r["id"]))[:top_k]
        return ranked(results, self.name)

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "fusion": f"RRF, score = sum 1/({self.rrf_k} + rank)",
                "candidates_per_retriever": self.candidates, "components": [r.info() for r in self.retrievers]}


class CrossEncoderScorer:
    def __init__(self, model: str = DEFAULT_RERANKER) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.model_name = model
        self.model = TextCrossEncoder(model_name=model)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.rerank(query, texts)]


class RerankRetriever:
    name = "rerank"

    def __init__(self, first_stage, *, scorer=None, candidates: int = 20) -> None:
        if candidates < 1:
            raise ValueError("candidates phải >= 1")
        self.first_stage, self.candidates = first_stage, candidates
        self.scorer = scorer or CrossEncoderScorer()

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        check_request(query, top_k, filters)
        pool = self.first_stage.search(query, filters=filters, top_k=max(self.candidates, top_k))
        if not pool:
            return []
        scores = self.scorer(query, [c["text"] for c in pool])
        if len(scores) != len(pool):
            raise RuntimeError(f"reranker trả {len(scores)} điểm cho {len(pool)} candidate")
        rescored = [{**{k: v for k, v in c.items() if k not in ("score", "rank", "strategy")}, "score": s,
                     "first_stage": {"strategy": c["strategy"], "rank": c["rank"], "score": c["score"]}}
                    for c, s in zip(pool, scores)]
        return ranked(sorted(rescored, key=lambda r: (-r["score"], r["id"]))[:top_k], self.name)

    def info(self) -> dict[str, Any]:
        return {"retriever": self.name, "reranker": getattr(self.scorer, "model_name", "custom"),
                "candidates": self.candidates, "first_stage": self.first_stage.info()}

    def close(self) -> None:
        if hasattr(self.first_stage, "close"):
            self.first_stage.close()


METHODS = ("dense", "bm25", "tfidf", "hybrid", "rerank")


class Retrievers:
    def __init__(self, chunks: list[dict[str, Any]] | None = None, *, candidates: int = 20,
                 rrf_k: int = 60, reranker: str = DEFAULT_RERANKER) -> None:
        self._chunks, self.candidates, self.rrf_k, self.reranker = chunks, candidates, rrf_k, reranker
        self._cache: dict[str, Any] = {}

    def chunks(self) -> list[dict[str, Any]]:
        if self._chunks is None:
            self._chunks = load_chunks()
        return self._chunks

    def get(self, method: str):
        if method in self._cache:
            return self._cache[method]
        if method == "dense":
            r = DenseRetriever()
        elif method == "bm25":
            r = BM25Retriever(self.chunks())
        elif method == "tfidf":
            r = TfidfRetriever(self.chunks())
        elif method == "hybrid":
            r = HybridRetriever([self.get("dense"), self.get("bm25")], candidates=self.candidates, rrf_k=self.rrf_k)
        elif method == "rerank":
            r = RerankRetriever(self.get("hybrid"), scorer=CrossEncoderScorer(self.reranker), candidates=self.candidates)
        else:
            raise ValueError(f"method phải thuộc {METHODS}")
        self._cache[method] = r
        return r

    def close(self) -> None:
        if "dense" in self._cache:
            self._cache["dense"].close()


def main():
    parser = argparse.ArgumentParser(description="Chạy thử một retriever")
    parser.add_argument("query", nargs="?", default="How does the book formally define a large language model?")
    parser.add_argument("--method", choices=METHODS, default="dense")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chapter")
    args = parser.parse_args()
    retrievers = Retrievers()
    try:
        r = retrievers.get(args.method)
        t = time.perf_counter()
        results = r.search(args.query, filters={"chapter": args.chapter} if args.chapter else None, top_k=args.top_k)
        print(json.dumps({**r.info(), "query": args.query, "latency_ms": round((time.perf_counter() - t) * 1000, 1)},
                         ensure_ascii=False, indent=2))
        for c in results:
            print(f"#{c['rank']} {c['score']:.4f} p{c['page']} [{c['id']}] {c['text'][:110]}...")
    finally:
        retrievers.close()


if __name__ == "__main__":
    main()
