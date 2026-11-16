import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter"

TABLE = "chunks"
# Only these keys may be used as filters; mapped to fixed column names (never built from input).
FILTER_COLUMNS = {"chapter": "chapter", "document_id": "document_id", "page": "page"}

# nomic-embed-text requires task prefixes (model card) for documents and queries.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


def setting(name: str, default: str | None = None) -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv(STARTER / ".env", override=False)
    except ImportError:
        pass
    value = os.getenv(name, default)
    if not value:
        raise RuntimeError(f"thiếu biến môi trường {name}: chép starter/.env.example thành starter/.env")
    return value


class OllamaEmbedder:
    def __init__(self, model_name: str | None = None, host: str | None = None, batch_size: int = 32) -> None:
        self.model_name = model_name or setting("OLLAMA_EMBED_MODEL")
        self.host = (host or setting("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.batch_size = batch_size
        # Warm-up call: loads the model (cold start) and measures the real dimension.
        self.dimensions = len(self._embed([QUERY_PREFIX + "warm-up"])[0])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed([DOCUMENT_PREFIX + t for t in texts[i:i + self.batch_size]]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._embed([QUERY_PREFIX + question])[0]

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model_name, "input": inputs}).encode("utf-8")
        request = urllib.request.Request(f"{self.host}/api/embed", data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                payload = json.load(response)
        except (urllib.error.URLError, TimeoutError) as error:
            # No fallback to fake vectors: stop and report.
            raise RuntimeError(f"Ollama không phản hồi tại {self.host}: {error}") from error
        vectors = payload.get("embeddings")
        if not vectors or len(vectors) != len(inputs):
            raise RuntimeError(f"Ollama trả {len(vectors or [])} vector cho {len(inputs)} input")
        expected = getattr(self, "dimensions", len(vectors[0]))
        if any(len(v) != expected for v in vectors):
            raise RuntimeError(f"số chiều vector không đồng nhất, mong đợi {expected}")
        return vectors


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _row(r) -> dict[str, Any]:
    return {"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3],
            "metadata": {"chapter": r[4], "printed_page": r[5], "char_start": r[6], "char_end": r[7]},
            "text": r[8]}


class PgVectorStore:
    def __init__(self, dimensions: int, database_url: str | None = None) -> None:
        import psycopg
        if dimensions <= 0:
            raise ValueError(f"số chiều không hợp lệ: {dimensions}")
        self.dimensions = dimensions
        self.conn = psycopg.connect(database_url or setting("DATABASE_URL"), connect_timeout=5)

    def close(self) -> None:
        self.conn.close()

    def stored_dimensions(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT atttypmod FROM pg_attribute WHERE attrelid = %s::regclass AND attname = 'embedding'", (TABLE,))
            return cur.fetchone()[0]

    def versions(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version'), extversion FROM pg_extension WHERE extname = 'vector'")
            pg, vec = cur.fetchone()
        return {"postgresql": pg, "pgvector": vec}

    def create_schema(self) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy      text    NOT NULL,
                    chunk_id      text    NOT NULL,
                    document_id   text    NOT NULL,
                    source        text    NOT NULL,
                    page          integer NOT NULL,
                    chapter       text    NOT NULL,
                    printed_page  integer,
                    char_start    integer NOT NULL,
                    char_end      integer NOT NULL,
                    text          text    NOT NULL,
                    embedding     vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_strategy_chapter ON {TABLE} (strategy, chapter)")
        if self.stored_dimensions() != self.dimensions:
            raise RuntimeError(f"bảng {TABLE} dùng vector({self.stored_dimensions()}), model trả {self.dimensions} chiều")

    def upsert(self, strategy: str, chunks: list, vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk nhưng {len(vectors)} vector")
        if any(len(v) != self.dimensions for v in vectors):
            raise ValueError(f"có vector không đủ {self.dimensions} chiều")
        ids = [c.chunk_id for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk_id bị trùng trong lô upsert")
        documents = sorted({c.document_id for c in chunks})
        rows = [(strategy, c.chunk_id, c.document_id, c.source, c.page, c.metadata["chapter"],
                 c.metadata.get("printed_page"), c.metadata["char_start"], c.metadata["char_end"],
                 c.text, _vector_literal(v)) for c, v in zip(chunks, vectors)]
        with self.conn.transaction(), self.conn.cursor() as cur:
            # Remove stale chunks of the same document that are no longer produced.
            cur.execute(f"DELETE FROM {TABLE} WHERE strategy = %s AND document_id = ANY(%s) AND NOT (chunk_id = ANY(%s))",
                        (strategy, documents, ids))
            cur.executemany(
                f"""INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, chapter,
                        printed_page, char_start, char_end, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                        document_id = EXCLUDED.document_id, source = EXCLUDED.source,
                        page = EXCLUDED.page, chapter = EXCLUDED.chapter,
                        printed_page = EXCLUDED.printed_page, char_start = EXCLUDED.char_start,
                        char_end = EXCLUDED.char_end, text = EXCLUDED.text, embedding = EXCLUDED.embedding""",
                rows)

    def count(self, strategy: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {TABLE} WHERE strategy = %s", (strategy,))
            return cur.fetchone()[0]

    def fetch_all(self, strategy: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(f"""SELECT chunk_id, document_id, source, page, chapter, printed_page, char_start, char_end, text
                            FROM {TABLE} WHERE strategy = %s ORDER BY page, char_start, chunk_id""", (strategy,))
            return [_row(r) for r in cur.fetchall()]

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(f"query vector có {len(query_vector)} chiều, cần {self.dimensions}")
        where, params = ["strategy = %s"], [strategy]
        for key, value in (filters or {}).items():
            if key not in FILTER_COLUMNS:
                raise ValueError(f"filter không được hỗ trợ: {key}")
            where.append(f"{FILTER_COLUMNS[key]} = %s")
            params.append(value)
        sql = f"""SELECT chunk_id, document_id, source, page, chapter, printed_page, char_start, char_end,
                         text, embedding <=> %s::vector AS distance
                  FROM {TABLE} WHERE {' AND '.join(where)} ORDER BY distance LIMIT %s"""
        with self.conn.cursor() as cur:
            cur.execute(sql, [_vector_literal(query_vector), *params, top_k])
            return [{**_row(r), "score": 1 - float(r[9])} for r in cur.fetchall()]
