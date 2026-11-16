import argparse
import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from store import ROOT, STARTER, OllamaEmbedder, PgVectorStore, setting

MANIFEST = ROOT / "data" / "manifest.json"
DOCUMENT_ID = "ai-engineering-2025"
SOURCE = "AI Engineering Guidebook.pdf"
# Strategy A (fixed 500/50) was chosen on dev Hit@3 in Day 6; it is the default index.
DEFAULT_STRATEGY = "A"
STRATEGIES = {"A": ("fixed", 500, 50), "B": ("paragraph", 1000, None)}

# Each page has a 'DailyDoseofDS.com' header and a printed page number (= pdf_page - 1) at the end.
HEADER = re.compile(r"^\s*DailyDoseofDS\.com\s*")
FOOTER_PAGE_NUMBER = re.compile(r"\n\s*\d{1,3}\s*$")
LINE_BREAK_HYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")
# Unit boundary = one space after . ! ? before an uppercase letter, ●, “ or (, or right before ●.
BOUNDARY = re.compile(r"(?<=[.!?])\s(?=[A-Z●“(])|\s(?=●)")


@dataclass
class DocumentPage:
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    source: str
    page: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------- loading ----------

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = HEADER.sub("", text)
    text = FOOTER_PAGE_NUMBER.sub("", text)
    text = LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    return " ".join(text.split())


def load_pdf(path: Path) -> tuple[list[DocumentPage], list[dict]]:
    from pypdf import PdfReader
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))["sources"][0]
    if manifest["source_id"] != DOCUMENT_ID:
        raise ValueError(f"manifest source_id={manifest['source_id']} khác {DOCUMENT_ID}")
    reader = PdfReader(path)
    if len(reader.pages) != manifest["physical_pages"]:
        raise ValueError(f"PDF có {len(reader.pages)} trang, manifest ghi {manifest['physical_pages']}")
    pages, skipped = [], []
    for info in manifest["pages"]:
        n = info["pdf_page"]
        if not info["counted"]:
            skipped.append({"pdf_page": n, "excluded_reason": info["excluded_reason"]})
            continue
        text = normalize_text(reader.pages[n - 1].extract_text() or "")
        if len(text) != info["chars"]:
            raise ValueError(f"Trang {n}: {len(text)} ký tự sau chuẩn hóa, manifest ghi {info['chars']}")
        pages.append(DocumentPage(DOCUMENT_ID, SOURCE, n, text,
                                  {"chapter": info["chapter"], "printed_page": info["printed_page"]}))
    if not pages:
        raise ValueError("Không nạp được trang nào: kiểm tra manifest và file PDF.")
    return pages, skipped


# ---------- chunking ----------

def _make_chunk(page: DocumentPage, kind: str, i: int, start: int, end: int) -> Chunk:
    if not 0 <= start < end <= len(page.text):
        raise ValueError(f"vị trí chunk không hợp lệ: start={start}, end={end}, len={len(page.text)}")
    return Chunk(f"{page.document_id}:p{page.page}:{kind}:{i}", page.document_id, page.source, page.page,
                 page.text[start:end], {**page.metadata, "chunk_index": i, "char_start": start, "char_end": end})


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError(f"cần 0 <= overlap < size, nhận size={size}, overlap={overlap}")
    kind, chunks, start = f"fixed{size}-{overlap}", [], 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        chunks.append(_make_chunk(page, kind, len(chunks), start, end))
        if end == len(page.text):  # no tail chunk containing only the overlap
            break
        start = end - overlap
    return chunks


def _split_units(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    units, start = [], 0
    for m in BOUNDARY.finditer(text):
        units.append((start, m.start()))
        start = m.end()
    units.append((start, len(text)))
    return units


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    if max_size <= 0:
        raise ValueError(f"max_size phải > 0, nhận max_size={max_size}")
    kind, chunks = f"para{max_size}", []
    cur_start = cur_end = None
    for s, e in _split_units(page.text):
        if e - s > max_size:                                  # sentence too long: hard split
            if cur_start is not None:
                chunks.append(_make_chunk(page, kind, len(chunks), cur_start, cur_end))
                cur_start = cur_end = None
            for a in range(s, e, max_size):
                chunks.append(_make_chunk(page, kind, len(chunks), a, min(a + max_size, e)))
        elif cur_start is None or e - cur_start <= max_size:  # still fits: extend
            cur_start = s if cur_start is None else cur_start
            cur_end = e
        else:                                                 # full: close and open new
            chunks.append(_make_chunk(page, kind, len(chunks), cur_start, cur_end))
            cur_start, cur_end = s, e
    if cur_start is not None:
        chunks.append(_make_chunk(page, kind, len(chunks), cur_start, cur_end))
    return chunks


def chunk_pages(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    kind, size, overlap = STRATEGIES[strategy]
    if kind == "fixed":
        return [c for p in pages for c in fixed_size(p, size, overlap)]
    return [c for p in pages for c in by_paragraph(p, size)]


# ---------- export ----------

def to_record(row: dict[str, Any]) -> dict[str, Any]:
    out = {"id": row["chunk_id"], "text": row["text"], "source_id": row["document_id"],
           "source": row["source"], "page": row["page"], "metadata": dict(row["metadata"])}
    if "score" in row:
        out["score"] = float(row["score"])
    return out


def export_chunks(store: PgVectorStore, strategy: str, output: Path) -> dict[str, Any]:
    rows = store.fetch_all(strategy)
    if not rows:
        raise SystemExit(f"Không có chunk nào cho strategy={strategy}; chạy với --reindex.")
    chunks = [to_record(r) for r in rows]
    for c in chunks:
        c["metadata"]["index_strategy"] = strategy
    payload = json.dumps(chunks, ensure_ascii=False, indent=2) + "\n"
    output.mkdir(parents=True, exist_ok=True)
    (output / "chunks.json").write_text(payload, encoding="utf-8")
    meta = {"exported_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "index_strategy": strategy, "chunks": len(chunks), "pages": len({c["page"] for c in chunks}),
            "embedding_dimensions": store.stored_dimensions(),
            "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}
    (output / "chunks_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main():
    parser = argparse.ArgumentParser(description="Nạp PDF vào pgvector và xuất chunks.json")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default=DEFAULT_STRATEGY)
    parser.add_argument("--reindex", action="store_true", help="đọc PDF, embed và upsert lại trước khi xuất")
    parser.add_argument("--output", type=Path, default=STARTER / "data")
    args = parser.parse_args()

    report: dict[str, Any] = {"strategy": args.strategy}
    if args.reindex:
        pdf = Path(setting("PDF_PATH", f"data/{SOURCE}"))
        pdf = pdf if pdf.is_absolute() else ROOT / pdf
        if not pdf.exists():
            raise SystemExit(f"Không thấy PDF tại {pdf}. Tải theo data/manifest.json hoặc đặt PDF_PATH.")
        t = time.perf_counter(); pages, skipped = load_pdf(pdf); report["load_pdf_s"] = round(time.perf_counter() - t, 2)
        chunks = chunk_pages(pages, args.strategy)
        embedder = OllamaEmbedder()
        t = time.perf_counter(); vectors = embedder.embed_documents([c.text for c in chunks]); report["embed_s"] = round(time.perf_counter() - t, 2)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "skipped_pages.json").write_text(json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")
        report.update(pages=len(pages), skipped_pages=len(skipped), embedding_model=embedder.model_name, cost_usd=0)
        store = PgVectorStore(embedder.dimensions)
        try:
            store.create_schema()
            store.upsert(args.strategy, chunks, vectors)
        finally:
            store.close()
    store = PgVectorStore(dimensions=1)  # read-only: dimension is read from the table
    try:
        report["export"] = export_chunks(store, args.strategy, args.output)
    finally:
        store.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
