import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from baseline import validate
from store import ROOT, STARTER

QUESTIONS = ROOT / "data" / "questions.json"
MINIMUM = {"dev": {"answerable": 15, "unanswerable": 2}, "test": {"answerable": 5, "unanswerable": 2}}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def answerable(questions: list[dict], split: str | None = None) -> list[dict]:
    return [q for q in questions if q["answerable"] and (split is None or q["split"] == split)]


def check_schema(questions: list[dict]) -> dict:
    ids = [q.get("id") for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError(f"ID trùng: {[i for i, n in Counter(ids).items() if n > 1]}")
    counts = {s: {"answerable": 0, "unanswerable": 0} for s in MINIMUM}
    for q in questions:
        qid = q.get("id")
        if q.get("split") not in MINIMUM or not isinstance(q.get("question"), str) or not q["question"].strip():
            raise ValueError(f"{qid}: split hoặc question không hợp lệ")
        if not isinstance(q.get("answerable"), bool):
            raise ValueError(f"{qid}: thiếu answerable boolean")
        if q["answerable"]:
            if q.get("expected_behavior") != "search_and_answer" or not q.get("gold") or not q.get("expected_answer"):
                raise ValueError(f"{qid}: câu có đáp án cần expected_answer, gold, expected_behavior=search_and_answer")
        else:
            check = q.get("absence_check") or {}
            if q.get("expected_behavior") != "search_and_abstain" or not q.get("absence_reason") or not check.get("absent_patterns"):
                raise ValueError(f"{qid}: câu thiếu evidence cần absence_reason, absence_check.absent_patterns, search_and_abstain")
            if "gold" in q or "expected_answer" in q:
                raise ValueError(f"{qid}: câu thiếu evidence không được có gold/expected_answer")
        counts[q["split"]]["answerable" if q["answerable"] else "unanswerable"] += 1
    for split, need in MINIMUM.items():
        for kind, n in need.items():
            if counts[split][kind] < n:
                raise ValueError(f"{split}: cần ít nhất {n} câu {kind}, hiện có {counts[split][kind]}")
    return counts


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def pdf_pages(path: Path) -> list[tuple[int, str]]:
    from pypdf import PdfReader
    return [(i, normalize(p.extract_text() or "")) for i, p in enumerate(PdfReader(path).pages, 1)]


def check_absence(questions: list[dict], pages: list[tuple[int, str]]) -> dict[str, list[str]]:
    found = {}
    for q in questions:
        if q["answerable"]:
            continue
        hits = [f"{pat!r} p{n}" for pat in q["absence_check"]["absent_patterns"]
                for n, text in pages if re.search(pat, text, re.IGNORECASE)]
        if hits:
            found[q["id"]] = hits
    return found


def main():
    parser = argparse.ArgumentParser(description="Kiểm tra benchmark: schema, số câu, gold trong chunk, câu thiếu evidence")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--chunks", type=Path, default=STARTER / "data" / "chunks.json")
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "AI Engineering Guidebook.pdf")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    counts = check_schema(questions)
    chunks = json.loads(args.chunks.read_text(encoding="utf-8"))
    validate(chunks, answerable(questions))
    pages = pdf_pages(args.pdf) if args.pdf.exists() else [(c["page"], normalize(c["text"])) for c in chunks]
    found = check_absence(questions, pages)
    if found:
        raise SystemExit(f"Câu thiếu evidence khớp pattern trong tài liệu: {json.dumps(found, ensure_ascii=False)}")
    types = Counter((q["split"], q["type"]) for q in questions)
    print(json.dumps({"counts": counts,
                      "types": {f"{s}/{t}": n for (s, t), n in sorted(types.items())},
                      "gold_in_chunk": f"{len(answerable(questions))}/{len(answerable(questions))}",
                      "absence_scope": f"{len(pages)} trang ({'PDF' if args.pdf.exists() else 'chunks.json'})"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
