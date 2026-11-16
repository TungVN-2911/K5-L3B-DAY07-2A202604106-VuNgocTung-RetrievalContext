import argparse
import json
import re
import statistics
import time
import unicodedata
from collections import Counter
from pathlib import Path

from baseline import hit, pack_context, validate
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


def run_check(args) -> None:
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


def first_hit(results: list[dict], gold: list[dict]) -> int | None:
    return next((r["rank"] for r in results if hit(r, gold)), None)


def evaluate_retriever(retriever, questions: list[dict], *, use_filter: bool, depth: int, context_k: int,
                       budget: int) -> tuple[list[dict], dict]:
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=depth)
        elapsed_ms = (time.perf_counter() - start) * 1000
        context, ids = pack_context(results[:context_k], budget=budget)
        rank = first_hit(results, q["gold"])
        traces.append({"query_id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                       "filter": filters, "results": [{**r, "hit": hit(r, q["gold"])} for r in results],
                       "context_ids": ids, "context_chars": len(context), "elapsed_ms": round(elapsed_ms, 1),
                       "first_hit_rank": rank, "hit1": rank == 1, "hit3": rank is not None and rank <= 3,
                       "context_hit": any(hit(r, q["gold"]) for r in results if r["id"] in ids)})
    n = len(traces)

    def rate(key):
        k = sum(t[key] for t in traces)
        return {"n": k, "N": n, "rate": round(k / n, 3)}

    summary = {"hit1": rate("hit1"), "hit3": rate("hit3"), "context_evidence": rate("context_hit"),
               f"mrr@{depth}": round(sum(1 / t["first_hit_rank"] for t in traces if t["first_hit_rank"]) / n, 3),
               f"recall@{depth}": {"n": sum(t["first_hit_rank"] is not None for t in traces), "N": n},
               "median_ms": round(statistics.median(t["elapsed_ms"] for t in traces), 1),
               "mean_context_chars": round(statistics.mean(t["context_chars"] for t in traces)), "cost_usd": 0}
    return traces, summary


def run_retrieval(args) -> None:
    from retrievers import Retrievers
    questions = answerable(load_questions(args.questions), args.split)
    retrievers = Retrievers(candidates=args.candidates, rrf_k=args.rrf_k, reranker=args.reranker)
    out = args.output / args.split
    out.mkdir(parents=True, exist_ok=True)
    config = {"split": args.split, "questions": len(questions), "use_filter": args.use_filter, "depth": args.depth,
              "context": f"top-{args.context_k}, budget {args.budget} ký tự (gồm header nguồn), pack_context của baseline",
              "candidates": args.candidates, "rrf_k": args.rrf_k}
    report = {"config": config, "methods": {}}
    try:
        for method in args.methods:
            start = time.perf_counter()
            retriever = retrievers.get(method)
            retriever.search("warm-up query", top_k=1)
            load_s = round(time.perf_counter() - start, 2)
            traces, summary = evaluate_retriever(retriever, questions, use_filter=args.use_filter, depth=args.depth,
                                                 context_k=args.context_k, budget=args.budget)
            (out / f"{method}.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            report["methods"][method] = {"info": retriever.info(), "load_s": load_s, **summary}
    finally:
        retrievers.close()
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"| Method | Hit@1 | Evidence Hit@3 | Context Evidence | MRR@{args.depth} | median ms | load s |")
    print("|---|---|---|---|---|---|---|")
    for method, m in report["methods"].items():
        f = lambda x: f"{x['n']}/{x['N']} ({x['rate']:.0%})"
        print(f"| {method} | {f(m['hit1'])} | {f(m['hit3'])} | {f(m['context_evidence'])} | "
              f"{m[f'mrr@{args.depth}']:.3f} | {m['median_ms']} | {m['load_s']} |")


def main():
    parser = argparse.ArgumentParser(description="Benchmark: kiểm tra bộ câu hỏi và đánh giá retrieval")
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="schema, số câu, gold trong chunk, câu thiếu evidence")
    check.add_argument("--chunks", type=Path, default=STARTER / "data" / "chunks.json")
    check.add_argument("--pdf", type=Path, default=ROOT / "data" / "AI Engineering Guidebook.pdf")
    retrieval = sub.add_parser("retrieval", help="Hit@1, Evidence Hit@3, Context Evidence Rate theo phương pháp")
    retrieval.add_argument("--split", choices=("dev", "test"), required=True)
    retrieval.add_argument("--methods", nargs="+", default=["dense", "bm25", "hybrid", "rerank"])
    retrieval.add_argument("--use-filter", action="store_true", help="dùng filter chapter người dùng nêu trong câu hỏi")
    retrieval.add_argument("--depth", type=int, default=10)
    retrieval.add_argument("--context-k", type=int, default=3)
    retrieval.add_argument("--budget", type=int, default=2000)
    retrieval.add_argument("--candidates", type=int, default=20)
    retrieval.add_argument("--rrf-k", type=int, default=60)
    retrieval.add_argument("--reranker", default="Xenova/ms-marco-MiniLM-L-12-v2")
    retrieval.add_argument("--output", type=Path, default=STARTER / "results" / "retrieval")
    args = parser.parse_args()
    run_check(args) if args.command == "check" else run_retrieval(args)


if __name__ == "__main__":
    main()
