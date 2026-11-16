import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

from baseline import hit, pack_context, validate
from store import ROOT, STARTER

QUESTIONS = ROOT / "data" / "questions.json"
# Corpus has no open license: committed traces keep only an excerpt; full text is rebuilt by ingest.py.
EXCERPT_CHARS = 160
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


def excerpt(chunk: dict, full_text: bool = False) -> dict:
    if full_text or len(chunk["text"]) <= EXCERPT_CHARS:
        return chunk
    return {**chunk, "text": chunk["text"][:EXCERPT_CHARS] + "…", "text_chars": len(chunk["text"])}


def first_hit(results: list[dict], gold: list[dict]) -> int | None:
    return next((r["rank"] for r in results if hit(r, gold)), None)


def evaluate_retriever(retriever, questions: list[dict], *, use_filter: bool, depth: int, context_k: int,
                       budget: int, full_text: bool = False) -> tuple[list[dict], dict]:
    traces = []
    for q in questions:
        filters = q.get("filter") if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q["question"], filters=filters, top_k=depth)
        elapsed_ms = (time.perf_counter() - start) * 1000
        context, ids = pack_context(results[:context_k], budget=budget)
        rank = first_hit(results, q["gold"])
        traces.append({"query_id": q["id"], "split": q["split"], "type": q["type"], "question": q["question"],
                       "filter": filters, "results": [{**excerpt(r, full_text), "hit": hit(r, q["gold"])} for r in results],
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
                                                 context_k=args.context_k, budget=args.budget, full_text=args.full_text)
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


def labeled(review: dict) -> bool:
    keys = ["citation_support"] + [k for k in ("answer_grade", "abstention_correct") if k in review]
    return all(review[k] is not None for k in keys) and None not in review["citation_support"]


def evaluate_agent_run(run: Path, questions: list[dict], chunks: dict[str, dict]) -> dict:
    from agent_metrics import aggregate
    traces = json.loads((run / "traces.json").read_text(encoding="utf-8"))
    by_q = {q["id"]: q for q in questions}
    ok = [t for t in traces if t["status"] != "error"]
    gold_q = [t for t in ok if by_q[t["query_id"]]["answerable"]]

    def gold_hit(ids, q):
        return any(i in chunks and hit(chunks[i], q["gold"]) for i in ids)

    out = {"run": run.name, "split": traces[0]["split"], "questions": len(traces),
           "status": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
           "tool_use": {"n": sum(t["searches"] > 0 for t in ok), "N": len(ok)},
           "searches": dict(sorted(Counter(str(t["searches"]) for t in ok).items())),
           "llm_calls_mean": round(statistics.mean(t["llm_calls"] for t in ok), 2) if ok else None,
           "context_evidence": {"n": sum(gold_hit(t["context_ids"], by_q[t["query_id"]]) for t in gold_q), "N": len(gold_q)},
           "citation_has_gold_auto": {"n": sum(gold_hit(t["citations"], by_q[t["query_id"]]) for t in gold_q
                                               if t["status"] == "answered"),
                                      "N": sum(t["status"] == "answered" for t in gold_q)},
           "tokens": {k: sum(t["usage"][k] for t in traces) for k in ("input", "output", "thinking")},
           "median_elapsed_s": round(statistics.median(t["elapsed_s"] for t in traces), 1)}
    reviews_path = run / "reviews.json"
    reviews = json.loads(reviews_path.read_text(encoding="utf-8")) if reviews_path.exists() else []
    pending = [r["query_id"] for r in reviews if not labeled(r)]
    if reviews and not pending:
        cases = [{"id": q["id"], "answerable": q["answerable"]} for q in questions if q["id"] in {t["query_id"] for t in traces}]
        out["human_labels"] = aggregate(cases, traces, reviews)
    else:
        out["human_labels"] = f"chưa chấm: {len(pending) if reviews else 'không có reviews.json'}"
    return out


def run_agent_eval(args) -> None:
    questions = load_questions(args.questions)
    chunks = {c["id"]: c for c in json.loads(args.chunks.read_text(encoding="utf-8"))}
    reports = [evaluate_agent_run(run, questions, chunks) for run in args.runs]
    (args.output).parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    f = lambda x: f"{x['n']}/{x['N']}"
    print("| Run | answered/abstain/error | Tool use | Searches | Context Evidence | Citation có gold (auto) | Grounded 0/1/2 (mean) | Citation hợp lệ | Abstention đúng | Từ chối nhầm | tokens in/out | median s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in reports:
        h = r["human_labels"]
        if isinstance(h, dict):
            g = h["grounded_answer"]
            human = (f"{g['0']}/{g['1']}/{g['2']} ({g['mean']:.2f})", f(h["citation_validity"]), f(h["abstention"]), f(h["false_abstention"]))
        else:
            human = (h, "", "", "")
        s = r["status"]
        print(f"| {r['run']} | {s['answered']}/{s['abstain']}/{s['error']} | {f(r['tool_use'])} | {r['searches']} | "
              f"{f(r['context_evidence'])} | {f(r['citation_has_gold_auto'])} | {' | '.join(human)} | "
              f"{r['tokens']['input']}/{r['tokens']['output']} | {r['median_elapsed_s']} |")


def run_review(args) -> None:
    if args.output and not args.confirm:
        sys.stdout = args.output.open("w", encoding="utf-8")
    questions = {q["id"]: q for q in load_questions(args.questions)}
    chunks = {c["id"]: c for c in json.loads(args.chunks.read_text(encoding="utf-8"))}
    for run in args.runs:
        path = run / "reviews.json"
        reviews = json.loads(path.read_text(encoding="utf-8"))
        if args.confirm:
            for r in reviews:
                if not labeled(r):
                    raise SystemExit(f"{run.name}/{r['query_id']}: còn nhãn trống, chưa thể xác nhận")
                r["labeled_by"] = args.confirm
            path.write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{run.name}: đã ghi labeled_by={args.confirm!r} cho {len(reviews)} nhãn")
            continue
        traces = {t["query_id"]: t for t in json.loads((run / "traces.json").read_text(encoding="utf-8"))}
        for r in reviews:
            q, t = questions[r["query_id"]], traces[r["query_id"]]
            label = r.get("answer_grade", r.get("abstention_correct"))
            print(f"\n{'=' * 100}\n[{run.name}] {q['id']} ({q['type']}) status={t['status']} | NHÃN: {label} | {r['note']}")
            print(f"Q: {q['question']}\nĐÁP ÁN CHUẨN: {q.get('expected_answer', q.get('absence_reason'))}")
            if q["answerable"]:
                print(f"GOLD: " + "; ".join(f"p{g['page']}: {g['evidence']}" for g in q["gold"]))
            print(f"TRẢ LỜI: {t.get('model_answer') or t['answer']}")
            print("SEARCH: " + " | ".join(f"{e['query']!r} filter={e['filter']}" for e in t["events"] if e["event"] == "search"))
            for cid, ok in zip(r["citations"], r["citation_support"]):
                c = chunks.get(cid, {"page": "?", "text": "(không có trong chunks.json)"})
                print(f"  CITE p{c['page']} hỗ trợ={ok} [{cid}]\n    {c['text']}")
    if args.context_labels:
        labels = json.loads(args.context_labels.read_text(encoding="utf-8"))
        if args.confirm:
            for c in labels:
                c["labeled_by"] = args.confirm
            args.context_labels.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"{args.context_labels.name}: đã ghi labeled_by={args.confirm!r} cho {len(labels)} nhãn")
        else:
            for c in labels:
                print(f"\n[context {c['method']}] {c['query_id']} điểm={c['context_score']} | {c['reason']}")
                for cid in c["context_ids"]:
                    print(f"  {'GOLD ' if cid in c['gold_in_context'] else '     '}[{cid}] {chunks.get(cid, {}).get('text', '?')[:220]}")


def main():
    # Windows consoles default to a legacy code page; force UTF-8 so Vietnamese text survives.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
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
    retrieval.add_argument("--full-text", action="store_true", help="ghi toàn văn chunk vào trace (mặc định chỉ trích đoạn)")
    agent = sub.add_parser("agent", help="Tool use, Context Evidence Rate của agent và tổng hợp nhãn chấm tay")
    agent.add_argument("runs", nargs="+", type=Path, help="thư mục chứa traces.json (+ reviews.json)")
    agent.add_argument("--chunks", type=Path, default=STARTER / "data" / "chunks.json")
    agent.add_argument("--output", type=Path, default=STARTER / "results" / "agent" / "eval.json")
    review = sub.add_parser("review", help="In nhãn nháp kèm toàn văn chunk được cite để người chấm duyệt")
    review.add_argument("runs", nargs="*", type=Path, help="thư mục chứa traces.json + reviews.json")
    review.add_argument("--context-labels", type=Path, help="file context_labels.json của retrieval")
    review.add_argument("--chunks", type=Path, default=STARTER / "data" / "chunks.json")
    review.add_argument("--confirm", help="tên người chấm; chỉ dùng sau khi đã đọc và sửa xong nhãn")
    review.add_argument("--output", type=Path, help="ghi ra file UTF-8 thay vì in ra màn hình")
    args = parser.parse_args()
    {"check": run_check, "retrieval": run_retrieval, "agent": run_agent_eval, "review": run_review}[args.command](args)


if __name__ == "__main__":
    main()
