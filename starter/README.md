# Starter Day 7

| File | Công dụng |
|---|---|
| `agentic.py` | Khung agent ví dụ với số lượt search, top-k và context budget có thể cấu hình |
| `baseline.py` | TF-IDF lexical baseline, chạy offline |
| `agent_metrics.py` | Tổng hợp nhãn đánh giá do bạn tự chấm |
| `store.py` | Đọc `.env`, embedder Ollama `nomic-embed-text`, pgvector store (dùng lại index Day 6) |
| `ingest.py` | Đọc PDF, chunking (fixed / paragraph), embed + upsert, xuất `starter/data/chunks.json` |
| `benchmark.py` | `check`: kiểm tra `data/questions.json`; `retrieval`: Hit@1, Evidence Hit@3, Context Evidence Rate, MRR theo phương pháp |
| `retrievers.py` | `dense` (pgvector), `bm25`, `tfidf` (bọc baseline), `hybrid` (RRF dense + BM25), `rerank` (cross-encoder trên candidate hybrid); chạy trực tiếp để thử một query |
| `agent.py` | Agent Gemini (REST, function calling): tool `search` / `answer` / `abstain`, citation `[C#]` map về chunk ID, trace từng câu |

## 0. Kết nối dữ liệu và vector store Day 6

Corpus: *AI Engineering Guidebook* (2025), 343/384 trang, strategy A = fixed 500/50, 621 chunk; `nomic-embed-text` 768 chiều, cosine, exact search. Nguồn, quyền dùng và sha256 trong [data/manifest.json](../data/manifest.json); benchmark gốc Day 6 ở `data/source_questions.jsonl`. PDF không commit, đặt tại `data/AI Engineering Guidebook.pdf`.

```bash
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
cp starter/.env.example starter/.env
cd starter && docker compose -p day6_student_starter up -d --wait && cd ..   # dùng lại volume Day 6
ollama pull nomic-embed-text
.venv/Scripts/python starter/ingest.py              # xuất chunks.json từ index có sẵn
.venv/Scripts/python starter/ingest.py --reindex    # khi volume rỗng: đọc PDF + embed lại (~80 s, 0 USD)
.venv/Scripts/python starter/retrievers.py --method rerank "What is RAG?"   # thử một retriever
.venv/Scripts/python starter/benchmark.py check                              # kiểm tra benchmark
.venv/Scripts/python starter/benchmark.py retrieval --split dev --methods tfidf bm25 dense hybrid rerank
```

### Agent (Gemini API)

Đặt `GEMINI_API_KEY` trong `starter/.env` (không commit). Model mặc định `gemini-3.5-flash-lite`, temperature 0; đổi bằng `GEMINI_MODEL`.

```bash
.venv/Scripts/python starter/agent.py --question "What is RAG?"                 # một câu
.venv/Scripts/python starter/agent.py --split dev --method rerank               # cả split -> starter/results/agent/dev-rerank/
.venv/Scripts/python starter/agent.py --split dev --method dense                # baseline dense
```

| Thành phần | Thiết kế |
|---|---|
| Tool | `search(query, chapter?)`, `answer(answer, citations)`, `abstain(reason)`; `mode: ANY` nên mỗi lượt model bắt buộc gọi một tool |
| Filter | Model chỉ đặt `chapter` khi câu hỏi nêu tên chương (giá trị nằm trong enum lấy từ metadata); `--use-filter` dùng filter người dùng ghi sẵn |
| Context | Mỗi search lấy top-4; chunk mới được gán mã `C#`, chunk đã thấy chỉ nhắc lại mã; tổng chunk mới ≤ 6000 ký tự |
| Dừng | Model gọi `answer`/`abstain`; hết 3 lượt search thì chỉ còn `answer`/`abstain` |
| Citation | Mã phải có trong context và xuất hiện trong text (`[C1]` hoặc `[C1, C2]`); sai thì phản hồi cho sửa 1 lần, sai tiếp thì abstain |
| Lỗi | HTTP 429/5xx retry có backoff (đọc `retryDelay`); hết retry thì `status: "error"`, không thay bằng kết quả giả |
| Output | `traces.json` (tool call, kết quả, context, câu trả lời, citation, token, thời gian), `summary.json`, `reviews.json` (mẫu nhãn để chấm tay, không ghi đè) |

Starter **chưa phải solution hoàn chỉnh**. Bạn phải nối model và vector store thật, triển khai ít nhất 3 phương pháp retrieval do bạn chọn, gồm dense trên vector store Day 6. Mock chỉ dùng để test code.

## 1. Kết nối agent

> Bài làm dùng [agent.py](agent.py) (Gemini, nhiều tool) thay cho khung `agentic.py`; phần dưới là hướng dẫn gốc của starter.

Trong `agentic.py`, hiện thực:

- `Controller.decide`: gọi model thật và trả `{action: "search", query: "..."}` hoặc `{action: "answer", answer: "... [chunk-id]", citations: ["chunk-id"]}`.
- `Retriever.search`: tạo query embedding tương thích Day 6, truy vấn vector store thật và trả chunk theo [định dạng dữ liệu](../data/README.md).

Tự viết chương trình nối hai phần với `run_agent`, hoặc thay thế bằng kiến trúc nhiều tool. Tham số `max_searches`, `top_k` và `budget` chỉ là mặc định của ví dụ, không phải giới hạn của đề. Xem `tests/test_agentic.py` để hiểu cách gọi; kết nối trong test là mock. Khung code kiểm tra citation ID có trong context, nhưng bạn phải tự đọc xem nguồn có hỗ trợ câu trả lời. Báo lỗi kết nối, không thay bằng kết quả giả.

## 2. Chạy TF-IDF baseline

Chạy từ thư mục gốc với Python 3.11 trở lên:

```bash
python3 -m unittest discover -s tests -v
python3 starter/baseline.py --chunks data/example_chunks.json --questions data/example_questions.json --split dev --output starter/results/example-dev
```

Kết quả có `summary.json` và `traces.json`. Dữ liệu mẫu chỉ có một câu, không đủ để nộp bài. TF-IDF là lexical retrieval, không dùng filter và không thay dense retrieval.

## 3. Dữ liệu thật

Chunk lấy thẳng từ pgvector bằng `ingest.py` (mục 0), ra `starter/data/chunks.json` theo schema Day 7. Benchmark ở [data/questions.json](../data/questions.json), kiểm tra bằng `benchmark.py`; xem [data/README.md](../data/README.md).

`--split dev` chỉ chấm dev. Sau khi chốt cấu hình, dùng `--split test` và chọn thư mục kết quả khác. Nếu test Day 6 từng dùng để chỉnh hệ thống, tạo test mới theo [GUIDE.md](../GUIDE.md). Baseline mẫu dùng evidence nằm trong một chunk và context từ top-3; cần mở rộng hoặc tự đánh giá nếu dùng multi-hop, context từ nhiều lượt hay schema khác. Khớp nguyên văn bằng code không thay thế đánh giá thủ công.

## 4. Tổng hợp đánh giá agent

Sau khi chạy agent thật và tự chấm, chuẩn bị ba JSON array:

- `cases.json`: mỗi câu có `id`, `split`, `answerable` (`true` nếu có đủ evidence).
- `traces.json`: mỗi kết quả có `query_id`, `status` (`answered`, `abstain`, `error`), `events`, `citations`. Thêm `query_id` vào kết quả `run_agent`; lỗi kết nối ghi riêng với `status: "error"`. Script đếm tool use qua event `search`: nếu dùng nhiều tool, chuẩn hóa mỗi lượt retrieval về event này và ghi tên tool trong trường khác, hoặc tự viết metric tương đương.
- `reviews.json`: mỗi nhãn có `query_id`, `citation_support` là danh sách đúng/sai ứng với từng citation. Câu có đáp án cần `answer_grade` 0–2 theo [BENCHMARK.md](../BENCHMARK.md); nếu bị từ chối, chấm 0. Câu thiếu evidence cần `abstention_correct` (`true`/`false`) do người chấm xác nhận. Lần chạy lỗi không cần nhãn.

```bash
python3 starter/agent_metrics.py --cases <cases.json> --traces <traces.json> --reviews <reviews.json> --split test --output starter/results/agent-test/summary.json
```

Thay đường dẫn trong `< >`. Script tổng hợp nhãn bạn đã chấm, không tự đánh giá nội dung. Mẫu số bằng 0 nghĩa là không áp dụng. Lỗi kết nối được báo riêng, không tính vào chất lượng; giữ trace và nhãn để đối chiếu.

## Lưu ý

TF-IDF dùng cách tách từ đơn giản. `baseline.py` dùng mặc định context budget 800 ký tự, gồm cả nguồn; chunk không vừa bị bỏ. Bạn có thể chọn cách khác. File kết quả cùng tên sẽ bị ghi đè. `starter/data/` và `starter/results/` không được Git theo dõi mặc định; nếu được phép chia sẻ, dùng `git add -f <đường-dẫn>`, hoặc ghi cách tái tạo. Không nộp dữ liệu mật và secret.

Khung agent ưu tiên giữ chunk từ lần search trước. Khi chunk đầu chiếm gần hết budget, lần search sau có thể không thêm được evidence; hãy kiểm tra trace và thử cách chọn context phù hợp trên dev. Citation ID hợp lệ hoặc verifier trả “đúng” không đảm bảo câu trả lời được hỗ trợ; cần chấm thủ công.
