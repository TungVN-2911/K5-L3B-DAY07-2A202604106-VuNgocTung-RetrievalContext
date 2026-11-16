# Báo cáo Day 7: Vũ Ngọc Tùng (2A202604106)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** *AI Engineering Guidebook* (DailyDoseofDS.com, bản 2025), PDF 384 trang, trong đó 343 trang có nội dung được index; 41 trang bị loại (front matter, bìa chương, trang dưới 200 ký tự). Không có giấy phép mở, giảng viên đồng ý cho dùng trong lab, PDF không commit. Nguồn, sha256 và cách lấy file ghi trong [data/manifest.json](../../data/manifest.json). Index có 621 chunk (strategy A của Day 6: fixed 500 ký tự, overlap 50).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker, dùng lại volume Day 6), cosine, exact search. Embedding `nomic-embed-text` qua Ollama 0.40.16, 768 chiều, tiền tố `search_document:` / `search_query:`. Chạy `ingest.py --reindex` ra cùng sha256 `chunks.json` với index có sẵn.
- **Agent model, cách chạy và phân chia dev/test:** `gemini-3.5-flash-lite` (Gemini API, function calling, temperature 0, thinking để mặc định). Lệnh chạy ở [starter/README.md](../../starter/README.md). Dev gồm 15 câu có đáp án và 3 câu thiếu evidence; test gồm 5 câu có đáp án và 3 câu thiếu evidence ([data/questions.json](../../data/questions.json)).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - q01–q20 lấy từ benchmark Day 6. Evidence là trích nguyên văn, phải nằm trọn trong một chunk, đúng nguồn và đúng trang (`benchmark.py check` kiểm 20/20). Day 6 dùng test chỉ để báo kết quả sau khi chốt chunking; khi soạn A2 mình có đọc câu test để soát schema nhưng không dùng để chỉnh.
  - u01–u06 là câu gần chủ đề mà sách không có thông tin. Đã dò `absent_patterns` trên text của toàn bộ 384 trang. Giới hạn: chữ nằm trong ảnh chưa được OCR.
  - Hit tính theo `baseline.hit`: chunk đúng nguồn, đúng trang và chứa nguyên văn evidence. Grounded answer chấm 0/1/2 theo BENCHMARK.md.

## Retrieval và thử nghiệm

Corpus, index, câu hỏi và nhãn giống nhau cho mọi cấu hình. Query là câu hỏi gốc, không viết lại; không dùng filter trừ khi ghi rõ. Thống kê trên 15 câu dev có đáp án. Context lấy top-3, budget 2000 ký tự (gồm header nguồn). Thời gian là trung vị mỗi query, chi phí 0 USD vì chạy local.

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense pgvector, top-10 (context top-3), không filter | Baseline Day 6 | dev | 11/15 (73%) | 15/15 (100%) | 15/15 (100%) | 53 ms / 0 USD |
| Phương pháp 2 | BM25 (k1=1.5, b=0.75, regex `\w+`), cùng 621 chunk | Từ khóa hiếm (GPT-3, HyDE, CAG) trúng tốt hơn dense | dev | 9/15 (60%) | 13/15 (87%) | 13/15 (87%) | 1.7 ms / 0 |
| Phương pháp 3 | Hybrid RRF(dense top-20, BM25 top-20), k=60 | Gộp hai tín hiệu bù lỗi cho nhau | dev | 9/15 (60%) | 14/15 (93%) | 14/15 (93%) | 23 ms / 0 |
| Cải thiện 1 (phương pháp 4) | Hybrid top-20 → cross-encoder `ms-marco-MiniLM-L-12-v2` → top-k | Cross-encoder đọc cặp query–chunk nên sửa được lỗi xếp hạng của dense và RRF | dev | **14/15 (93%)** | **15/15 (100%)** | **15/15 (100%)** | 701 ms / 0 |
| Cải thiện 2 | Như cải thiện 1 + filter `chapter` người dùng nêu (q12, q14, q15) | Filter loại chunk cùng chủ đề khác chương | dev | 14/15 | 15/15 | 15/15 | 638 ms / 0 |
| Cải thiện 3 | Như cải thiện 1, 50 candidate | Recall stage 1 cao hơn thì rerank có thêm lựa chọn | dev | 14/15 | 15/15 | 15/15 | 1653 ms / 0 |
| Cải thiện 4 | Như cải thiện 1, reranker `bge-reranker-base` (1 GB) | Reranker lớn hơn xếp tốt hơn | dev | 14/15 | 15/15 | 15/15 | 1969 ms / 0 |
| TF-IDF (starter) | `baseline.py`, tham khảo | | dev | 7/15 (47%) | 11/15 (73%) | 11/15 (73%) | 1.4 ms / 0 |
| Cấu hình đã chọn | | | test | | | | |
| Baseline | | | test | | | | |

Hybrid RRF và rerank là hai phương pháp khác nhau, đo riêng. Rerank đồng thời là cải thiện 1. MRR@10 trên dev: TF-IDF 0.636, BM25 0.735, dense 0.844, hybrid 0.761, rerank 0.967. Cải thiện 2 (filter) không thay đổi kết quả: q12 và q14 vốn đã ở hạng 1, còn q15 với BM25 vẫn ở hạng 3 vì các chunk gây nhiễu nằm cùng chương `ai_agents`. Cải thiện 3 và 4 cũng không tăng chất lượng mà chỉ làm chậm hơn 2.4 đến 2.8 lần.

**Lỗi dev, nguyên nhân, cách thử và kết quả:**

1. **Agent từ chối nhầm vì citation viết gộp** (`dev-*-v0`). Model ghi `[C1, C2]`, trong khi validator chỉ nhận `[C1]`. Kết quả: rerank chỉ trả lời 9/15 câu có đáp án (từ chối nhầm 6), dense 10/15 (từ chối nhầm 5). Các câu này có context đúng (Context Evidence 15/15 và 14/15) và nội dung trả lời đúng, nhưng bị validator chặn. Giả thuyết: lỗi nằm ở bộ parse, không ở model. Cách sửa: nhận cả nhóm `[C1, C2]` / `[C1; C2]` và nói rõ định dạng khi yêu cầu model sửa. Sau khi sửa, cả hai cấu hình trả lời 15/15 câu có đáp án, từ chối nhầm 0/15, abstention vẫn 3/3.
2. **Hybrid RRF làm tụt evidence đã đứng đầu ở dense.** Ở q06 ("What does fine-tuning mean?"), dense xếp định nghĩa ở hạng 1, BM25 xếp hạng 9 vì nhiều chunk lặp lại từ "fine-tuning". RRF coi hai bên ngang nhau nên chunk này về hạng 4, rơi khỏi top-3. Ở q02, RRF hòa điểm tuyệt đối (hạng 1+2 so với 2+1) nên thứ tự do ID quyết định. Giả thuyết: cần một bước chấm trực tiếp cặp query–chunk. Sau khi thêm rerank, q06, q03, q09, q10 lên hạng 1; Hit@1 tăng từ 9/15 lên 14/15.
3. **Khẳng định không có trong context** (dev-dense q12). Câu trả lời có cụm "and dynamic data". Thông tin này có trong sách (p144 c0) nhưng chunk đó không được retrieve, nên model đã suy diễn. Câu này chấm 1/2. Với cấu hình rerank, câu trả lời q12 chỉ dùng nội dung có trong context.

**Tiêu chí chọn, chốt trước khi chạy test:** (1) Grounded answer trung bình trên dev; (2) abstention đúng và từ chối nhầm; (3) Hit@1/MRR retrieval trên dev; (4) nếu vẫn hòa thì chọn cấu hình có thời gian và chi phí thấp hơn. Cấu hình chọn: **agent `gemini-3.5-flash-lite` + retriever rerank** (hybrid RRF dense+BM25 top-20 → MiniLM-L-12 → top-4), tối đa 3 search, budget 6000 ký tự, không filter cố định. Lý do: grounded 2.00 so với 1.93, Hit@1 14/15 so với 11/15, abstention bằng nhau. Đánh đổi: mỗi lượt retrieval chậm hơn dense khoảng 650 ms, nhỏ so với thời gian gọi API. Baseline chạy trên test: cùng agent, retriever dense.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - Tool `search(query, chapter?)`, `answer(answer, citations)`, `abstain(reason)`. Dùng `mode: ANY` nên mỗi lượt model phải gọi một tool.
  - Mỗi search lấy top-4. Chunk mới được gán mã `C#` và đưa nguyên văn; chunk đã có chỉ nhắc lại mã. Tổng text chunk mới tối đa 6000 ký tự.
  - Dừng khi model gọi `answer` hoặc `abstain`. Sau 3 lượt search, model chỉ còn được gọi `answer` hoặc `abstain`.
  - Citation phải nằm trong context và được đánh dấu trong text. Sai thì cho sửa một lần, sai tiếp thì từ chối.
  - Filter `chapter` chỉ nhận giá trị trong enum, được model đặt khi câu hỏi nêu tên chương. Trace ghi cả `model_chapter` và filter thực tế.
  - Lỗi HTTP 429/5xx được retry có backoff; hết retry thì ghi `status: error`, không thay bằng kết quả giả.

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Dense v0 (trước sửa) / dev | 18/18 | search {1: 13, 2: 1, 3: 4} | chưa chấm (chỉ dùng để chẩn đoán lỗi 1) | | 10/10 | 3/3 | 5/15 |
| Rerank v0 (trước sửa) / dev | 18/18 | search {1: 14, 3: 4} | chưa chấm | | 9/9 | 3/3 | 6/15 |
| Dense (baseline) / dev | 18/18 | search {1: 13, 2: 1, 3: 4}; 46 lượt LLM | 0/1/14 (1.93) | 22/22 | 15/15 | 3/3 | 0/15 |
| Rerank (chọn) / dev | 18/18 | search {1: 14, 2: 1, 3: 3}; 44 lượt LLM | 0/0/15 (2.00) | 25/25 | 15/15 | 3/3 | 0/15 |
| Rerank (chọn) / test | | | | | | | |
| Dense (baseline) / test | | | | | | | |

Nhãn dev do Claude soạn nháp sau khi đọc đối chiếu chunk được cite (trường `labeled_by` trong `reviews.json`); người làm bài cần xác nhận. Context Evidence của agent (context cuối có chứa evidence gold): 15/15 ở cả hai cấu hình sau khi sửa. Mọi câu thiếu evidence đều dùng hết 3 lượt search rồi mới từ chối.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval: [starter/results/retrieval/](../../starter/results/retrieval/) (`dev`, `filter`, `cand50`, `bge`).
  - Agent: [starter/results/agent/](../../starter/results/agent/) (`traces.json`, `reviews.json`, `summary.json`); tổng hợp ở `dev-eval.json`.
  - Token trung bình mỗi câu dev: rerank 2194 input / 104 output, dense 2348 / 99.
  - Thời gian trung vị mỗi câu: rerank 13.3 s, dense 6.7 s. Phần lớn là chờ API và retry 429 của free tier, câu nhanh nhất chỉ 2–3 s.
  - Chi phí retrieval 0 USD. Chi phí Gemini: chưa có giá niêm yết cho `gemini-3.5-flash-lite` trong code, chỉ ghi token.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:** xem ba lỗi dev ở trên. Giới hạn:
  - Dev chỉ có 15 câu có đáp án, mỗi câu tương ứng 6.7 điểm phần trăm.
  - Câu hỏi chủ yếu tra cứu một đoạn, chưa có câu multi-hop.
  - Một chunk 500 ký tự đôi khi cắt ngang ý (q13).

## Thử thách nâng cao (nếu thực hiện)

Chưa thực hiện.
