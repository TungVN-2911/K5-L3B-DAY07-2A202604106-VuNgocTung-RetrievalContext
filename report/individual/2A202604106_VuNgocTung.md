# Báo cáo Day 7: Vũ Ngọc Tùng (2A202604106)

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** *AI Engineering Guidebook* (DailyDoseofDS.com, bản 2025), PDF 384 trang, trong đó 343 trang có nội dung được index; 41 trang bị loại (front matter, bìa chương, trang dưới 200 ký tự). Không có giấy phép mở, giảng viên đồng ý cho dùng trong lab, PDF không commit. Repo để public theo sự cho phép đó; trace retrieval chỉ giữ trích đoạn tối đa 160 ký tự mỗi chunk, kèm chunk ID, trang và `char_start`/`char_end`, đủ để dựng lại bằng `ingest.py` từ PDF gốc. Các commit trước `1470f0a` còn chứa toàn văn chunk trong trace; mình không viết lại lịch sử git. Nguồn, sha256 và cách lấy file ghi trong [data/manifest.json](../../data/manifest.json). Index có 621 chunk (strategy A của Day 6: fixed 500 ký tự, overlap 50).
- **Vector store, embedding model, phiên bản và số chiều:** PostgreSQL 16.16 + pgvector 0.8.7 (Docker, dùng lại volume Day 6), cosine, exact search. Embedding `nomic-embed-text` qua Ollama 0.40.16, 768 chiều, tiền tố `search_document:` / `search_query:`. Chạy `ingest.py --reindex` ra cùng sha256 `chunks.json` với index có sẵn.
- **Agent model, cách chạy và phân chia dev/test:** `gemini-3.5-flash-lite` (Gemini API, function calling, temperature 0, thinking để mặc định). Lệnh chạy ở [starter/README.md](../../starter/README.md). Dev gồm 15 câu có đáp án và 3 câu thiếu evidence; test gồm 5 câu có đáp án và 3 câu thiếu evidence ([data/questions.json](../../data/questions.json)).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - q01–q20 lấy từ benchmark Day 6. Evidence là trích nguyên văn, phải nằm trọn trong một chunk, đúng nguồn và đúng trang (`benchmark.py check` kiểm 20/20). Day 6 dùng test chỉ để báo kết quả sau khi chốt chunking; khi soạn A2 mình có đọc câu test để soát schema nhưng không dùng để chỉnh.
  - u01–u06 là câu gần chủ đề mà sách không có thông tin. Đã dò `absent_patterns` trên text của toàn bộ 384 trang. Giới hạn: chữ nằm trong ảnh chưa được OCR.
  - Hit tính theo `baseline.hit`: chunk đúng nguồn, đúng trang và chứa nguyên văn evidence. Grounded answer chấm 0/1/2 theo BENCHMARK.md.

## Retrieval và thử nghiệm

Corpus, index, câu hỏi và nhãn giống nhau cho mọi cấu hình. Query là câu hỏi gốc, không viết lại; không dùng filter trừ khi ghi rõ. Bảng retrieval dưới đây dùng query gốc. Trong agent, model tự viết lại query (ví dụ q01: "formally define a large language model") và có thể search nhiều lượt; kết quả của pipeline này được báo riêng ở Context Evidence của agent, từng bước ghi trong `events` của trace. Thống kê trên 15 câu dev có đáp án. Context lấy top-3, budget 2000 ký tự (gồm header nguồn). Thời gian là trung vị mỗi query, chi phí 0 USD vì chạy local.

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
| Cấu hình đã chọn | Rerank (hybrid top-20 → MiniLM-L-12), như cải thiện 1 | Chốt trên dev, không chỉnh sau khi chạy test | test | 3/5 (60%) | 5/5 (100%) | 5/5 (100%) | 576 ms / 0 USD |
| Baseline | Dense pgvector | | test | 4/5 (80%) | 5/5 (100%) | 5/5 (100%) | 42 ms / 0 USD |

Hybrid RRF và rerank là hai phương pháp khác nhau, đo riêng. Rerank đồng thời là cải thiện 1. MRR@10 trên dev: TF-IDF 0.636, BM25 0.735, dense 0.844, hybrid 0.761, rerank 0.967. Cải thiện 2 (filter) không thay đổi kết quả: q12 và q14 vốn đã ở hạng 1, còn q15 với BM25 vẫn ở hạng 3 vì các chunk gây nhiễu nằm cùng chương `ai_agents`. Cải thiện 3 và 4 cũng không tăng chất lượng mà chỉ làm chậm hơn 2.4 đến 2.8 lần.

**Kết quả test (chạy một lần sau khi chốt):** Hit@1 của rerank thấp hơn dense (3/5 so với 4/5, MRR 0.800 so với 0.867), ngược với dev. Ở q17 và q18, rerank xếp hạng 1 một chunk cùng chủ đề nằm ngay trang bên cạnh (p289 nhắc lại "tool-name hallucinations"; p365 "Load LoRA Adapters"), điểm chỉ chênh 4.12 so với 4.01 và 3.71 so với 3.69, evidence ở hạng 2. Ở q20, rerank đưa evidence từ hạng 3 lên hạng 1. Evidence Hit@3 và Context Evidence bằng nhau (5/5). Với 5 câu, một câu tương ứng 20 điểm phần trăm, nên chưa đủ để kết luận rerank kém hơn dense. Mình giữ nguyên cấu hình, không chỉnh sau khi xem test.

**Chấm context 0/1/2 trên test** (top-3 context retrieval, 5 câu có đáp án × 2 cấu hình; [context_labels.json](../../starter/results/retrieval/test/context_labels.json)): dense 2/2/2/2/2, rerank 2/2/2/2/2. Cả 5 câu đều có chunk chứa nguyên văn evidence, đủ điều kiện và nguồn trang. Không có câu nào ở mức 0 hoặc 1.

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
| Rerank (chọn) / test | 8/8 | search {1: 4, 2: 1, 3: 3}; 23 lượt LLM | 0/1/4 (1.80) | 7/7 | 5/5 | 3/3 | 0/5 |
| Dense (baseline) / test | 8/8 | search {1: 3, 2: 1, 3: 4}; 26 lượt LLM | 0/1/4 (1.80) | 7/7 | 5/5 | 3/3 | 0/5 |

Trên test, cả hai cấu hình đều có 0 lỗi provider. Context Evidence của agent 5/5. Câu q19 ở cả hai cấu hình chấm 1: chọn đúng RFT và đúng điều kiện, nhưng tự diễn giải RFT là "Reasoning Fine-Tuning"; sách ghi Reinforcement Fine-Tuning (p92), trang này không có trong context. Đây là lỗi model tự thêm kiến thức ngoài context. Test: token rerank 24120/727, dense 28744/815; thời gian trung vị mỗi câu rerank 24.6 s, dense 5.7 s, chủ yếu do chờ retry 429. Số liệu lấy từ `agent_metrics.py` gốc ([test-rerank/agent_metrics.json](../../starter/results/agent/test-rerank/agent_metrics.json)).

Nhãn dev và test do Claude soạn nháp sau khi đọc đối chiếu chunk được cite. Người làm bài đã đọc lại toàn bộ bằng `benchmark.py review` (in câu trả lời kèm toàn văn chunk được cite) và xác nhận giữ nguyên; trường `labeled_by` trong `reviews.json` và `context_labels.json` ghi rõ cả hai bước. Context Evidence của agent (context cuối có chứa evidence gold): 15/15 ở cả hai cấu hình sau khi sửa. Mọi câu thiếu evidence đều dùng hết 3 lượt search rồi mới từ chối.

**Tác động của tham số agent** (dev, retriever rerank, mỗi lần đổi một tham số so với cấu hình chọn). Các thí nghiệm này **chạy sau khi đã chạy test**, chỉ để đo tác động theo GUIDE §3, không dùng để chọn hay đổi cấu hình. Chỉ số tự động lấy từ trace; grounded answer chưa chấm tay. Kết quả ở [starter/results/agent/ablation/](../../starter/results/agent/ablation/).

| Cấu hình | Giả thuyết | Trả lời / từ chối | Context Evidence | Từ chối nhầm | Search {lượt: số câu} | Lượt LLM / citation bị chặn | Token in/out | Trung vị s |
|---|---|---|---|---|---|---|---|---|
| Chọn: top-4, 3 search, 6000 ký tự | | 15/3 | 15/15 | 0/15 | {1: 14, 2: 1, 3: 3} | 44 / 1 | 39 488 / 1 865 | 13.3 |
| top-k = 2 | Ít chunk hơn thì tốn ít token hơn nhưng dễ thiếu evidence | 15/3 | 15/15 | 0/15 | {1: 13, 2: 2, 3: 3} | 48 / 4 | 36 097 / 1 874 | 26.5 |
| max_searches = 1 | Câu có đáp án không đổi; câu thiếu evidence từ chối sớm hơn | 15/3 | 15/15 | 0/15 | {1: 18} | 45 / 9 | 38 091 / 2 145 | 5.9 |
| budget = 1500 ký tự | Context bị cắt nên mất evidence ở câu cần nhiều chunk | 14/4 | 14/15 | **1/15 (q10)** | {1: 12, 2: 1, 3: 5} | 48 / 1 | 38 779 / 1 900 | 5.0 |

- **top-k = 2:** token input giảm khoảng 9% mà Context Evidence không đổi, nhưng hai câu cần thêm một lượt search và citation bị chặn nhiều hơn (4 so với 1).
- **max_searches = 1:** 3/3 câu thiếu evidence vẫn được từ chối đúng chỉ sau một lượt search, và tổng token gần như bằng cấu hình chọn. Tuy nhiên citation bị chặn tăng lên 9 lần: khi chỉ còn được gọi `answer`/`abstain`, model hay quên đánh dấu `[C#]`. Trên dev, tăng lên 3 search chủ yếu chỉ làm câu thiếu evidence tốn thêm token, nhưng giúp câu như q12 tìm lại với query khác.
- **budget = 1500:** xác nhận giả thuyết. Ở q10, chunk evidence p121 đứng hạng 4 ngay lượt search đầu nhưng bị loại vì ba chunk trên đã dùng 1325/1500 ký tự; các lượt search sau cũng không thêm được vì budget đã gần hết, nên agent từ chối nhầm. Budget 6000 ký tự (khoảng 12 chunk) đủ cho cả 18 câu dev (trung bình context 2077 ký tự).
- Thời gian trung vị dao động 5–27 s chủ yếu do retry 429 của free tier, nên không phản ánh chi phí tính toán của từng cấu hình.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q06 (rerank, dev) | Có evidence | `search("fine-tuning definition mean")` → C1 = p68 c0 (hạng 1, có gold), C3 = p68 c1 | "Fine-tuning means adjusting the weights of a pre-trained model on a new dataset … [p68 c0][p68 c1]" | 2, citation 2/2 đúng. Dense+BM25 (hybrid) để chunk này ở hạng 4; rerank kéo lên hạng 1 |
| q12 (dense, dev) | Có evidence, có filter | 2 search, model tự đặt `chapter=rag` vì câu hỏi nêu "In the RAG chapter" → p143 c0 (gold), p143 c1 | "…caching it directly in the model's key-value memory. By dividing knowledge into static … and dynamic data … [p143 c0][p143 c1]" | 1: ý chính đúng, nhưng "dynamic data" chỉ có ở p144 c0, không nằm trong context |
| u01 (rerank, dev) | Thiếu evidence | 3 search: "BM25 parameters k1 b hybrid retrieval…", "BM25 k1 b hybrid search", "BM25" → không có chunk nào nói BM25 | `abstain`: "The retrieved chunks do not contain any information about BM25 parameters…" | Từ chối đúng, không đưa giá trị k1/b nào |
| q19 (rerank, test) | Có evidence, có điều kiện | 1 search → C1, C2 = p94 (C2 chứa gold) | "…you should use RFT (Reasoning Fine-Tuning), as a reward function can score correctness … [p94 c0][p94 c1]" | 1: chọn đúng RFT và điều kiện, nhưng diễn giải tên sai (sách: Reinforcement, p92), ý này không có trong context |
| u06 (rerank, test) | Thiếu evidence | 3 search "GPT-4 maximum context window tokens", "GPT-4 tokens context window", "GPT-4 context" → p380, p142, p112, p181, p25… | `abstain`: "do not contain information about the maximum context window of GPT-4" | Từ chối đúng; không lấy nhầm "4k" của Llama3-70B (p331) |
| q05 (rerank-v0, dev) | Lỗi | 1 search → C1 = p51 (gold) | Model trả lời đúng nhưng ghi `[C1, C3]`; validator v0 chỉ nhận `[C1]` nên từ chối hai lần, kết quả là `abstain` | Lỗi 1: từ chối nhầm do parser. Sau khi sửa: trả lời, chấm 2 |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Retrieval: [starter/results/retrieval/](../../starter/results/retrieval/) (`dev`, `filter`, `cand50`, `bge`).
  - Agent: [starter/results/agent/](../../starter/results/agent/) (`traces.json`, `reviews.json`, `summary.json`); tổng hợp ở `dev-eval.json`.
  - Token trung bình mỗi câu dev: rerank 2194 input / 104 output, dense 2348 / 99.
  - Thời gian trung vị mỗi câu: rerank 13.3 s, dense 6.7 s. Phần lớn là chờ API và retry 429 của free tier, câu nhanh nhất chỉ 2–3 s.
  - Chi phí: **0 USD**. Retrieval, embedding và rerank chạy local; Gemini dùng **free tier** của Google AI Studio nên không phát sinh phí. Token vẫn được ghi để ước tính nếu chuyển sang gói trả phí: dev rerank 39 488 input / 1 865 output, test rerank 24 120 / 727. Đánh đổi của free tier: giới hạn tốc độ gây lỗi 429 và retry, làm thời gian trung vị mỗi câu tăng lên 6–25 s, trong khi câu không bị giới hạn chỉ mất 2–4 s.
- **Lệnh tái lập test:**
  - `python starter/benchmark.py retrieval --split test --methods dense rerank`
  - `python starter/agent.py --split test --method rerank` (và `--method dense`)
  - `python starter/benchmark.py agent starter/results/agent/test-dense starter/results/agent/test-rerank --output starter/results/agent/test-eval.json`
- **Kết luận:** agent đạt từ chối đúng 6/6 và từ chối nhầm 0/20 trên dev và test, citation hợp lệ 32/32 với rerank. Trên dev, rerank tốt hơn dense về retrieval (Hit@1 14/15 so với 11/15) và grounded (2.00 so với 1.93). Trên test, hai cấu hình bằng nhau về câu trả lời (1.80), còn Hit@1 rerank thấp hơn (3/5 so với 4/5). Lợi ích của rerank chưa được xác nhận trên test vì test quá nhỏ; đánh đổi là thêm khoảng 550–650 ms mỗi lượt retrieval. Lỗi còn lại chủ yếu nằm ở bước sinh câu trả lời: model thêm chi tiết ngoài context (q12 dev-dense, q19 test).
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:** xem ba lỗi dev ở trên. Giới hạn:
  - Dev chỉ có 15 câu có đáp án, mỗi câu tương ứng 6.7 điểm phần trăm.
  - Câu hỏi chủ yếu tra cứu một đoạn, chưa có câu multi-hop.
  - Một chunk 500 ký tự đôi khi cắt ngang ý (q13).

## Thử thách nâng cao (nếu thực hiện)

> **Trạng thái:** đã xong tài liệu, index, benchmark và thử nghiệm trên dev. **Test chưa chạy.** Tỷ lệ đạt test và điều kiện bonus sẽ điền sau khi chạy cấu hình đã chốt một lần trên test.

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - *IPCC AR6 Synthesis Report: Climate Change 2023* (Full Volume), 186 trang PDF, ISBN 978-92-9169-164-7, DOI 10.59327/IPCC/AR6-9789291691647.
  - Trang ii ghi: *"Short extracts from this publication may be reproduced without authorization provided that complete source is clearly indicated."* PDF không commit; cách lấy file và sha256 ở [data/advanced/manifest.json](../../data/advanced/manifest.json).
  - Đã thử trước *OECD Economic Surveys: Viet Nam 2025* (139 trang, CC BY 4.0) nhưng bỏ: mọi hình đánh số đều là biểu đồ, ảnh duy nhất là 4 ảnh bìa trang trí, nên không đủ nhóm "Ảnh, sơ đồ".
- **Cách xử lý text, bảng, ảnh, biểu đồ** ([starter/multimodal.py](../../starter/multimodal.py)):
  - *Text:* các block của PyMuPDF (bỏ header/footer) được gom thành chunk ≤ 900 ký tự, giữ bbox. Kết quả: 804 chunk.
  - *Trang có hình/bảng (51 trang):* chọn khi trang có caption "Figure/Table…" (kể cả "Table 2.2 Projected…" không có dấu ":"), có > 40 nét vector, hoặc có ảnh lớn. Mỗi trang được render 150 dpi và gửi Gemini vision (`gemini-3.5-flash-lite`). Vision trả JSON gồm loại phần tử (table/chart/diagram/map/image/decorative), nhãn, tiêu đề, `box_2d` và mô tả có cấu trúc (bảng chép thành markdown, biểu đồ ghi trục/chú giải/giá trị, sơ đồ ghi hộp/mũi tên/ký hiệu).
  - *Kết quả:* 98 phần tử (42 biểu đồ, 25 bản đồ, 23 sơ đồ, 8 bảng), 1 ảnh trang trí bị loại. Mỗi phần tử lưu bbox, ảnh crop 150 dpi và text PDF trong vùng. Mô tả dài được tách thành part có cùng nhãn, ra 155 chunk trực quan; tổng index 959 chunk trong pgvector (strategy `ipcc`, `nomic-embed-text`).
  - *Sửa nhãn vùng:* 41 phần tử bị vision gán nhầm nhãn (thường đoán "Figure SPM.x" khi caption nằm ở trang kế bên). Nhãn được đặt lại theo caption thật trong text layer (cùng trang, trang sau, trang sau nữa, trang trước); nhãn gốc lưu ở `vision_label`.
  - *Màu chính xác từ vector:* 53 vùng có màu ô được chú giải bằng ô mẫu. Mỗi ô ghi `hàng | cột -> #hex (ý nghĩa chú giải)`, đọc thẳng từ fill của PDF.
- **Phần chưa xử lý và ảnh hưởng:**
  - p39 (Figure SPM.5): vision trả `finishReason: RECITATION`; thử lại với prompt rút gọn và prompt diễn đạt lại vẫn ra rỗng. Chỉ còn caption/text được index. Benchmark không hỏi hình này.
  - 22 trang không có chunk text (bìa, trang trắng, trang chỉ có hình). Các trang chỉ có hình đã được phủ bằng vision.
  - Chấm chấm biểu thị độ tin cậy (•/••/•••) trong SPM.7 không được trích tự động; agent phải đọc từ ảnh.
- **Kiểm tra trích xuất với bản gốc:**
  - Table 2.2: 12/12 ô khớp ảnh trang.
  - Table 3.1: 5 ô dùng trong benchmark khớp (890 [640–1160], 510 [330–710], 1.7 [1.5–1.8], 59 [50–77], 20 [13–41]).
  - Table 2.1: mô tả vision **sai** ô human contribution của "Ocean heat content" (ghi orange/very likely; gốc là đỏ cam/extremely likely). Đây là lý do thêm bước trích màu vector. Màu vector của các hàng Ocean heat content, Greenland (Table 2.1) và Efficient livestock systems, Forest-based adaptation (SPM.7) đều khớp bản gốc.
- **Cách chia benchmark, schema evidence:** [data/advanced/questions.json](../../data/advanced/questions.json), commit `59dc17a` **trước** mọi thử nghiệm.
  - Test 30 câu: bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4, thiếu evidence 6. Dev 23 câu: 19 có đáp án, phủ 5 nhóm, cộng 4 câu thiếu evidence.
  - Mỗi câu có nhóm, đáp án, đơn vị, điều kiện, phép tính và evidence (trang PDF và trang in, vùng như "Table 3.1" / "Figure SPM.7 panel a", vị trí hàng/cột, giá trị). Có thêm evidence thay thế (ví dụ SPM.7 ↔ Figure 4.4, SPM.6 ↔ Figure 4.2) và lý do chọn câu.
  - Gold đọc từ text layer hoặc ảnh trang gốc, không lấy từ pipeline. Text evidence được kiểm tự động bằng so khớp nguyên văn; câu thiếu evidence được kiểm bằng pattern trên toàn bộ 186 trang (giới hạn: chữ trong ảnh raster).
  - Dev và test không hỏi cùng một ô/giá trị.

**Retrieval trên dev** (19 câu có đáp án, index cuối). Một kết quả tính là trúng khi cùng trang và cùng vùng Table/Figure với evidence trực quan, hoặc cùng trang và chứa trích dẫn với evidence text. Multi-hop phải trúng đủ mọi evidence.

| Phương pháp | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Bảng (Hit@5) | Multi-hop (Hit@5) | ms/query |
|---|---|---|---|---|---|---|---|
| BM25 | 11/19 | 16/19 | 16/19 | 17/19 | 4/5 | 1/3 | 4 |
| Dense | 12/19 | 14/19 | 16/19 | 17/19 | 4/5 | 1/3 | 50 |
| **Hybrid RRF** | **14/19** | **16/19** | 16/19 | **18/19** | 4/5 | 1/3 | 27 |
| Hybrid + rerank MiniLM | 11/19 | 14/19 | 14/19 | 18/19 | 2/5 | 1/3 | 5961 |

Rerank kém và chậm vì chunk bảng/hình dài, MiniLM chỉ đọc khoảng 512 token đầu. Multi-hop một lượt chỉ đạt 1/3, nên agent tìm nhiều lượt cho từng phần.

**Agent trên dev** (`gemini-3.5-flash-lite`, retriever hybrid). Nhãn đạt/không đạt do Claude soạn nháp theo tiêu chí của ADVANCED.md: đúng và đủ, citation đúng trang và vùng, không thêm khẳng định sai. Từ v5 trở đi chấm chặt câu so sánh (phải nêu quan hệ). Trace ở [starter/results/advanced/agent/](../../starter/results/advanced/agent/).

| Lượt | Thay đổi / giả thuyết | Kết quả dev | Lỗi chính |
|---|---|---|---|
| quota-error | Cấu hình đầu: view ảnh vùng gốc | 13 trả lời, 10 lỗi 429 | Hết 500 request/ngày của free tier, không tính chất lượng |
| (flash) | Thử `gemini-3.5-flash` | 5 câu rồi dừng | Free tier chỉ 20 request/ngày |
| v1 | Như trên, đủ quota | 21/23 | ad04, ad05 đọc nhầm sắc độ màu |
| v2 | + sửa nhãn vùng, + màu vector, + tool `zoom` 300 dpi | ~18/23 | Zoom liên tục, hết lượt view, từ chối nhầm 4 câu: **bỏ zoom** |
| v3 | Bỏ zoom; chỉ ghi màu khi có chú giải khớp (Table 3.1 hết nhiễu) | 21/23 | ad04; ad13 chưa nêu quan hệ so sánh |
| v4 | + ghi chú "tên màu trong mô tả là đoán" | 20/23 | ad02 lỗi 503; ad04; ad12 thiếu citation cho một ý |
| v5 | Dòng màu ghi thẳng ý nghĩa chú giải; kiểm tra citation theo từng câu có số; retry 10 lần | 19/23 | ad04, ad05; ad13, ad14 không nêu quan hệ |
| v6 | + mở rộng part anh em có màu; + quy tắc nêu quan hệ so sánh | 20/23 | ad02 thêm giá trị sai cho C1b; ad04, ad05 |
| v7 (dang dở) | + trả màu chính xác kèm ảnh khi `view`; + "chỉ trả lời phần được hỏi" | 18/19 câu có đáp án đã chạy | ad05 (feasibility Low đọc thành Medium); 4 câu thiếu evidence chưa chạy |

Bốn câu thiếu evidence của dev được từ chối đúng ở mọi lượt đã chạy đủ (v1, v3–v6). Lỗi còn lại chủ yếu là đọc ký hiệu và màu trong ma trận SPM.7.

- **Cấu hình sẽ chốt (theo dev):** hybrid RRF (dense + BM25, 30 candidate) → top-5, tối đa 4 search, 3 lượt `view` (ảnh crop vùng gốc kèm màu vector), budget 12000 ký tự, mở rộng part anh em có màu, kiểm tra citation theo từng câu có số, retry 10 lần, temperature 0. Test sẽ chạy **một lần** với cấu hình này.

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---|---|---|
| Bảng | chưa chạy | | |
| Ảnh, sơ đồ | chưa chạy | | |
| Biểu đồ | chưa chạy | | |
| Multi-hop | chưa chạy | | |
| Text | chưa chạy | | |
| Thiếu bằng chứng | chưa chạy | | |
| **Toàn bộ** | chưa chạy | | Cần ≥ 27/30 |

- **Thời gian, chi phí, lỗi hệ thống:**
  - Index: vision 51 trang, khoảng 11 phút lần đầu (có cache), 75 605 token input / 31 046 output; embed 959 chunk khoảng 19 giây.
  - Agent dev: 4–50 giây mỗi câu khi không bị giới hạn tốc độ.
  - Chi phí 0 USD (Gemini free tier, embedding/rerank local).
  - Free tier giới hạn 500 request/ngày cho `flash-lite`. Các lượt chạy phải chia nhiều phiên và có lúc đổi sang API key khác; lỗi 429/503 được retry và ghi vào trace.
- **Giới hạn:**
  - `flash-lite` đọc sắc độ gần nhau và số chấm không ổn định giữa các lần chạy (ad05 đúng ở v3, v4, sai ở v5–v7).
  - Nhãn chấm là bản nháp, cần người làm bài xác nhận.
  - Dev nhỏ (19 câu có đáp án), mỗi câu tương ứng khoảng 5 điểm phần trăm.
- **Tái lập:**

```bash
.venv/Scripts/python -m pip install -r requirements.txt     # có pymupdf
# PDF tại data/advanced/IPCC_AR6_SYR_FullVolume.pdf (xem manifest), GEMINI_API_KEY trong starter/.env
.venv/Scripts/python starter/multimodal.py index                      # text + vision + màu vector + pgvector (cache vision ở starter/data/advanced/vision)
.venv/Scripts/python starter/multimodal.py retrieval --split dev
.venv/Scripts/python starter/multimodal.py run --split dev --method hybrid
.venv/Scripts/python starter/multimodal.py run --split test --method hybrid   # một lần, sau khi chốt
.venv/Scripts/python starter/multimodal.py score starter/results/advanced/agent/test-hybrid   # sau khi chấm labels.json
```
