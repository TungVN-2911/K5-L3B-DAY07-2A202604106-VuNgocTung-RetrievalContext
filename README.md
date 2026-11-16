# Day 7 — Agentic RAG

## Bài làm: Vũ Ngọc Tùng (2A202604106)

| Thành phần | Lựa chọn |
|---|---|
| Corpus | *AI Engineering Guidebook* (2025), 343/384 trang, 621 chunk; nguồn và quyền dùng ở [data/manifest.json](data/manifest.json) (PDF không commit) |
| Vector store | PostgreSQL 16 + pgvector 0.8.7 của Day 6, `nomic-embed-text` 768 chiều (Ollama) |
| Retrieval | `dense`, `bm25`, `hybrid` (RRF), `rerank` (hybrid → cross-encoder MiniLM-L-12), `tfidf` (starter) |
| Agent | `gemini-3.5-flash-lite`, function calling `search` / `answer` / `abstain` ([starter/agent.py](starter/agent.py)) |
| Benchmark | 26 câu: dev 15 + 3 thiếu evidence, test 5 + 3 ([data/questions.json](data/questions.json)) |
| Báo cáo | [report/individual/2A202604106_VuNgocTung.md](report/individual/2A202604106_VuNgocTung.md) |
| Kết quả | [starter/results/](starter/results/): retrieval (`dev`, `test`, thử nghiệm), agent (trace, nhãn, metrics) |

Chạy từ thư mục gốc (Windows, Python 3.12):

```bash
py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
cp starter/.env.example starter/.env          # điền GEMINI_API_KEY
cd starter && docker compose -p day6_student_starter up -d --wait && cd ..
ollama pull nomic-embed-text
.venv/Scripts/python starter/ingest.py        # --reindex nếu volume rỗng (cần PDF tại data/)
.venv/Scripts/python -m unittest discover -s tests -v
.venv/Scripts/python starter/benchmark.py check
.venv/Scripts/python starter/benchmark.py retrieval --split dev --methods tfidf bm25 dense hybrid rerank
.venv/Scripts/python starter/agent.py --split dev --method rerank
.venv/Scripts/python starter/benchmark.py agent starter/results/agent/dev-dense starter/results/agent/dev-rerank
```

Chi tiết từng file và thiết kế agent ở [starter/README.md](starter/README.md).

---

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
