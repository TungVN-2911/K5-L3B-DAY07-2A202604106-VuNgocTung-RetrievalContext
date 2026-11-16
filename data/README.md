# Định dạng dữ liệu

Các file `example_*.json` chỉ minh họa định dạng, không đủ để nộp bài. Dữ liệu thật là JSON array UTF-8; nếu dùng định dạng khác, ghi cách chuyển trong README.

- **Chunk:** `id`, `text`, `source_id`, `source`, `page`, `metadata`. ID phải duy nhất; giữ phiên bản hoặc ngày nếu có.
- **Câu có đáp án:** `id`, `split` (`dev` hoặc `test`), `question`, `expected_answer`, `gold`: danh sách `{source_id, page, evidence}`. Với schema mẫu, mỗi evidence phải nằm trọn trong một chunk đúng nguồn và trang; các phần tử `gold` là evidence thay thế. Với câu multi-hop cần nhiều đoạn, bạn có thể mở rộng schema, ghi rõ nhóm evidence bắt buộc và cách chấm. Có thể thêm `filter` do người dùng cung cấp trước.
- **Câu thiếu evidence:** `id`, `split`, `question`, `answerable: false`, `expected_behavior: "search_and_abstain"`, `absence_reason`. Xem `agent_cases_example.json`.

Không đưa `gold` hoặc nhãn đáp án vào model và retrieval tool. Ghi nguồn dữ liệu và quyền sử dụng. Cách chuyển một số định dạng Day 6: [starter/README.md](../starter/README.md).

## Benchmark bài làm

`questions.json` là benchmark Day 7 (26 câu), dùng luôn làm `cases.json` cho `agent_metrics.py` vì có sẵn `id`, `split`, `answerable`.

| Split | Có đáp án | Thiếu evidence |
|---|---:|---:|
| dev | 15 (q01–q15) | 3 (u01–u03) |
| test | 5 (q16–q20) | 3 (u04–u06) |

- **q01–q20** chuyển từ `source_questions.jsonl` (benchmark Day 6), giữ nguyên split. Test Day 6 chỉ dùng để báo kết quả sau khi chốt chunking, chưa dùng để chỉnh agent Day 7. Hai chỗ sửa ghi trong trường `note`: q11 bỏ trang 133 (evidence chỉ ở p132); q13 rút evidence về câu cốt lõi vì đoạn gốc vắt qua hai chunk fixed500-50.
- **u01–u06** là câu gần chủ đề nhưng thiếu evidence. `absence_check.absent_patterns` được dò trên text NFKC của toàn bộ 384 trang PDF; `related_pages` là trang nói về chủ đề gần nhất. Giới hạn: chữ nằm trong ảnh chưa OCR nên không kiểm được tự động.
- Kiểm tra: `python starter/benchmark.py` (schema, số câu tối thiểu, gold nằm trọn trong chunk đúng trang, absence).
