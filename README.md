# HDU Admission Chatbot

Đây là hệ thống trợ lý ảo tư vấn tuyển sinh cho Trường Đại học Hồng Đức (HDU), xây dựng dựa trên kỹ thuật RAG (Retrieval-Augmented Generation), kết hợp các mô hình ngôn ngữ lớn (LLM) như GPT-4o-mini, LLaMA, Gemini, cùng giao diện người dùng bằng Streamlit.

## Tính năng

- Giao diện chat thời gian thực
- Truy vấn thông minh theo miền dữ liệu chuyên biệt (thông tin chung, đề án tuyển sinh, điểm chuẩn, học bổng, chương trình đào tạo…)
- Hỗ trợ nhiều mô hình ngôn ngữ: GPT-4o-mini (OpenAI), LLaMA (local), Gemini (Google)
- Hệ thống nhúng văn bản sử dụng OpenAI hoặc HuggingFace Embeddings
- Vector database dùng ChromaDB, dễ mở rộng và cập nhật dữ liệu
- Tự động phân loại câu hỏi để chọn cơ sở dữ liệu phù hợp
- Giao diện quản lý cuộc trò chuyện, cho phép tiếp tục hoặc tạo mới
- Cấu hình mô hình tùy ý ngay trong giao diện Streamlit

## Kiến trúc hệ thống

- `streamlit_app.py`: Giao diện người dùng (frontend)
- `backend.py`: Xử lý truy vấn người dùng, lấy context phù hợp, tạo prompt
- `chatbot.py`: Load mô hình LLaMA fine-tuned, generate câu trả lời
- `model.py`: Truy vấn ChromaDB, phân loại miền, tạo QA chain
- `config.json`: File cấu hình API key (OpenAI, HuggingFace)

## Cài đặt

1. Clone repo:

```bash
git clone https://github.com/yourusername/hdu-chatbot.git
cd hdu-chatbot
```

2. Chạy ứng dụng: streamlit run streamlit_app.py
Yêu cầu hệ thống
Python >= 3.10

GPU (khuyến nghị nếu sử dụng LLaMA local)

Các thư viện chính: streamlit, langchain, transformers, torch, peft, chromadb, huggingface_hub, langchain_openai, langchain_community, langchain_google_genai

Tùy chọn mô hình
LLM:
Chọn llama: Mô hình LLaMA-2-7b-chat chạy local (fine-tuned)


Embedding:
Chọn HuggingFace Embeddings (default: all-MiniLM-L6-v2)

Dữ liệu & thư mục quan trọng
chroma_db_hdu/: Thư mục chứa vector database cho từng domain

models1/: Thư mục chứa adapter model (nếu dùng LLaMA local)

Yêu cầu các file: adapter_config.json, adapter_model.safetensors, tokenizer.json, tokenizer_config.json, special_tokens_map.json

Cách hoạt động
Người dùng nhập câu hỏi

Hệ thống phân loại miền thông tin phù hợp

Vector DB tương ứng được truy vấn để lấy context

Prompt được tạo ra (gồm context + lịch sử hội thoại)

LLM sinh câu trả lời phù hợp

Câu trả lời được hiển thị trên giao diện cùng với nguồn (nếu có)

Ghi chú
Nếu dùng mô hình llama, bạn cần đảm bảo thư mục models1 chứa đầy đủ các file model và tokenizer

Có thể mở rộng thêm miền dữ liệu, vector store và tùy biến mô hình dễ dàng

Tác giả
Dự án được phát triển phục vụ tuyển sinh tại Đại học Hồng Đức (HDU). Bạn có thể sử dụng và tùy biến cho các mục đích tư vấn giáo dục khác.
