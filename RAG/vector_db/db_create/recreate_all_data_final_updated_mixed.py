import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# Khai báo đường dẫn và collection
base_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db_hdu"
data_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\data_hdu"
collection_names = ["de_an_tuyen_sinh", "diem_chuan_tuyen_sinh", "hoc_bong_viec_lam", "thong_tin_chung", "thong_tin_dao_tao"]

# Tải mô hình embedding
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Đường dẫn đến Tesseract (nếu dùng OCR)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Điều chỉnh nếu cần

# Hàm đọc nội dung từ file PDF (hỗ trợ OCR)
def read_pdf(file_path):
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        if text.strip():
            return text
        else:
            print(f"Không đọc được văn bản trực tiếp từ {file_path}, chuyển sang OCR...")
            images = convert_from_path(file_path, dpi=300)
            text = ""
            for img in images:
                text += pytesseract.image_to_string(img, lang='vie')
            return text
    except Exception as e:
        print(f"Lỗi khi đọc PDF {file_path}: {str(e)}")
        return ""

# Hàm đọc nội dung từ file TXT
def read_txt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Lỗi khi đọc TXT {file_path}: {str(e)}")
        return ""

# Tạo và lưu vector vào ChromaDB
for collection_name in collection_names:
    # Tạo thư mục collection nếu chưa tồn tại
    collection_dir = os.path.join(base_dir, collection_name)
    if not os.path.exists(collection_dir):
        os.makedirs(collection_dir)
        print(f"Đã tạo thư mục: {collection_dir}")

    # Đường dẫn đến file chroma.sqlite3
    db_path = os.path.join(base_dir, collection_name, "chroma.sqlite3")

    # Kết nối ChromaDB (giả lập, bạn có thể dùng client ChromaDB)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tạo bảng collections nếu chưa có
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    """)
    cursor.execute("INSERT OR IGNORE INTO collections (id, name) VALUES (?, ?)", (collection_name, collection_name))

    # Tạo bảng embeddings nếu chưa có
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            collection_id TEXT,
            vector BLOB,
            encoding TEXT
        )
    """)

    # Đọc file từ thư mục con tương ứng trong data_dir
    collection_data_dir = os.path.join(data_dir, collection_name)
    if os.path.exists(collection_data_dir):
        for filename in os.listdir(collection_data_dir):
            file_path = os.path.join(collection_data_dir, filename)
            print(f"Đang xử lý file: {file_path}")

            # Đọc nội dung file
            if filename.lower().endswith('.pdf'):
                text = read_pdf(file_path)
            elif filename.lower().endswith('.txt'):
                text = read_txt(file_path)
            else:
                continue

            if not text.strip():
                print(f"Không đọc được nội dung từ {file_path}")
                continue

            # Tạo vector embedding
            embedding = model.encode(text, convert_to_numpy=True)
            print(f"Đã tạo vector cho {filename}, chiều dài: {len(embedding)}")

            # Lưu vector vào bảng embeddings
            vector_binary = embedding.tobytes()
            file_id = f"{collection_name}_{filename}"
            cursor.execute("INSERT INTO embeddings (id, collection_id, vector, encoding) VALUES (?, ?, ?, ?)",
                           (file_id, collection_name, vector_binary, 'FLOAT32'))

    conn.commit()
    conn.close()
    print(f"Đã xử lý xong collection: {collection_name}")