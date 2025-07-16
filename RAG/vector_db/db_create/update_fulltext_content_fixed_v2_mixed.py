import os
import sqlite3
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

# Đường dẫn gốc chứa các collection và dữ liệu
base_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db_hdu"
data_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\data_hdu"
collection_names = ["de_an_tuyen_sinh", "diem_chuan_tuyen_sinh", "hoc_bong_viec_lam", "thong_tin_chung", "thong_tin_dao_tao"]

# Đường dẫn đến Tesseract (nếu chưa thêm vào PATH)
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
                text += pytesseract.image_to_string(img, lang='vie') + "\n"
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

# Hàm cập nhật nội dung gốc vào bảng embedding_fulltext_search_content
def update_fulltext_content(collection_name, db_path):
    print(f"Cập nhật nội dung gốc cho collection: {collection_name}")

    # Kết nối database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tạo bảng embedding_fulltext_search_content nếu chưa có
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embedding_fulltext_search_content (
            id TEXT PRIMARY KEY,
            content TEXT
        )
    """)

    # Đọc file từ thư mục con tương ứng trong data_dir
    collection_data_dir = os.path.join(data_dir, collection_name)
    if not os.path.exists(collection_data_dir):
        print(f"Không tìm thấy thư mục dữ liệu cho collection {collection_name}.")
        conn.close()
        return

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

        # Lưu nội dung vào bảng embedding_fulltext_search_content
        file_id = f"{collection_name}_{filename}"
        cursor.execute("""
            INSERT OR REPLACE INTO embedding_fulltext_search_content (id, content)
            VALUES (?, ?)
        """, (file_id, text))

    conn.commit()
    conn.close()
    print(f"Đã cập nhật nội dung gốc cho collection {collection_name}.")

# Lặp qua từng collection và cập nhật nội dung
for collection_name in collection_names:
    db_path = os.path.join(base_dir, collection_name, "chroma.sqlite3")
    if not os.path.exists(db_path):
        print(f"Không tìm thấy file chroma.sqlite3 cho collection {collection_name}.")
        continue
    update_fulltext_content(collection_name, db_path)

print("Hoàn thành cập nhật nội dung gốc cho tất cả collection.")