import sqlite3
import os
import numpy as np

# Đường dẫn gốc chứa các collection
base_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db_hdu"
collection_names = ["de_an_tuyen_sinh", "diem_chuan_tuyen_sinh", "hoc_bong_viec_lam", "thong_tin_chung", "thong_tin_dao_tao"]

# Hàm chuyển vector từ embeddings_queue sang embeddings
def transfer_embeddings(collection_name, db_path):
    print(f"Chuyển vector cho collection: {collection_name}")
    
    # Kết nối database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Tạo bảng embeddings nếu chưa có
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id TEXT PRIMARY KEY,
            collection_id TEXT,
            vector BLOB,
            encoding TEXT
        )
    """)

    # Kiểm tra bảng embeddings_queue có tồn tại không
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings_queue'")
    if not cursor.fetchone():
        print(f"Không tìm thấy bảng embeddings_queue trong {db_path}. Bỏ qua collection {collection_name}.")
        conn.close()
        return

    # Lấy tất cả vector từ embeddings_queue
    cursor.execute("SELECT id, vector, encoding, metadata FROM embeddings_queue")
    queued_embeddings = cursor.fetchall()

    if not queued_embeddings:
        print(f"Không có vector trong embeddings_queue cho collection {collection_name}.")
        conn.close()
        return

    # Chuyển vector sang bảng embeddings
    for embedding in queued_embeddings:
        file_id, vector_binary, encoding, metadata = embedding
        cursor.execute("""
            INSERT OR REPLACE INTO embeddings (id, collection_id, vector, encoding)
            VALUES (?, ?, ?, ?)
        """, (file_id, collection_name, vector_binary, encoding))

    # Xóa bảng embeddings_queue sau khi chuyển
    cursor.execute("DROP TABLE IF EXISTS embeddings_queue")

    conn.commit()
    conn.close()
    print(f"Đã chuyển {len(queued_embeddings)} vector cho collection {collection_name}.")

# Lặp qua từng collection và chuyển vector
for collection_name in collection_names:
    db_path = os.path.join(base_dir, collection_name, "chroma.sqlite3")
    if not os.path.exists(db_path):
        print(f"Không tìm thấy file chroma.sqlite3 cho collection {collection_name}.")
        continue
    transfer_embeddings(collection_name, db_path)

print("Hoàn thành chuyển vector cho tất cả collection.")