import os
import sqlite3
import json
from tqdm import tqdm

def update_fulltext_content_for_collection(collection_name, base_dir):
    print(f"\nCập nhật nội dung văn bản cho collection: {collection_name}")
    
    # Đường dẫn đến file chroma.sqlite3
    db_path = os.path.join(base_dir, collection_name, "chroma.sqlite3")
    if not os.path.exists(db_path):
        print(f"Không tìm thấy file {db_path}, bỏ qua...")
        return False
    
    # Kết nối đến database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Kiểm tra schema
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    tables = [table[0] for table in tables]
    print(f"Các bảng trong {collection_name}: {tables}")
    
    if "embeddings" not in tables:
        print(f"Không tìm thấy bảng embeddings trong {collection_name}, bỏ qua...")
        conn.close()
        return False
    
    if "embedding_metadata" not in tables:
        print(f"Không tìm thấy bảng embedding_metadata trong {collection_name}, bỏ qua...")
        conn.close()
        return False
    
    if "embedding_fulltext_search_content" not in tables:
        print(f"Không tìm thấy bảng embedding_fulltext_search_content trong {collection_name}, bỏ qua...")
        conn.close()
        return False
    
    # Kiểm tra schema của bảng embedding_metadata
    cursor.execute("PRAGMA table_info(embedding_metadata);")
    meta_columns = cursor.fetchall()
    meta_column_names = [col[1] for col in meta_columns]
    print(f"Các cột trong embedding_metadata: {meta_column_names}")
    
    # Kiểm tra schema của bảng embedding_fulltext_search_content
    cursor.execute("PRAGMA table_info(embedding_fulltext_search_content);")
    fts_columns = cursor.fetchall()
    fts_column_names = [col[1] for col in fts_columns]
    print(f"Các cột trong embedding_fulltext_search_content: {fts_column_names}")
    
    # Truy vấn tất cả bản ghi từ embeddings
    cursor.execute("SELECT rowid FROM embeddings")
    rows = cursor.fetchall()
    print(f"Tìm thấy {len(rows)} bản ghi trong embeddings")
    
    if not rows:
        print("Không có dữ liệu để xử lý, bỏ qua...")
        conn.close()
        return False
    
    # Xử lý từng bản ghi
    updated_count = 0
    for row in tqdm(rows, desc=f"Cập nhật nội dung văn bản {collection_name}"):
        rowid = row[0]
        
        try:
            # Lấy metadata từ embedding_metadata
            cursor.execute(
                "SELECT string_value FROM embedding_metadata WHERE id = ? AND key = 'metadata'",
                (rowid,)
            )
            metadata_row = cursor.fetchone()
            if not metadata_row:
                print(f"Không tìm thấy metadata cho rowid={rowid}")
                continue
            
            metadata = json.loads(metadata_row[0]) if metadata_row[0] else {}
            
            # Lấy nội dung văn bản từ metadata['source']
            text = None
            if 'source' in metadata:
                file_path = metadata['source']
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        print(f"Đã đọc nội dung từ file {file_path} cho rowid={rowid}: {text[:50]}...")
                except Exception as e:
                    print(f"Lỗi khi đọc file {file_path} cho rowid={rowid}: {str(e)}")
            
            # Nếu không đọc được từ file, thử lấy từ các trường khác trong metadata
            if not text:
                for key in metadata:
                    if isinstance(metadata[key], str) and len(metadata[key]) > 10:
                        text = metadata[key]
                        print(f"Lấy nội dung từ metadata['{key}'] cho rowid={rowid}: {text[:50]}...")
                        break
            
            # Nếu vẫn không có nội dung, bỏ qua
            if not text:
                print(f"Không tìm thấy nội dung văn bản cho rowid={rowid}")
                continue
            
            # Kiểm tra xem bản ghi đã tồn tại trong embedding_fulltext_search_content chưa
            cursor.execute(
                "SELECT rowid FROM embedding_fulltext_search_content WHERE rowid = ?",
                (rowid,)
            )
            fts_row = cursor.fetchone()
            
            if fts_row:
                # Cập nhật bản ghi hiện có
                cursor.execute(
                    "UPDATE embedding_fulltext_search_content SET c0 = ? WHERE rowid = ?",
                    (text, rowid)
                )
            else:
                # Thêm bản ghi mới
                cursor.execute(
                    "INSERT INTO embedding_fulltext_search_content (rowid, c0) VALUES (?, ?)",
                    (rowid, text)
                )
            
            updated_count += 1
            
            # Commit theo batch
            if updated_count % 100 == 0:
                conn.commit()
                print(f"Đã cập nhật {updated_count} bản ghi trong {collection_name}")
        
        except Exception as e:
            print(f"Lỗi khi cập nhật nội dung cho rowid={rowid}: {str(e)}")
            continue
    
    # Commit cuối cùng
    conn.commit()
    
    # Kiểm tra số lượng bản ghi trong embedding_fulltext_search_content
    cursor.execute("SELECT COUNT(*) FROM embedding_fulltext_search_content")
    fts_count = cursor.fetchone()[0]
    print(f"Số lượng bản ghi trong embedding_fulltext_search_content: {fts_count}")
    
    conn.close()
    print(f"Hoàn thành cập nhật nội dung văn bản cho {collection_name}!")
    print(f"Đã cập nhật thành công {updated_count} bản ghi")
    return updated_count > 0

# Đường dẫn đến thư mục chứa chroma_db
base_dir = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db"

# Danh sách các collection
collections = ["xac_thuc_chung_chi_ngoai_ngu"]

# Chạy trên tất cả collection
successful_collections = []
for collection in collections:
    success = update_fulltext_content_for_collection(collection, base_dir)
    if success:
        successful_collections.append(collection)

print("\nTổng kết:")
print(f"Các collection xử lý thành công: {successful_collections}")
print(f"Số lượng collection xử lý thành công: {len(successful_collections)}/{len(collections)}")