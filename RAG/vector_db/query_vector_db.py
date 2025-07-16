import os
import re
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from typing import List
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chỉ định thư mục lưu vector DB
CHROMA_DB_DIRECTORY = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db_hdu_2\all_domains"

# Khởi tạo embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def chunk_text_data(file_path: str, chunk_size: int = 500) -> List[dict]:
    """
    Đọc và chia nhỏ dữ liệu từ file text thành các chunk.
    Args:
        file_path: Đường dẫn đến file text (hdu_data_final.txt)
        chunk_size: Kích thước tối đa mỗi chunk (ký tự)
    Returns:
        Danh sách các dict với content và metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Chia nội dung thành các chunk dựa trên tiêu đề hoặc đoạn
        chunks = []
        current_chunk = ""
        current_metadata = {"source": file_path, "type": "text", "section_title": "General"}
        lines = content.splitlines()

        # Regex để xác định tiêu đề (bắt đầu bằng * hoặc các tiêu đề lớn như "I.", "B.")
        title_pattern = re.compile(r'^\s*(?:\*|\w+\.\s+|[IVX]+\.\s+|\*+\s+)(.+?)(?=\s*$|\s*[:])')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Kiểm tra nếu dòng là tiêu đề
            title_match = title_pattern.match(line)
            if title_match:
                # Lưu chunk hiện tại nếu có
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "metadata": current_metadata.copy()
                    })
                    current_chunk = ""
                # Cập nhật tiêu đề mới
                current_metadata["section_title"] = title_match.group(1).strip()
                continue

            # Thêm dòng vào chunk hiện tại
            current_chunk += line + " "
            # Nếu chunk vượt quá kích thước, lưu lại
            if len(current_chunk) > chunk_size:
                chunks.append({
                    "content": current_chunk.strip(),
                    "metadata": current_metadata.copy()
                })
                current_chunk = ""

        # Lưu chunk cuối nếu còn
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "metadata": current_metadata.copy()
            })

        logger.info(f"Chia được {len(chunks)} chunks từ {file_path}")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking text data: {str(e)}")
        raise

def vectorize_text_to_chroma(file_path: str, collection_name: str = "all_domains"):
    """
    Vector hóa dữ liệu text và lưu vào ChromaDB.
    Args:
        file_path: Đường dẫn đến file text
        collection_name: Tên collection trong ChromaDB
    Returns:
        ChromaDB instance
    """
    try:
        # Chuẩn bị dữ liệu
        text_data = chunk_text_data(file_path)

        # Tạo danh sách Document
        documents = [
            Document(
                page_content=data["content"],
                metadata=data["metadata"]
            ) for data in text_data
        ]

        # Tạo và lưu ChromaDB
        if not os.path.exists(CHROMA_DB_DIRECTORY):
            os.makedirs(CHROMA_DB_DIRECTORY)

        chroma_db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=CHROMA_DB_DIRECTORY,
            collection_name=collection_name
        )

        # Lưu vĩnh viễn
        logger.info(f"Vectorized and saved {len(documents)} documents to ChromaDB at {CHROMA_DB_DIRECTORY}")
        return chroma_db

    except Exception as e:
        logger.error(f"Error vectorizing text to ChromaDB: {str(e)}")
        raise

def load_vector_db(collection_name: str = "all_domains"):
    """
    Tải vector database từ đường dẫn đã lưu
    Args:
        collection_name: Tên collection
    Returns:
        ChromaDB instance
    """
    try:
        vectordb = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedding_model,
            collection_name=collection_name
        )
        
        count = vectordb._collection.count()
        logger.info(f"Đã tải Vector DB với {count} document chunks.")
        return vectordb

    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def query_vector_db(query_text: str, k: int = 3):
    """
    Truy vấn vector database để tìm kiếm thông tin tương tự
    Args:
        query_text: Câu truy vấn
        k: Số lượng kết quả trả về
    Returns:
        Danh sách tài liệu tương đồng
    """
    try:
        vectordb = load_vector_db()
        results = vectordb.similarity_search_with_relevance_scores(query_text, k=k)
        
        print(f"\nKết quả cho truy vấn: '{query_text}'")
        print("="*80)
        for i, (doc, score) in enumerate(results, 1):
            print(f"Kết quả {i}:")
            print(f"Độ tương đồng: {score:.4f}")
            print(f"Nội dung: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            print("-" * 80)
        
        return results

    except Exception as e:
        logger.error(f"Error querying vector database: {str(e)}")
        return []

def explore_collection_metadata():
    """
    Xem thông tin về collection và metadata
    """
    try:
        vectordb = load_vector_db()
        collection = vectordb._collection
        ids = collection.get()["ids"]
        print(f"Số lượng documents: {len(ids)}")
        
        if ids:
            print("\nMetadata của 5 document đầu tiên:")
            for i, doc_id in enumerate(ids[:5]):
                metadata = collection.get(ids=[doc_id])["metadatas"][0]
                print(f"Document {i+1} - ID: {doc_id}")
                print(f"Metadata: {metadata}")
                print("-" * 50)

    except Exception as e:
        logger.error(f"Error exploring collection metadata: {str(e)}")

if __name__ == "__main__":
    # Vector hóa dữ liệu từ hdu_data_final.txt
    file_path = r"E:\document\lab\DATN_ChatbotHUST\RAG\data_hdu\hdu_data_final.txt"  # Thay bằng đường dẫn thực tế
    vectorize_text_to_chroma(file_path)
    
    # Kiểm tra metadata
    explore_collection_metadata()
    
    # Thử một số truy vấn mẫu
    query_vector_db("Thông tin chung về trường đại học Hồng Đức", k=2)
    query_vector_db("Các ngành đào tạo", k=2)
    
    # Cho phép người dùng nhập truy vấn
    while True:
        user_query = input("\nNhập câu truy vấn của bạn (hoặc 'q' để thoát): ")
        if user_query.lower() == 'q':
            break
        query_vector_db(user_query, k=3)