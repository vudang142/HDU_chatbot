# Cài đặt các thư viện cần thiết
# pip install langchain chromadb sentence-transformers pypdf langchain_community pdf2image pytesseract Pillow

import os
import glob
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pytesseract
from PIL import Image
import pdf2image

# Cấu hình thư mục lưu trữ database và dữ liệu
CHROMA_DB_DIRECTORY = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db_hdu\de_an_tuyen_sinh"
DATA_DIRECTORY = r"E:\document\lab\DATN_ChatbotHUST\RAG\data_hdu\de_an_tuyen_sinh"

# 1. Khởi tạo embedding model từ HuggingFace
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 2. Hàm chuẩn hóa văn bản tiếng Việt sau OCR
def normalize_vietnamese_text(text):
    """Chuẩn hóa văn bản tiếng Việt, sửa các lỗi ký tự thường gặp từ OCR."""
    if not text:
        return text
    
    # Bước 1: Chuyển đổi các ký tự bị nhận diện sai thường gặp
    replacements = {
        r"TRU'ŌNG": "TRƯỜNG",
        r"DÈ ÁN": "ĐỀ ÁN",
        r"Sự phạm": "Sư phạm",
        r"ĐĀO TAO": "Đào tạo",
        r"Vặt lý": "Vật lý",
        r"Sịnh học": "Sinh học",
        r"Ngũ vãn": "Ngữ văn",
        r"Địa chì": "Địa chỉ",
        r"Đãng ký": "Đăng ký",
        r"Hòng": "Hồng",
        r"ĐAI HQC": "ĐẠI HỌC",
        r"HÔNG ĐÚC": "HỒNG ĐỨC",
        r"SU'ŌNG": "SƯỜNG",
        r"Ō": "Ơ",
        r"Ũ": "Ữ",
        r"Ā": "Ă",
        r"'": "",  # Loại bỏ ký tự thừa
    }
    
    normalized_text = text
    for wrong, correct in replacements.items():
        normalized_text = re.sub(wrong, correct, normalized_text, flags=re.IGNORECASE)
    
    # Bước 2: Loại bỏ các ký tự thừa hoặc không cần thiết
    normalized_text = re.sub(r"\s{2,}", " ", normalized_text)  # Loại bỏ khoảng trắng thừa
    normalized_text = re.sub(r"footnotes \d+\.\d+…", "", normalized_text)  # Loại bỏ chuỗi footnotes không cần thiết
    normalized_text = re.sub(r"3\s+3\s+3\s+…", "", normalized_text)  # Loại bỏ chuỗi lặp "3 3 3 ..."
    
    # Bước 3: Chuẩn hóa dấu câu (ví dụ: thay thế các ký tự không chuẩn)
    normalized_text = normalized_text.replace("…", "...").replace("–", "-")
    
    return normalized_text.strip()

# 3. Hàm trích xuất văn bản từ PDF bằng OCR
def extract_text_with_ocr(pdf_path, page_num):
    """Chuyển trang PDF thành ảnh và trích xuất văn bản bằng OCR, sau đó chuẩn hóa."""
    try:
        images = pdf2image.convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        if images:
            text = pytesseract.image_to_string(images[0], lang='vie')
            # Chuẩn hóa văn bản sau OCR
            normalized_text = normalize_vietnamese_text(text)
            return normalized_text
        return ""
    except Exception as e:
        print(f"  - Lỗi OCR trang {page_num} của {os.path.basename(pdf_path)}: {str(e)}")
        return ""

# 4. Hàm đọc và xử lý tài liệu (PDF và TXT)
def process_documents(directory_path):
    """Đọc và xử lý các file PDF/TXT trong thư mục, áp dụng OCR và chuẩn hóa nếu cần."""
    documents = []
    
    # Xử lý các file PDF
    for pdf_path in glob.glob(os.path.join(directory_path, "*.pdf")):
        try:
            print(f"Đang xử lý PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pdf_docs = loader.load()
            valid_pages = 0
            
            for i, doc in enumerate(pdf_docs):
                page_content = doc.page_content.strip()
                
                # Nếu trang không có nội dung, thử OCR
                if not page_content:
                    text = extract_text_with_ocr(pdf_path, i + 1)
                    if text:
                        doc.page_content = text
                        documents.append(doc)
                        valid_pages += 1
                    else:
                        print(f"  - Trang {i+1} không nhận diện được nội dung, bỏ qua")
                else:
                    # Chuẩn hóa nội dung nếu có
                    doc.page_content = normalize_vietnamese_text(page_content)
                    documents.append(doc)
                    valid_pages += 1
            
            print(f"  → Đọc được {valid_pages}/{len(pdf_docs)} trang từ file {os.path.basename(pdf_path)}")
        
        except Exception as e:
            print(f"Lỗi xử lý file PDF {pdf_path}: {str(e)}")
    
    # Xử lý các file TXT
    for txt_path in glob.glob(os.path.join(directory_path, "*.txt")):
        try:
            print(f"Đang xử lý TXT: {txt_path}")
            loader = TextLoader(txt_path, encoding="utf-8")
            txt_docs = loader.load()
            
            if txt_docs and txt_docs[0].page_content.strip():
                # Chuẩn hóa nội dung file TXT
                for doc in txt_docs:
                    doc.page_content = normalize_vietnamese_text(doc.page_content)
                documents.extend(txt_docs)
                print(f"  → Đọc thành công file {os.path.basename(txt_path)}")
            else:
                print(f"  - File {os.path.basename(txt_path)} rỗng, bỏ qua")
        
        except UnicodeDecodeError:
            print(f"  - Lỗi encoding với utf-8, thử các encoding khác...")
            for encoding in ["latin-1", "cp1252", "ISO-8859-1"]:
                try:
                    loader = TextLoader(txt_path, encoding=encoding)
                    txt_docs = loader.load()
                    if txt_docs and txt_docs[0].page_content.strip():
                        for doc in txt_docs:
                            doc.page_content = normalize_vietnamese_text(doc.page_content)
                        documents.extend(txt_docs)
                        print(f"  → Đọc thành công với encoding {encoding}")
                        break
                except Exception:
                    continue
        except Exception as e:
            print(f"Lỗi xử lý file TXT {txt_path}: {str(e)}")
    
    print(f"Tổng cộng đọc được {len(documents)} tài liệu")
    return documents

# 5. Hàm chia nhỏ tài liệu thành các chunk
def split_documents(documents):
    """Chia nhỏ tài liệu thành các chunk để tạo embeddings."""
    if not documents:
        print("Không có tài liệu để chia nhỏ!")
        return []
    
    try:
        # Tối ưu cho tiếng Việt với chunk_size và separators phù hợp
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Lọc các chunk quá ngắn
        valid_chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) >= 50]
        
        print(f"Đã tạo {len(chunks)} chunks tổng cộng")
        print(f"Lọc bỏ {len(chunks) - len(valid_chunks)} chunks quá ngắn")
        print(f"Còn lại {len(valid_chunks)} chunks hợp lệ")
        
        # In mẫu chunk để kiểm tra
        if valid_chunks:
            print("Mẫu chunk đầu tiên:")
            print("-" * 40)
            print(valid_chunks[0].page_content[:200] + "...")
            print("-" * 40)
        
        return valid_chunks
    
    except Exception as e:
        print(f"Lỗi khi chia nhỏ tài liệu: {str(e)}")
        return []

# 6. Hàm tạo và lưu trữ vector database
def create_vector_db(documents, persist_directory=CHROMA_DB_DIRECTORY):
    """Tạo và lưu trữ vector database bằng Chroma."""
    if not documents:
        print("Không có tài liệu để tạo vector database!")
        return None
    
    try:
        # Tạo vector database
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        
        # Lưu database
        print(f"Đã tạo và lưu vector database tại: {persist_directory}")
        return vectordb
    
    except Exception as e:
        print(f"Lỗi khi tạo vector database: {str(e)}")
        return None

# 7. Hàm truy vấn vector database
def query_vector_db(query_text, k=3):
    """Truy vấn vector database và trả về kết quả."""
    try:
        # Tải vector database
        vectordb = Chroma(
            persist_directory=CHROMA_DB_DIRECTORY,
            embedding_function=embedding_model
        )
        
        # Thực hiện truy vấn
        results = vectordb.similarity_search(query_text, k=k)
        
        print(f"Kết quả cho truy vấn: '{query_text}'")
        for i, doc in enumerate(results):
            print(f"Kết quả {i+1}:")
            print(f"Nội dung: {doc.page_content[:150]}...")
            print(f"Metadata: {doc.metadata}")
            print("-" * 50)
        
        return results
    
    except Exception as e:
        print(f"Lỗi khi truy vấn vector database: {str(e)}")
        return []

# 8. Hàm chính để thực thi quy trình
def main(data_directory=DATA_DIRECTORY):
    """Quy trình chính: Đọc tài liệu, chia nhỏ, tạo vector database, và kiểm tra."""
    # Kiểm tra thư mục dữ liệu
    if not os.path.exists(data_directory):
        print(f"Thư mục {data_directory} không tồn tại!")
        return
    
    # Bước 1: Đọc và xử lý tài liệu
    print("Đang đọc tài liệu...")
    documents = process_documents(data_directory)
    
    if not documents:
        print("Không tìm thấy tài liệu nào để xử lý!")
        return
    
    # Kiểm tra nội dung tài liệu
    print("=== Kiểm tra nội dung tài liệu ===")
    empty_docs = 0
    for i, doc in enumerate(documents[:5]):  # Kiểm tra 5 tài liệu đầu tiên
        content_len = len(doc.page_content.strip())
        print(f"Tài liệu {i+1}: Độ dài nội dung = {content_len}")
        print(f"Metadata: {doc.metadata}")
        if content_len > 0:
            print(f"Nội dung đầu: {doc.page_content[:100]}...")
        else:
            empty_docs += 1
    
    print(f"Tổng số tài liệu rỗng: {empty_docs}/{len(documents)}")
    
    # Lọc tài liệu hợp lệ
    valid_documents = [doc for doc in documents if len(doc.page_content.strip()) > 0]
    print(f"Số tài liệu hợp lệ sau lọc: {len(valid_documents)}/{len(documents)}")
    
    # Bước 2: Chia nhỏ tài liệu
    print("Đang chia nhỏ tài liệu...")
    chunks = split_documents(valid_documents)
    
    if not chunks:
        print("Không có chunks nào được tạo. Dừng lại!")
        return
    
    # Bước 3: Tạo vector database
    print("Đang tạo vector database...")
    vectordb = create_vector_db(chunks)
    
    # Bước 4: Kiểm tra vector database
    if vectordb:
        print("Kiểm tra vector database...")
        query = "Thử nghiệm truy vấn database"
        query_vector_db(query, k=3)

# 9. Thực thi chương trình
if __name__ == "__main__":
    main()
    # Để truy vấn thêm, gọi hàm query_vector_db
    # query_vector_db("Nội dung bạn muốn tìm kiếm")