from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field
from typing import List, Optional, Any
import os
import json
import logging
import warnings
warnings.filterwarnings('ignore')
from transformers import pipeline
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
import requests
from search_web import test_search
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def create_prompt():
    template = """Bạn là giáo viên phòng tư vấn tuyển sinh của trường Đại học Hồng Đức.
    Sử dụng thông tin sau đây để trả lời câu hỏi một cách chính xác và ngắn gọn, truyền tải đầy đủ thông tin bạn nhận được.
    Chỉ sử dụng thông tin từ context dưới đây. Không sử dụng bất kỳ thông tin nào ngoài context, đặc biệt không sử dụng thông tin về Đại học Bách Khoa Hà Nội hoặc các trường khác.
    Hãy nói với vai trò là một thầy cô giáo trong trường, sử dụng ngôn ngữ thân thiện và dễ hiểu.
    
    Nếu không tìm thấy thông tin trong context hoặc thông tin không đầy đủ, hãy trả lời:
    "Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question", "web_url"])
    return prompt
# Cấu hình logging chỉ cho __main__
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        # Sử dụng os.path.normpath để chuẩn hóa đường dẫn theo hệ điều hành
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_model(api_key, model_type="gpt-4o-mini"):
    try:
        if model_type == "gemini":
            return ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=0.5,
                convert_system_message_to_human=True
            )
        elif model_type == "gpt-4o-mini":
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                openai_api_key=api_key
            )
            
        elif model_type == "llama":
            login(token=api_key)
            
            pipe = pipeline(
                "text-generation",
                model="meta-llama/Llama-2-7b-chat-hf",
                token=api_key,
                temperature=0.5,
                top_p=0.95,
                repetition_penalty=1.15,
                max_new_tokens=4096
            )
            
            return HuggingFacePipeline(pipeline=pipe)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def classify_question_domain(question):
    """
    Phân loại câu hỏi thuộc miền nào
    Returns: tên collection trong ChromaDB
    """
    # Định nghĩa từ khóa cho từng miền
    domains = {
        "thong_tin_chung": ["thông tin chung", "giới thiệu", "tổng quan", "trường", "hồng đức", "đại học hồng đức"],
        "de_an_tuyen_sinh": ["đề án", "đề án tuyển sinh", "chỉ tiêu", "phương thức", "xét tuyển", "tuyển sinh", "thời gian xét tuyển"],
        "diem_chuan_tuyen_sinh": [
            "điểm chuẩn", "điểm trúng tuyển", "điểm đầu vào", 
            "điểm thi", "điểm xét tuyển",
            "điểm chuẩn 2024", "điểm trúng tuyển 2024",  # Thêm từ khóa kết hợp
            "năm 2024", "2024"
        ],
        "hoc_bong_viec_lam": ["hướng nghiệp", "nghề nghiệp", "định hướng"],
        "thong_tin_dao_tao": ["thông tin đào tạo", "đào tạo", "chương trình học", "khoa", "ngành học", "đội ngũ giảng viên", "nhóm ngành sư phạm"]
    }
    
    question = question.lower()
    
    # Kiểm tra từng domain
    for domain, keywords in domains.items():
        if any(keyword in question for keyword in keywords):
            print("Tên domain là: ", domain)
            return domain
    
    # Nếu không tìm thấy domain nào phù hợp, trả về "Q_A"
    print("Không phân loại được domain cụ thể, sử dụng domain mặc định: Q_A")
    return "Q_A"
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

def load_vector_db(api_key, collection_name, embedding_model_choice):
    try:
        config = load_config()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(current_dir, "chroma_db_hdu")
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        # Ánh xạ tên collection
        collection_mapping = {
            "thong_tin_chung": "thong_tin_chung",
            "de_an_tuyen_sinh": "de_an_tuyen_sinh",
            "diem_chuan_tuyen_sinh": "diem_chuan_tuyen_sinh",
            "hoc_bong_viec_lam": "hoc_bong_viec_lam",
            "thong_tin_dao_tao": "thong_tin_dao_tao",
            "Q_A": "Q_A"  # Thêm mapping cho domain Q_A
        }
        
        mapped_collection_name = collection_mapping.get(collection_name, collection_name)
        
        # Chọn embedding model
        if embedding_model_choice == "HuggingFace Embeddings":
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        else:
            embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        
        # Khởi tạo ChromaDB
        # Nếu collection_name là "Q_A", chỉ sử dụng qa_db
        if collection_name == "Q_A":
            domain_db = None
            qa_db = Chroma(
                collection_name="Q_A",
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        else:
            domain_db = Chroma(
                persist_directory=persist_directory + f'\{mapped_collection_name}',
                embedding_function=embedding_model
            )
            
            qa_db = Chroma(
                collection_name="Q_A",
                persist_directory=persist_directory,
                embedding_function=embedding_model
            )
        
        print("mapped_collection_name: ", mapped_collection_name)
        
        # Kiểm tra số lượng tài liệu
        domain_count = domain_db._collection.count() if domain_db else 0
        qa_count = qa_db._collection.count() if qa_db else 0
        print(f"Domain DB ({mapped_collection_name}) count: {domain_count}")
        print(f"QA DB count: {qa_count}")
        
        if domain_count == 0 and collection_name != "Q_A":
            print(f"⚠️ Domain DB ({mapped_collection_name}) rỗng!")
        if qa_count == 0:
            print(f"⚠️ QA DB rỗng!")
        
        return domain_db, qa_db
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise
def get_relevant_chunks(question, domain_db, qa_db, k=3, min_score=0.01):
    try:
        print(f"\n=== Bắt đầu tìm kiếm cho câu hỏi: '{question}' ===")
        
        # Debug: Kiểm tra đầu vào
        print(f"Kiểm tra DB đầu vào - domain_db: {'Có' if domain_db else 'Không'}, qa_db: {'Có' if qa_db else 'Không'}")

        # Thử tìm trong domain_db
        domain_results = []
        if domain_db:
            try:
                print("\n[DOMAIN DB] Thực hiện similarity_search_with_relevance_scores...")
                raw_domain_results = domain_db.similarity_search_with_relevance_scores(question, k=k)
                print(f"Tìm thấy {len(raw_domain_results)} kết quả thô từ domain DB")
                
                domain_results = [(doc, score) for doc, score in raw_domain_results if score >= min_score]
                print(f"Còn {len(domain_results)} kết quả sau khi lọc (score >= {min_score})")
                
                if not domain_results:
                    print("⚠️ Không có kết quả nào từ domain DB đạt ngưỡng score")
                    print("Top 3 scores thô:", [(score, doc.page_content[:50]+'...') for doc, score in raw_domain_results[:3]])
            except Exception as e:
                print(f"❌ Lỗi khi tìm trong domain DB: {str(e)}")
                domain_results = []

        # Thử tìm trong qa_db
        qa_results = []
        if qa_db:
            try:
                print("\n[QA DB] Thực hiện similarity_search_with_relevance_scores...")
                raw_qa_results = qa_db.similarity_search_with_relevance_scores(question, k=k)
                print(f"Tìm thấy {len(raw_qa_results)} kết quả thô từ QA DB")
                
                qa_results = [(doc, score) for doc, score in raw_qa_results if score >= min_score]
                print(f"Còn {len(qa_results)} kết quả sau khi lọc (score >= {min_score})")
                
                if not qa_results:
                    print("⚠️ Không có kết quả nào từ QA DB đạt ngưỡng score")
                    print("Top 3 scores thô:", [(score, doc.page_content[:50]+'...') for doc, score in raw_qa_results[:3]])
            except Exception as e:
                print(f"❌ Lỗi khi tìm trong QA DB: {str(e)}")
                qa_results = []

        # Chọn kết quả phù hợp nhất
        results = []
        if domain_results:
            print("\nƯu tiên sử dụng kết quả từ domain DB")
            results = domain_results
        elif qa_results:
            print("\nSử dụng kết quả từ QA DB do domain DB không có kết quả")
            results = qa_results
        else:
            print("\n⛔ Không tìm thấy kết quả nào từ cả hai DB")
            return []

        # Chuyển đổi kết quả
        documents = []
        print("\nCác kết quả đã lọc:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n--- Kết quả {i} ---")
            print(f"Score: {score:.4f}")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
            metadata = doc.metadata.copy()
            metadata.update({
                'score': f"{score:.4f}",
                'doc_id': doc.metadata.get('doc_id', None)
            })
            
            documents.append(Document(
                page_content=doc.page_content,
                metadata=metadata
            ))

        print(f"\n=== Kết thúc tìm kiếm, trả về {len(documents)} documents ===")
        return documents
        
    except Exception as e:
        print(f"\n🔥 LỖI NGHIÊM TRỌNG trong get_relevant_chunks: {str(e)}")
        logger.error(f"Error getting relevant chunks: {str(e)}", exc_info=True)
        return []
def process_chunks(docs, all_docs, selected_chunks):
    """
    Helper function để xử lý chunks từ một database
    """
    try:
        doc_ids = all_docs.get('ids', [])
        
        for doc in docs:
            doc_id = doc.metadata.get('doc_id')
            if not doc_id:
                continue
                
            try:
                current_index = doc_ids.index(doc_id)
                selected_chunks.add(current_index)
                
                # Thêm chunk trước
                if current_index > 0:
                    selected_chunks.add(current_index - 1)
                    
                # Thêm chunk sau
                if current_index < len(doc_ids) - 1:
                    selected_chunks.add(current_index + 1)
            except ValueError:
                continue
    except Exception as e:
        logger.warning(f"Error processing chu1  nks: {str(e)}")
        # Không raise exception để tránh làm gián đoạn luồng chính

class CustomRetriever(BaseRetriever, BaseModel):
    domain_db: Any = Field(default=None)
    qa_db: Any = Field(default=None)
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, domain_db, qa_db):
        super().__init__()
        self.domain_db = domain_db
        self.qa_db = qa_db

    def get_relevant_documents(self, query: str) -> List[Document]:
        return get_relevant_chunks(query, self.domain_db, self.qa_db)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError("Async retrieval not implemented")

def create_qa_chain(domain_db, qa_db, model, prompt, web_url):
    try:
        retriever = CustomRetriever(domain_db, qa_db)
        qa_chain = RetrievalQA.from_chain_type(
            llm=model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt.partial(web_url=web_url)
            }
        )
        return qa_chain
    except Exception as e:
        logger.error(f"Error creating QA chain: {str(e)}")
        raise


def get_answer(question, model_type="llama", embedding_model_choice="HuggingFace Embeddings"):
    try:
        config = load_config()
        api_key = config.get('OPENAI_API_KEY') if model_type == "gpt-4o-mini" else config.get('HUGGINGFACE_TOKEN')
        
        if not api_key:
            raise Exception(f"API key not found for {model_type} in config.json")
        
        domain = classify_question_domain(question)
        print(f"\nDomain được phân loại: {domain}")
        model = load_model(api_key, model_type)
        domain_db, qa_db = load_vector_db(api_key, domain, embedding_model_choice)
        web_url = test_search(question)

        # Lấy documents và kiểm tra
        documents = get_relevant_chunks(question, domain_db, qa_db)
        print('TÔI Ở ĐÂY 3')
        print("\nCác chunks được tìm thấy:")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Nội dung: {doc.page_content}")
            print("Metadata:")
            for key, value in doc.metadata.items():
                print(f"  {key}: {value}")
                
        if not documents:
            fallback_message = (
                f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
            )
            return {
                "answer": fallback_message,
                "source_documents": [],
                "domain": domain
            }
            
        prompt = create_prompt()
        qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
        
        # Thêm xử lý kết quả
        result = qa_chain({"query": question})
        answer = result["result"]
        
        # In ra câu trả lời
        print("\nCâu trả lời:")
        print(answer)
        print("\nNguồn tham khảo:")
        for doc in result.get("source_documents", []):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'Unknown')
            score = doc.metadata.get('score', 'Unknown')
            print(f"- Nguồn: {source}")
            print(f"  Trang: {page}")
            print(f"  Độ tương đồng: {score}")
            print(f"  Nội dung: {doc.page_content[:200]}...")
        
        return {
            "answer": answer,
            "source_documents": result.get("source_documents", []),
            "domain": domain
        }
        
    except Exception as e:
        logger.error(f"Error getting answer: {str(e)}")
        raise

def get_db_path():
    """Trả về đường dẫn chuẩn đến thư mục chroma_db_hdu"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "chroma_db_hdu")
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(db_path):
        os.makedirs(db_path)
    return os.path.normpath(db_path)

if __name__ == "__main__":
    print("\nChọn mô hình:")
    print("1. GPT-4o-mini")
    print("2. Llama-2-7b-chat")
    
    model_choice = input("\nChọn model (1-2): ")
    model_type = "gpt-4o-mini" if model_choice == "1" else "llama"
    
    while True:
        question = input("\nCâu hỏi: ")
        if question.lower() in ['quit', 'q', 'exit']:
            break
            
        try:
            result = get_answer(question, model_type)
            
        except Exception as e:
            print(f"Error: {str(e)}")
            logger.error(f"Error in main loop: {str(e)}")