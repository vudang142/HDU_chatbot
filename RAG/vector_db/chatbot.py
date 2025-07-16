import os
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import logging
import sys
from typing import List, Tuple, Set
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel, PeftConfig
from langchain_openai import OpenAIEmbeddings
from model import get_relevant_chunks, load_vector_db, classify_question_domain
# Tắt logging
logging.disable(logging.CRITICAL)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vector_db.model import get_answer, load_config
from search_web import test_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    _instance = None
    _pipe = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.__init__()
        return cls._instance
    
    def __init__(self):
        if self._pipe is not None:
            return
        self._pipe = self.load_fine_tuned_model()

    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned GPU memory")

    def load_fine_tuned_model(self):
        try:
            if torch.cuda.is_available():
                self.clean_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "models")
            
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                temperature=0.3,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation=True,
                max_new_tokens=4096
            )
            
            return pipe
            
        except Exception as e:
            logger.error(f"Error loading fine-tuned model: {str(e)}")
            raise

def get_data_with_context(question: str, embedding_model_choice) -> Tuple[str, str, Set[str]]:
    """Lấy dữ liệu từ database hoặc mô hình Hugging Face kèm context mở rộng"""
    try:
        # 1. Khởi tạo và cấu hình
        config = load_config()
        huggingface_token = config.get("HUGGINGFACE_TOKEN")
        if not huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in config")

        # 2. Khởi tạo mô hình embedding Hugging Face
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # Thay đổi thành "cuda" nếu có GPU
        )

        # 3. Phân loại domain và tìm kiếm web
        domain = classify_question_domain(question)
        web_url = test_search(question)
        
        # 4. Load vector databases
        domain_db, qa_db = load_vector_db(huggingface_token, domain, "HuggingFace Embeddings")
        
        # 5. Lấy chunks với relevance scores
        documents = get_relevant_chunks(question, domain_db, qa_db)
        
        if not documents:
            print("không có documents")
            return None, web_url, None
        
        # 6. Sắp xếp tài liệu theo score
        sorted_documents = sorted(documents, key=lambda x: float(x.metadata.get('score', 0)), reverse=True)
        
        # Tạo danh sách các tóm tắt nội dung của tài liệu để loại bỏ trùng lặp
        document_summaries = {}
        for doc in sorted_documents:
            # Lấy 100 ký tự đầu tiên làm tóm tắt để so sánh trùng lặp
            content_summary = doc.page_content[:100].strip()
            if content_summary not in document_summaries:
                document_summaries[content_summary] = doc
        
        # Chuyển thành danh sách duy nhất
        unique_documents = list(document_summaries.values())
        
        # Giới hạn số lượng tài liệu
        top_n = min(3, len(unique_documents))
        top_documents = unique_documents[:top_n]

        expanded_context_parts = []
        sources = set()
        max_doc_length = 500
        max_context_tokens = 1500
        total_tokens = 0

        for i, doc in enumerate(top_documents):
            current_metadata = doc.metadata
            source = current_metadata.get('source', 'Unknown')
            page = current_metadata.get('page', 'Unknown')
            score = current_metadata.get('score', 'Unknown')
            
            # Thêm source vào danh sách
            source_info = f"{source} (trang {page})" if page != 'Unknown' else source
            sources.add(source_info)
            
            # Cắt ngắn nội dung tài liệu
            main_content = doc.page_content[:max_doc_length]
            
            # Tạo context chỉ với nội dung chính
            context_part = f"Đoạn {i+1} (độ liên quan: {score}) từ {source_info}: {main_content}"
            
            # Ước tính số token
            token_estimate = len(context_part) // 4
            if total_tokens + token_estimate > max_context_tokens:
                print(f"DEBUG: Stopping at document {i+1} to avoid exceeding {max_context_tokens} tokens")
                break
            
            expanded_context_parts.append(context_part)
            total_tokens += token_estimate

        # 8. Tạo context tổng hợp
        context = "\n\n".join(expanded_context_parts)
        print(f"DEBUG: Estimated total tokens: {total_tokens}")
        
        return context, web_url, sources

    except Exception as e:
        print(f"DEBUG: Error in get_data_with_context: {str(e)}")
        logger.error(f"Error in get_data_with_context: {str(e)}")
        return None, None, None

def generate_response(prompt: str) -> str:
    try:
        model_manager = ModelManager()
        model_manager.clean_gpu_memory()
        
        pipe = model_manager._pipe
        response = pipe(prompt)[0]['generated_text']
        
        model_manager.clean_gpu_memory()
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise

def create_prompt_llama(question: str, context: str, max_tokens: int = 2000) -> str:
    """
    Phiên bản tối ưu với kiểm tra độ dài token
    Args:
        question: Câu hỏi
        context: Context
        max_tokens: Giới hạn token cho prompt
    """
    base_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Bạn là giáo viên tư vấn tuyển sinh Đại học Hồng Đức. Ưu tiên sử dụng thông tin trong context. Nếu không có, hãy trả lời dựa trên kiến thức của bạn một cách phù hợp và thân thiện.

1. Đọc kỹ thông tin trong context
2. Trả lời câu hỏi dựa CHÍNH XÁC trên thông tin có sẵn
3. Không bịa đặt hay suy đoán thông tin không có trong context
4. Trả lời với giọng điệu thân thiện, chuyên nghiệp
5. Nếu không tìm thấy thông tin phù hợp, hướng dẫn tìm thêm trên website chính thức

QUAN TRỌNG - Định dạng câu trả lời:
- Sử dụng bullet points (•) và xuống dòng giữa các ý trong danh sách
- Sử dụng số thứ tự (1., 2., 3.) cho các bước/phương thức và xuống dòng giữa các ý
- Xuống dòng giữa các ý chính
- In đậm tên các phương thức/chương trình quan trọng
- Tách thành đoạn ngắn, dễ đọc
### Input:
Context: {context}
Câu hỏi: {question}

### Response:"""
    
    # Ước tính token (1 token ≈ 4 ký tự tiếng Việt)
    base_length = len(base_prompt.format(context="", question=question)) // 4
    question_length = len(question) // 4
    available_for_context = max_tokens - base_length - question_length - 100  # buffer
    
    # Cắt context nếu quá dài
    if len(context) // 4 > available_for_context:
        context = context[:available_for_context * 4] + "..."
        print(f"DEBUG: Context được cắt xuống {available_for_context * 4} ký tự")
    
    return base_prompt.format(context=context, question=question)


def main():
    try:
        while True:
            question = input("\nNhập câu hỏi (hoặc 'q' để thoát): ")
            if question.lower() == 'q':
                break
            
            # Lấy dữ liệu và context mở rộng
            context, web_url, sources = get_data_with_context(question, model_type="llama", embedding_model_choice="HuggingFace Embeddings")
            
            if context is None:
                print(f"\nKhông tìm thấy thông tin phù hợp. Bạn có thể tham khảo thêm tại: {web_url}")
                continue
            
            # Tạo prompt và generate câu trả lời
            prompt = create_prompt_llama(question, context, max_tokens=2000)
            response = generate_response(prompt)
            
            # In kết quả
            print("\nCâu trả lời:")
            print("-" * 50)
            print(response)
            print("-" * 50)
            
            # if sources:
            #     print("\nNguồn tham khảo:")
            #     for source in sources:
            #         print(f"- {source}")
            
            # if web_url:
                # print(f"\nTham khảo thêm tại: {web_url}")
                
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        print(f"Lỗi: {str(e)}")

if __name__ == "__main__":
    main()