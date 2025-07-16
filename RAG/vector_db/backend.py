from main import (
    load_config, 
    load_model,
    clear_gpu_memory
)
from model import (
    create_prompt,
    create_qa_chain,
    get_relevant_chunks,
    load_config,
    CustomRetriever,
    load_vector_db,
    classify_question_domain
)
# from test_combined import create_prompt_combined, test_combined, ModelManager, load_config as load_config_combined
from search_web import test_search
from chatbot import generate_response, create_prompt_llama, get_data_with_context  # Nhập các hàm từ chatbot.py
import logging
import os
import torch
import gc

logger = logging.getLogger(__name__)

class ChatBackend:
    def __init__(self):
        # self.qa_chain = None  
        self.current_model = None
        self.config = load_config()
        self.chat_history = []  # Thêm biến lưu lịch sử chat
        
    @staticmethod
    def clean_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("Cleaned GPU memory")

    def setup_qa_chain(self, embedding_model_choice, llm_model_choice="llama"):
        try:
            api_key = self.config.get('HUGGINGFACE_TOKEN')  # Dùng HuggingFace token cho Llama
            if not api_key:
                raise ValueError(f"API key not found for {llm_model_choice}")
                
            model = load_model(api_key, llm_model_choice)
            self.current_model = llm_model_choice
            return model
        except Exception as e:
            logger.error(f"Error setting up QA chain: {str(e)}")
            raise
    def get_file_name(self, file_path):
        if not file_path or file_path == "Unknown":
            return "Unknown"
        # Lấy tên file từ đường dẫn đầy đủ
        base_name = os.path.basename(file_path)
        # Loại bỏ phần mở rộng file
        name_without_ext = os.path.splitext(base_name)[0]
        # Chuyển dấu gạch dưới thành khoảng trắng và viết hoa chữ đầu
        return name_without_ext.replace('_', ' ').title()

    def get_domain_name(self, domain):
        domain_names = {
            "thong_tin_chung": "thong_tin_chung",
            "de_an_tuyen_sinh": "de_an_tuyen_sinh",
            "diem_chuan_tuyen_sinh": "diem_chuan_tuyen_sinh",
            "hoc_bong_viec_lam": "hoc_bong_viec_lam",
            "thong_tin_dao_tao": "thong_tin_dao_tao",
        }
        return domain_names.get(domain, domain)

    def get_chat_response(self, query, model_choice=None, embedding_model_choice=None):
        if model_choice == "gpt-4o-mini":
            try:
                api_key = self.config.get('OPENAI_API_KEY') if model_choice == "gpt-4o-mini" else self.config.get('HUGGINGFACE_TOKEN')
                
                domain = classify_question_domain(query)
                domain_db, qa_db = load_vector_db(api_key, domain, embedding_model_choice)

                # Kiểm tra số lượng tài liệu trong domain_db và qa_db
                domain_doc_count = domain_db._collection.count() if hasattr(domain_db, '_collection') else 0
                qa_doc_count = qa_db._collection.count() if hasattr(qa_db, '_collection') else 0
                print(f"DEBUG - Domain DB ({domain}) has {domain_doc_count} documents.")
                print(f"DEBUG - QA DB has {qa_doc_count} documents.")
                
                web_url = test_search(query)
                
                # Debug: In ra thông tin documents
                print("\nDEBUG - Documents from databases:")
                if domain_doc_count:
                    domain_docs = get_relevant_chunks(query, domain_db, None)
                if qa_doc_count:    
                    qa_docs = get_relevant_chunks(query, None, qa_db)
                
                print(f"DEBUG - Domain docs count: {len(domain_docs)}")
                if domain_docs:
                    for i, doc in enumerate(domain_docs):
                        print(f"DEBUG - Domain doc {i+1}: {doc.page_content[:100]}...")
                
                print(f"DEBUG - QA docs count: {len(qa_docs)}")
                if qa_docs:
                    for i, doc in enumerate(qa_docs):
                        print(f"DEBUG - QA doc {i+1}: {doc.page_content[:100]}...")
                
                documents = domain_docs + qa_docs
                
                if not documents:
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không thể lấy dữ liệu từ cơ sở dữ liệu. "
                        f"Tuy nhiên, bạn có thể tìm kiếm thông tin ở liên kết sau: {web_url}. "
                        f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": domain
                    }
            # ... (phần còn lại của hàm không thay đổi)
                    
                model = self.setup_qa_chain(llm_model_choice=model_choice)
                prompt = create_prompt()
                qa_chain = create_qa_chain(domain_db, qa_db, model, prompt, web_url)
                
                result = qa_chain({"query": query})
                
                # Tổ chức sources theo domain
                sources = []
                
                # Xử lý documents từ qa_db
                if qa_docs:
                    qa_sources = []
                    for i, doc in enumerate(qa_docs, 1):
                        try:
                            content = doc.page_content.strip() if hasattr(doc, 'page_content') else ''
                            if len(content) > 200:
                                content = content[:200] + "..."
                            
                            score = 0.0
                            metadata = getattr(doc, 'metadata', {})
                            if isinstance(metadata, dict):
                                # Thử lấy score từ metadata nếu có
                                score = metadata.get('score', 0.0)
                                if not isinstance(score, (int, float)):
                                    # Nếu score không phải số, tính theo công thức cũ
                                    score = 0.8259 - (i-1)*0.0045
                            else:
                                # Nếu không có metadata, tính theo công thức cũ
                                score = 0.8259 - (i-1)*0.0045
                            
                            qa_sources.append({
                                "vector": f"Vector {i}",
                                "score": f"{float(score):.4f}",  # Đảm bảo score là string với 4 chữ số thập phân
                                "content": content
                            })
                        except Exception as e:
                            logger.error(f"Error processing qa document {i}: {str(e)}")
                            continue
                    
                    if qa_sources:
                        sources.append({
                            "name": "Q&A Database",
                            "vectors": qa_sources
                        })
                
                # Lấy câu trả lời từ result
                answer = result.get("result", "")  # Sử dụng get() để tránh KeyError
                if not answer and result.get("answer"):  # Thử lấy từ key "answer" nếu "result" không có
                    answer = result["answer"]
                
                response = {
                    "answer": answer,
                    "sources": sources,
                    "domain": domain
                }
                
                # Lưu vào lịch sử
                if not hasattr(self, 'chat_history'):
                    self.chat_history = []
                
                self.chat_history.append({
                    "question": query,
                    "response": response
                })
                
                return response
                
            except Exception as e:
                print(f"Error in get_chat_response: {str(e)}")
                logger.error(f"Error details: {str(e)}")
                return {
                    "answer": "Xin lỗi, đã có lỗi xảy ra khi xử lý câu hỏi của bạn.",
                    "sources": [],
                    "domain": domain
                }
        if model_choice == "llama":
            try:
                self.clean_gpu_memory()  # Dọn dẹp bộ nhớ GPU trước khi xử lý
                domain = classify_question_domain(query)
                logger.info(f"Classified domain: {domain}")
                
                # Khởi tạo domain_db và qa_db
                config = load_config()
                huggingface_token = config.get("HUGGINGFACE_TOKEN")
                if not huggingface_token:
                    raise ValueError("HUGGINGFACE_TOKEN not found in config")
                domain_db, qa_db = load_vector_db(huggingface_token, domain, embedding_model_choice)
                
                # === THÊM PHẦN DEBUG TỪ GPT ===
                # Kiểm tra số lượng tài liệu trong domain_db và qa_db
                domain_doc_count = domain_db._collection.count() if hasattr(domain_db, '_collection') else 0
                qa_doc_count = qa_db._collection.count() if hasattr(qa_db, '_collection') else 0
                print(f"DEBUG - Domain DB ({domain}) has {domain_doc_count} documents.")
                print(f"DEBUG - QA DB has {qa_doc_count} documents.")
                
                web_url = test_search(query)
                domain_docs = []
                qa_docs = []
                # Debug: In ra thông tin documents
                print("\nDEBUG - Documents from databases:")
                if domain_doc_count != 0 and qa_doc_count == 0:
                    domain_docs = get_relevant_chunks(query, domain_db, None)
                    qa_docs = None
                elif qa_doc_count !=0 and domain_doc_count == 0:    
                    qa_docs = get_relevant_chunks(query, None, qa_db)
                    domain_docs = None
                
                if domain_docs and not qa_docs:
                    context_parts = []
                    for i, doc in enumerate(domain_docs, 1):
                        content = doc.page_content.strip()
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        context_parts.append(f"Nguồn {i}: {source}\nNội dung: {content}")
                    context = "\n\n".join(context_parts)
                elif qa_docs and not domain_docs:
                    context_parts = []
                    for i, doc in enumerate(qa_docs, 1):
                        content = doc.page_content.strip()
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        context_parts.append(f"Nguồn {i}: {source}\nNội dung: {content}")
                    context = "\n\n".join(context_parts)
                else:
                    context_parts = []
                    for i, doc in enumerate(domain_docs + qa_docs, 1):
                        content = doc.page_content.strip()
                        metadata = doc.metadata
                        source = metadata.get('source', 'Unknown')
                        context_parts.append(f"Nguồn {i}: {source}\nNội dung: {content}")
                    context = "\n\n".join(context_parts)
                # === KẾT THÚC PHẦN DEBUG ===
                
                if not context:
                    fallback_message = (
                        f"Xin lỗi, hiện tại hệ thống của tôi không tìm thấy dữ liệu phù hợp với câu hỏi của bạn."
                        f"Nếu cần hỗ trợ thêm, hãy cho tôi biết nhé!"
                    )
                    return {
                        "answer": fallback_message,
                        "sources": [],
                        "domain": domain
                    }
                prompt = create_prompt_llama(query, context, max_tokens=2000)
                response = generate_response(prompt)
                if "### Response:" in response:
                    answer = response.split("### Response:")[-1].strip()
                else:
                    answer = response.strip()

                # answer = domain_docs[0].page_content
                
                # Thêm nguồn tham khảo và URL
                sources = None
                web_url = None
                # Dọn dẹp bộ nhớ GPU
                self.clean_gpu_memory()
                # print('context: ', context)
                print('answer: ', answer)
                return {
                    "answer": answer,
                    "sources": list(sources) if sources is not None else [],
                    "domain": self.get_domain_name(domain)
                }

            except Exception as e:
                logger.error(f"Error in get_chat_response for Llama: {str(e)}")
                return {
                    "answer": f"Đã xảy ra lỗi: {str(e)}",
                    "sources": [],
                    "domain": None
                }
        
    def cleanup(self):
        try:
            clear_gpu_memory()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}") 
