import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
import warnings
import streamlit as st
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from langchain_community.llms import HuggingFacePipeline
import torch
from langchain.schema import Document, BaseRetriever
from pydantic import BaseModel, Field
from typing import List, Any
import logging
import gc
from search_web import test_search
from peft import PeftModel, PeftConfig
from langchain_community.embeddings import HuggingFaceEmbeddings

# Cấu hình logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

def load_config():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config.json")
        config_path = os.path.normpath(config_path)
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)
            return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def load_model(api_key, model_type="gpt-4o-mini"):
    try:
        if model_type == "gpt-4o-mini":
            return ChatOpenAI(
                model_name="gpt-4o-mini",
                temperature=0.5,
                openai_api_key=api_key
            )
        elif model_type == "llama":
            # Dọn bộ nhớ GPU nếu có
            if torch.cuda.is_available():
                clear_gpu_memory()
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            
            # Lấy đường dẫn đến thư mục models
            current_dir = os.path.dirname(os.path.abspath(__file__))
            adapter_path = os.path.join(current_dir, "..", "Fine_tuning", "models")
            
            # Load cấu hình PEFT và lấy tên mô hình cơ sở
            peft_config = PeftConfig.from_pretrained(adapter_path)
            base_model_name = peft_config.base_model_name_or_path
            
            # Load mô hình cơ sở
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                load_in_8bit=True
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load adapter weights
            model = PeftModel.from_pretrained(
                base_model,
                adapter_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            
            # Tạo pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                return_full_text=False,
                truncation=True
            )
            
            # Tạo một class wrapper để match với interface của ChatOpenAI
            class LlamaWrapper:
                def __init__(self, pipe):
                    self._pipe = pipe
                
                def clean_gpu_memory(self):
                    clear_gpu_memory()
                
                def __call__(self, prompt):
                    return self._pipe(prompt)[0]['generated_text']
            
            return LlamaWrapper(pipe)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_vector_db(api_key, collection_name, embedding_model_choice="OpenAI Embeddings"):
    try:
        config = load_config()
        persist_directory = r"E:\document\lab\DATN_ChatbotHUST\RAG\vector_db\chroma_db"
        
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            
        collection_path = os.path.join(persist_directory, collection_name)
        if not os.path.exists(collection_path):
            os.makedirs(collection_path)
            
        qa_path = os.path.join(persist_directory, "Q_A")
        if not os.path.exists(qa_path):
            os.makedirs(qa_path)
        
        # Chọn embedding model dựa trên embedding_model_choice
        if embedding_model_choice == "OpenAI Embeddings":
            embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
        elif embedding_model_choice == "HuggingFace Embeddings":
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
            )
        else:
            raise ValueError(f"Unknown embedding model choice: {embedding_model_choice}")
        
        domain_db = Chroma(
            collection_name=collection_name,
            persist_directory=collection_path,
            embedding_function=embedding_model
        )
        
        qa_db = Chroma(
            collection_name="Q_A",
            persist_directory=qa_path,
            embedding_function=embedding_model
        )
        
        # Debug: Kiểm tra số lượng tài liệu trong domain_db và qa_db
        domain_doc_count = domain_db._collection.count() if hasattr(domain_db, '_collection') else 0
        qa_doc_count = qa_db._collection.count() if hasattr(qa_db, '_collection') else 0
        print(f"Domain DB ({collection_name}) count: {domain_doc_count}")
        print(f"QA DB count: {qa_doc_count}")
        if domain_doc_count == 0:
            print(f"⚠️ Domain DB ({collection_name}) rỗng!")
        if qa_doc_count == 0:
            print("⚠️ QA DB rỗng!")
        
        return domain_db, qa_db
    except Exception as e:
        logger.error(f"Error loading vector database: {str(e)}")
        raise

def classify_question_domain(question):
    domains = {
        "thong_tin_chung": ["thông tin chung", "giới thiệu", "tổng quan", "trường", "bách khoa"],
        "de_an_tuyen_sinh": ["đề án", "chỉ tiêu", "phương thức"],
        "xet_tuyen_tai_nang": ["tài năng", "xet tuyển tài năng", "năng khiếu"],
        "diem_chuan_tuyen_sinh": [
            "điểm chuẩn", "điểm trúng tuyển", "điểm đầu vào", 
            "điểm thi", "điểm xét tuyển",
            "điểm chuẩn 2024", "điểm trúng tuyển 2024",
            "năm 2024", "2024"
        ],
        "ky_thi_danh_gia_tu_duy": ["tư duy", "kỳ thi", "đánh giá", "ĐGTD", "thi tư duy"],
        "xac_thuc_chung_chi_ngoai_ngu": ["ngoại ngữ", "chứng chỉ", "tiếng anh", "xác thực"],
        "huong_nghiep": ["hướng nghiệp", "nghề nghiệp", "định hướng", "ngành học"]
    }
    
    question = question.lower()
    for domain, keywords in domains.items():
        if any(keyword in question for keyword in keywords):
            return domain
    return "Q_A"

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()