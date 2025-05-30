#!/usr/bin/env python3
"""
PDF ChatBot - Complete RAG Implementation
Interactive chat interface with PDF upload and question answering
Now supports Local LLM models (Llama & Mistral) via HuggingFace
"""

import os
import logging
import streamlit as st
from typing import List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="PDF & Excel ChatBot - Local LLM",
    page_icon="📊💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from vector_store import VectorStore, RetrievalService
from excel_processor import ExcelProcessor

# Try to import local LLM service
try:
    from llm_service_local import LocalLLMService, create_optimized_llm_service, RECOMMENDED_MODELS
    LOCAL_LLM_AVAILABLE = True
    logger.info("✅ Local LLM service available")
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logger.warning("⚠️ Local LLM service not available")

# GGUF Model support import'unu diğer import'ların sonuna ekle
try:
    from llama_cpp import Llama
    GGUF_AVAILABLE = True
    logger.info("✅ llama-cpp-python available for GGUF models")
except ImportError:
    GGUF_AVAILABLE = False
    logger.warning("⚠️ llama-cpp-python not available for GGUF models")

@dataclass
class ChatMessage:
    """Container for chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None

# Add GGUF service class BEFORE LLMService class
class GGUFModelService:
    """Service for GGUF models using llama-cpp-python - A100 Optimized"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GGUF model with A100 optimizations"""
        try:
            logger.info(f"🚀 Loading GGUF model: {self.model_name} from path: {self.model_path}")
            
            # A100 GPU optimized settings
            import torch
            
            # Detect GPU and set optimal parameters
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                if "A100" in gpu_name:
                    # A100 specific optimizations
                    n_ctx = 8192        # Large context window for A100
                    n_batch = 4096      # Large batch size for A100's memory
                    n_threads = 8       # More CPU threads for A100 systems
                    logger.info("🎯 A100 detected - Using maximum performance settings")
                elif gpu_memory > 20:
                    # High-end GPU settings
                    n_ctx = 6144
                    n_batch = 3072
                    n_threads = 6
                    logger.info("🚀 High-end GPU detected - Using optimized settings")
                else:
                    # Standard GPU settings
                    n_ctx = 4096
                    n_batch = 2048
                    n_threads = 4
                    logger.info("📊 Standard GPU detected - Using balanced settings")
            else:
                # CPU fallback
                n_ctx = 2048
                n_batch = 512
                n_threads = 4
                logger.warning("⚠️ No GPU detected - Using CPU settings")
            
            # Initialize with optimized settings
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,           # Dynamic context window
                n_gpu_layers=-1,       # Offload all layers to GPU
                n_batch=n_batch,       # Dynamic batch size
                verbose=True,
                n_threads=n_threads,   # Dynamic CPU threads
                # A100 specific optimizations
                use_mmap=True,         # Memory mapping for efficiency
                use_mlock=False,       # Don't lock memory (let OS manage)
                numa=False,            # Disable NUMA for single GPU
                # Performance optimizations
                low_vram=False,        # A100 has plenty of VRAM
                f16_kv=True,          # Use half precision for key-value cache
                logits_all=False,     # Don't compute logits for all tokens
                vocab_only=False,     # Load full model
                # Threading and batching
                n_threads_batch=n_threads,  # Threads for batch processing
            )
            
            logger.info("✅ GGUF model loaded successfully")
            logger.info(f"📊 Context window: {n_ctx}")
            logger.info(f"🔢 Batch size: {n_batch}")
            logger.info(f"🧵 CPU threads: {n_threads}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            self.llm = None
    
    def generate_response(self, query: str, context: str, chat_history=None) -> Tuple[str, float]:
        """Generate response using GGUF model with A100 optimizations"""
        llm_start_time = datetime.now()
        duration_seconds: float = 0.0

        if self.llm is None:
            fallback_response = self._generate_fallback_response(query, context)
            llm_end_time = datetime.now()
            duration_seconds = (llm_end_time - llm_start_time).total_seconds()
            return fallback_response, duration_seconds
        
        try:
            system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. Verilen bağlam bilgilerini kullanarak Türkçe cevap ver.

Kurallar:
- Sadece bağlam bilgilerini kullan
- Eğer bağlamda cevap yoksa "Bu bilgi dokümanlarda bulunmuyor" de
- Finansal terimleri doğru kullan
- Sayısal verileri dikkatli kontrol et
- Kısa ve net cevaplar ver"""

            # Mistral chat format
            prompt = f"<s>[INST] {system_prompt}\n\nBağlam Bilgileri:\n{context}\n\nSoru: {query} [/INST]"
            
            # A100 optimized generation parameters
            response = self.llm(
                prompt,
                max_tokens=1024,       # Increased for A100
                temperature=0.7,
                top_p=0.95,           # Nucleus sampling
                top_k=40,             # Top-k sampling
                repeat_penalty=1.1,   # Prevent repetition
                stop=["</s>", "[INST]", "[/INST]"],  # Stop sequences
                # A100 performance optimizations
                stream=False,         # Non-streaming for batch efficiency
                echo=False,          # Don't echo prompt
                # Threading for A100
                threads=8 if "A100" in str(self.llm) else 4,
            )
            
            # Extract response text
            if isinstance(response, dict) and 'choices' in response:
                response_text = response['choices'][0]['text'].strip()
            elif isinstance(response, dict) and 'content' in response:
                response_text = response['content'].strip()
            else:
                response_text = str(response).strip()
            
            # Clean up response
            response_text = self._clean_response(response_text)
            
            llm_end_time = datetime.now()
            duration_seconds = (llm_end_time - llm_start_time).total_seconds()
            
            logger.info(f"✅ GGUF response generated in {duration_seconds:.2f}s")
            return response_text, duration_seconds
            
        except Exception as e:
            logger.error(f"❌ GGUF generation failed: {e}")
            fallback_response = self._generate_fallback_response(query, context)
            llm_end_time = datetime.now()
            duration_seconds = (llm_end_time - llm_start_time).total_seconds()
            return fallback_response, duration_seconds
    
    def _clean_response(self, response_text: str) -> str:
        """Clean and format the model response"""
        # Remove unwanted tokens and formatting
        response_text = response_text.replace("</s>", "")
        response_text = response_text.replace("[INST]", "")
        response_text = response_text.replace("[/INST]", "")
        response_text = response_text.replace("<s>", "")
        
        # Remove extra whitespace
        response_text = " ".join(response_text.split())
        
        # Remove any repeated patterns
        lines = response_text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines[-3:]:  # Avoid recent repetitions
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Fallback when GGUF model fails"""
        if context:
            return f"""
**[GGUF Model Hatası]**

Sorunuz: {query}

Bulunan İlgili Bilgiler:
{context[:500]}...

💡 GGUF model düzgün yüklendiğinde bu bilgileri analiz ederek detaylı cevap verebilirim.
"""
        else:
            return "Bu konuda dokümanlarda ilgili bilgi bulunamadı."
    
    def get_model_info(self) -> dict:
        """Get GGUF model information"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.llm is not None,
            "device": "gpu" if self.llm else "unknown",
            "model_path": self.model_path,
            "service_type": "gguf"
        }

# Modify existing LLMService class - add gguf_model_path parameter
class LLMService:
    """
    LLM service supporting GGUF, OpenAI API and Local models
    """
    
    def __init__(self, 
                 use_local: bool = True,
                 model_choice: str = "llama_3_1_8b",
                 api_key: Optional[str] = None,
                 gguf_model_path: Optional[str] = None):
        
        self.use_local = use_local and LOCAL_LLM_AVAILABLE
        self.model_choice = model_choice
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.gguf_model_path = gguf_model_path
        
        self.local_service = None
        self.gguf_service = None
        self.openai_client = None
        
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the appropriate LLM service"""
        
        # Try GGUF model first if path provided
        if self.gguf_model_path and GGUF_AVAILABLE:
            try:
                logger.info(f"🚀 Initializing GGUF model: {self.gguf_model_path}")
                self.gguf_service = GGUFModelService(self.gguf_model_path)
                if self.gguf_service.llm:
                    logger.info("✅ GGUF model service initialized")
                    return
            except Exception as e:
                logger.error(f"❌ GGUF model initialization failed: {e}")
        
        # Original logic remains the same
        if self.use_local:
            try:
                logger.info(f"🚀 Initializing local LLM: {self.model_choice}")
                self.local_service = create_optimized_llm_service(self.model_choice)
                logger.info("✅ Local LLM service initialized")
                return
            except Exception as e:
                logger.error(f"❌ Local LLM initialization failed: {e}")
                logger.info("🔄 Falling back to OpenAI API...")
        
        # Fallback to OpenAI
        if self.api_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=self.api_key)
                logger.info("✅ OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI library not installed")
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
        else:
            logger.warning("No OpenAI API key provided")
    
    def generate_response(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> Tuple[str, Optional[float]]:
        """
        Generate response using available LLM service and return response with its duration.
        """
        response_text: str
        duration_seconds: Optional[float] = None
        
        # Try GGUF model first
        if self.gguf_service and self.gguf_service.llm:
            try:
                response_text, duration_seconds = self.gguf_service.generate_response(query, context, chat_history)
                return response_text, duration_seconds
            except Exception as e:
                logger.error(f"GGUF model failed: {e}")
        
        # Try local LLM
        if self.local_service:
            try:
                # Placeholder: Modify LocalLLMService.generate_response to return Tuple[str, float] as well
                # response_text, duration_seconds = self.local_service.generate_response(query, context, chat_history)
                # return response_text, duration_seconds
                logger.warning("LocalLLMService duration timing not implemented yet in this refactor.")
                response_text = self.local_service.generate_response(query, context, chat_history)
                return response_text, None # Duration not available yet for this path
            except Exception as e:
                logger.error(f"Local LLM failed: {e}")
        
        # Try OpenAI API
        if self.openai_client:
            try:
                openai_start_time = datetime.now()
                response_text = self._generate_openai_response(query, context, chat_history)
                openai_end_time = datetime.now()
                duration_seconds = (openai_end_time - openai_start_time).total_seconds()
                return response_text, duration_seconds
            except Exception as e:
                logger.error(f"OpenAI API failed: {e}")
        
        # Fallback response
        fallback_start_time = datetime.now()
        response_text = self._generate_fallback_response(query, context)
        fallback_end_time = datetime.now()
        duration_seconds = (fallback_end_time - fallback_start_time).total_seconds()
        return response_text, duration_seconds
    
    def get_service_info(self) -> dict:
        """Get information about the current LLM service"""
        
        if self.gguf_service and self.gguf_service.llm:
            return self.gguf_service.get_model_info()
        elif self.local_service:
            local_info = self.local_service.get_model_info()
            return {
                "service_type": "local",
                "model_name": local_info["model_name"],
                "status": "active" if local_info["model_loaded"] else "failed",
                "device": local_info.get("device", "unknown"),
                "gpu_info": local_info.get("gpu_name", "N/A")
            }
        elif self.openai_client:
            return {
                "service_type": "openai_api",
                "model_name": "gpt-3.5-turbo",
                "status": "active"
            }
        
        return {
            "service_type": "unknown",
            "model_name": "none",
            "status": "not_available"
        }

    # Keep existing methods unchanged...
    def _generate_openai_response(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """Generate response using OpenAI API"""
        
        system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. 
        Sana verilen bağlam bilgilerini kullanarak kullanıcının sorularını Türkçe olarak cevapla.
        
        Kurallar:
        - Sadece verilen bağlam bilgilerini kullan
        - Eğer bağlamda cevap yoksa "Bu bilgi dokümanlarda bulunmuyor" de
        - Finansal terimleri doğru kullan
        - Kaynakları belirt
        - Kısa ve net cevaplar ver
        """
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            for msg in chat_history[-5:]:
                messages.append({"role": msg.role, "content": msg.content})
        
        user_message = f"""
        Bağlam Bilgileri:
        {context}
        
        Kullanıcı Sorusu:
        {query}
        """
        
        messages.append({"role": "user", "content": user_message})
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Fallback response when no LLM available"""
        
        if context:
            return f"""
            **[Demo Modu - LLM servisi mevcut değil]**
            
            Sorunuz: {query}
            
            Bulunan İlgili Bilgiler:
            {context[:500]}...
            
            💡 LLM servisi aktif olduğunda bu bilgileri analiz ederek detaylı cevap veririm.
            """
        else:
            return "Bu konuda dokümanlarda ilgili bilgi bulunamadı."

# Update PDFChatBot constructor
class PDFChatBot:
    """
    Main ChatBot class with GGUF support
    """
    
    def __init__(self, use_local_llm: bool = True, model_choice: str = "llama_3_1_8b", gguf_model_path: str = None):
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.excel_processor = ExcelProcessor()
        self.text_processor = TextProcessor(chunk_size=800, overlap_size=150)
        self.vector_store = VectorStore()
        
        # Create a simple embedding service wrapper
        class EmbeddingServiceWrapper:
            def __init__(self, model):
                self.model_name = model.model_name if hasattr(model, 'model_name') else str(model)
                self.model = model
            
            def generate_embedding(self, text: str):
                return self.model.encode(text)
        
        # Initialize retrieval service with wrapper
        embedding_wrapper = EmbeddingServiceWrapper(self.text_processor.model)
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=embedding_wrapper
        )
        
        # Initialize LLM service with GGUF support
        self.llm_service = LLMService(
            use_local=use_local_llm,
            model_choice=model_choice,
            gguf_model_path=gguf_model_path
        )
        
        # Chat state
        self.chat_history: List[ChatMessage] = []
        self.processed_files: List[str] = []
        
        logger.info("✅ PDF ChatBot initialized with GGUF support")
    
    def process_pdf_file(self, pdf_file) -> Tuple[str, Optional[float]]:
        """Process uploaded PDF file and add to vector store. Returns summary and duration."""
        process_start_time = datetime.now()
        duration_seconds: Optional[float] = None

        if pdf_file is None:
            return "❌ Lütfen bir PDF dosyası yükleyin.", None
        
        if pdf_file.name in self.processed_files:
            return f"ℹ️ {pdf_file.name} zaten işlenmiş. Yeni sorular sorabilirsiniz.", None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                temp_path = tmp_file.name
            
            logger.info(f"Processing uploaded PDF: {pdf_file.name}")
            pdf_result = self.pdf_processor.process_pdf(temp_path)
            
            embedded_chunks = self.text_processor.process_document_pages(
                pdf_result.pages, 
                pdf_file.name
            )
            
            self.vector_store.add_documents(embedded_chunks)
            self.processed_files.append(pdf_file.name)
            os.unlink(temp_path)
            
            stats = self.text_processor.get_processing_stats(embedded_chunks)
            vector_stats = self.retrieval_service.get_retrieval_stats()
            logger.info(f"Vector store now contains {vector_stats['vector_store_stats'].get('total_documents', 0)} documents")
            
            summary = f"""
            ✅ **PDF başarıyla işlendi!**
            
            📊 **İstatistikler:**
            - 📄 Sayfa sayısı: {pdf_result.total_pages}
            - 🧩 Toplam chunk: {stats['total_chunks']}
            - 📝 Metin chunks: {stats.get('text_chunks', 0)}
            - 📊 Tablo chunks: {stats.get('table_chunks', 0)}
            - 📚 Tablolu sayfalar: {stats.get('pages_with_tables', 0)}
            - 📝 Toplam karakter: {stats['total_characters']:,}
            - 🌍 Dil: {', '.join(stats['language_distribution'].keys())}
            - 🗃️ Vector store'da: {vector_stats['vector_store_stats'].get('total_documents', 0)} chunk
            
            💬 Artık bu dokümana ve tablolarına dair sorular sorabilirsiniz!
            """
            
            process_end_time = datetime.now()
            duration_seconds = (process_end_time - process_start_time).total_seconds()
            return summary, duration_seconds
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            process_end_time = datetime.now() # Record time even on failure for debugging
            duration_seconds = (process_end_time - process_start_time).total_seconds()
            return f"❌ PDF işlenemedi: {str(e)}", duration_seconds
    
    def process_excel_file(self, excel_file) -> Tuple[str, Optional[float]]:
        """Process uploaded Excel file and add to vector store. Returns summary and duration."""
        process_start_time = datetime.now()
        duration_seconds: Optional[float] = None

        if excel_file is None:
            return "❌ Lütfen bir Excel dosyası yükleyin.", None
        
        if excel_file.name in self.processed_files:
            return f"ℹ️ {excel_file.name} zaten işlenmiş. Yeni sorular sorabilirsiniz.", None
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(excel_file.name)[1]) as tmp_file:
                tmp_file.write(excel_file.read())
                temp_path = tmp_file.name
            
            logger.info(f"Processing uploaded Excel: {excel_file.name}")
            excel_result = self.excel_processor.process_excel(temp_path)
            
            embedded_chunks = self.text_processor.process_excel_sheets(
                excel_result.sheets, 
                excel_file.name
            )
            
            self.vector_store.add_documents(embedded_chunks)
            self.processed_files.append(excel_file.name)
            os.unlink(temp_path)
            
            summary_data = self.excel_processor.get_excel_summary(excel_result)
            vector_stats = self.retrieval_service.get_retrieval_stats()
            logger.info(f"Vector store now contains {vector_stats['vector_store_stats'].get('total_documents', 0)} documents")
            total_text_length = sum(len(sheet.text_content) for sheet in excel_result.sheets)
            sheet_names = [sheet.sheet_name for sheet in excel_result.sheets]
            
            summary = f"""
            ✅ **Excel dosyası başarıyla işlendi!**
            
            📊 **İstatistikler:**
            - 📄 Sayfa sayısı: {summary_data['total_sheets']}
            - 📝 Toplam satır: {summary_data['total_rows']}
            - 📋 Toplam sütun: {summary_data['total_columns']}
            - 🔢 Sayısal sütunlar: {summary_data['total_numeric_columns']}
            - 🧩 Toplam chunk: {len(embedded_chunks)}
            - 📝 Toplam karakter: {total_text_length:,}
            - 🗃️ Vector store'da: {vector_stats['vector_store_stats'].get('total_documents', 0)} chunk
            - 📊 Sayfalar: {', '.join(sheet_names)}
            
            💬 Artık bu Excel dosyasındaki verilere dair sorular sorabilirsiniz!
            
            **📋 Örnek Sorular:**
            - "Hangi sayfalarda hangi veriler var?"
            - "Toplam satır sayısı kaç?"
            - "En yüksek değer nedir?"
            - "Tablo verilerini özetle"
            """
            
            process_end_time = datetime.now()
            duration_seconds = (process_end_time - process_start_time).total_seconds()
            return summary, duration_seconds
            
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            process_end_time = datetime.now() # Record time even on failure for debugging
            duration_seconds = (process_end_time - process_start_time).total_seconds()
            return f"❌ Excel dosyası işlenemedi: {str(e)}", duration_seconds
    
    def chat(self, query: str, chat_history_display: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Handle chat interaction"""
        if not query.strip():
            return "", chat_history_display
        
        try:
            # Retrieve relevant context
            context_result = self.retrieval_service.retrieve_context(
                query=query,
                n_results=5
            )
            
            # Generate response
            response, duration_seconds = self.llm_service.generate_response(
                query=query,
                context=context_result.combined_context,
                chat_history=self.chat_history
            )
            
            # Add sources information if available
            if context_result.results:
                sources = []
                for result in context_result.results[:3]:  # Top 3 sources
                    source_info = f"📄 {result.source_file} (Sayfa {result.page_number})"
                    sources.append(source_info)
                
                response += f"\n\n**Kaynaklar:**\n" + "\n".join(sources)
            
            # Update chat history
            user_msg = ChatMessage(
                role="user",
                content=query,
                timestamp=datetime.now()
            )
            
            assistant_msg = ChatMessage(
                role="assistant", 
                content=response,
                timestamp=datetime.now(),
                sources=[r.source_file for r in context_result.results]
            )
            
            self.chat_history.extend([user_msg, assistant_msg])
            
            # Update display history
            if chat_history_display is None:
                chat_history_display = []
            
            chat_history_display.append([query, response])
            
            return "", chat_history_display
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            error_response = f"❌ Cevap oluşturulamadı: {str(e)}"
            
            if chat_history_display is None:
                chat_history_display = []
                
            chat_history_display.append([query, error_response])
            return "", chat_history_display
    
    def generate_single_response(self, query: str) -> Tuple[str, Optional[float], Optional[float]]:
        """
        Generate a single response for the given query, returning response, total duration, and LLM duration.
        Optimized version with smart retrieval for large documents
        """
        total_query_start_time = datetime.now()
        llm_duration_seconds: Optional[float] = None

        if not query.strip():
            return "❌ Lütfen bir soru girin.", (datetime.now() - total_query_start_time).total_seconds(), None
        
        try:
            # <<< Geliştirme Başlangıcı: Top-k (n_results) 3 olarak sabitlendi >>>
            n_results = 3 
            logger.info(f"🔍 Retrieving top {n_results} results for query.")
            # Eski dinamik n_results mantığı yorum satırı haline getirildi veya kaldırıldı.
            # vector_stats = self.retrieval_service.get_retrieval_stats()
            # total_docs = vector_stats['vector_store_stats'].get('total_documents', 0)
            # 
            # n_results = 5 # Varsayılan
            # if total_docs > 200: n_results = 8
            # elif total_docs > 100: n_results = 6
            # 
            # if n_results != 5: logger.info(f"🔍 Document mode: {total_docs} chunks, retrieving {n_results} results")
            # <<< Geliştirme Sonu: Top-k (n_results) 3 olarak sabitlendi >>>
            
            context_result = self.retrieval_service.retrieve_context(query=query, n_results=n_results)
            
            if not context_result.combined_context.strip():
                return "❌ Bu soruya cevap verebilmek için ilgili bilgi dokümanlarda bulunamadı. Lütfen farklı bir soru deneyin.", (datetime.now() - total_query_start_time).total_seconds(), None
            
            # Context filtering logic (can be kept or adjusted)
            vector_stats = self.retrieval_service.get_retrieval_stats() # For total_docs in filtering
            total_docs_for_filtering = vector_stats['vector_store_stats'].get('total_documents', 0)
            if total_docs_for_filtering > 200 and len(context_result.combined_context) > 3000: # Bu eşik değerleri ayarlanabilir
                context_parts = context_result.combined_context.split('\n\n')
                filtered_context = ""
                for part in context_parts:
                    if len(filtered_context + part) < 2000: # Bu karakter limiti de ayarlanabilir
                        filtered_context += part + "\n\n"
                    else: break
                if filtered_context:
                    context_result.combined_context = filtered_context.strip()
                    logger.info(f"📝 Context filtered for efficiency: {len(context_result.combined_context)} chars")
            
            # <<< Geliştirme Başlangıcı: LLM'e giden prompt'u loglama >>>
            logger.info(f"--- LLM INPUT START ---")
            logger.info(f"QUERY: {query}")
            logger.info(f"CONTEXT (first 500 chars): {context_result.combined_context[:500]}...")
            logger.info(f"CONTEXT (length): {len(context_result.combined_context)} chars")
            # GGUFModelService içindeki prompt formatını burada da loglayabiliriz (opsiyonel)
            # system_prompt_for_log = "Sen finansal dokümanları analiz eden uzman bir asistansın..."
            # full_prompt_for_log = f"<s>[INST] {system_prompt_for_log}\\n\\nBağlam Bilgileri:\\n{context_result.combined_context}\\n\\nSoru: {query} [/INST]"
            # logger.info(f"FULL PROMPT (length): {len(full_prompt_for_log)} chars")
            logger.info(f"--- LLM INPUT END ---")
            # <<< Geliştirme Sonu: LLM'e giden prompt'u loglama >>>

            response, llm_duration_seconds = self.llm_service.generate_response(
                query=query,
                context=context_result.combined_context,
                chat_history=self.chat_history
            )
            
            if context_result.results:
                sources = [f"📄 {result.source_file} (Sayfa {result.page_number})" for result in context_result.results[:3]]
                response += f"\n\n**Kaynaklar:**\n" + "\n".join(sources)
            
            user_msg = ChatMessage(role="user", content=query, timestamp=datetime.now())
            assistant_msg = ChatMessage(
                role="assistant", 
                content=response,
                timestamp=datetime.now(),
                sources=[r.source_file for r in context_result.results] if context_result.results else []
            )
            self.chat_history.extend([user_msg, assistant_msg])
            
            total_query_duration_seconds = (datetime.now() - total_query_start_time).total_seconds()
            return response, total_query_duration_seconds, llm_duration_seconds
            
        except Exception as e:
            logger.error(f"Single response generation failed: {e}")
            total_query_duration_seconds = (datetime.now() - total_query_start_time).total_seconds()
            return f"❌ Cevap oluşturulamadı: {str(e)}\n\nLütfen tekrar deneyin veya farklı bir soru sorun.", total_query_duration_seconds, None
    
    def get_stats(self) -> str:
        """Get current chatbot statistics including table information and timing."""
        try:
            retrieval_stats = self.retrieval_service.get_retrieval_stats()
            vector_store_actual_stats = retrieval_stats['vector_store_stats'] # Stats from FAISS via VectorStore.get_collection_stats()
            llm_info = self.llm_service.get_service_info()
            
            # Stats from TextProcessor for the last processed file (if any)
            current_processing_stats = self.text_processor.get_processing_stats()

            # Timing Information (from session_state)
            model_load_time = st.session_state.get('model_load_duration_seconds')
            file_process_time = st.session_state.get('last_file_processing_duration_seconds')
            query_response_time = st.session_state.get('last_query_response_duration_seconds')
            llm_inference_time = st.session_state.get('last_llm_inference_duration_seconds')
            
            timing_info = "\n\n**⏱️ Son İşlem Süreleri:**\n"
            if model_load_time is not None: timing_info += f"- Model Yükleme: {model_load_time:.2f} s\n"
            if file_process_time is not None: timing_info += f"- Son Dosya İşleme: {file_process_time:.2f} s\n"
            if query_response_time is not None: timing_info += f"- Son Sorgu Yanıtlama (Toplam): {query_response_time:.2f} s\n"
            if llm_inference_time is not None: timing_info += f"- Son LLM Çıkarımı: {llm_inference_time:.2f} s\n"
            if timing_info == "\n\n**⏱️ Son İşlem Süreleri:**\n": timing_info = "\n\n**⏱️ Son İşlem Süreleri:**\n- Henüz bir işlem yapılmadı.\n"

            # Use content_type_distribution from vector_store_actual_stats for overall view
            overall_content_distribution = vector_store_actual_stats.get('content_type_distribution', {})
            text_chunks_overall = 0
            table_chunks_overall = 0
            excel_chunks_overall = 0
            for ctype, count in overall_content_distribution.items():
                if ctype == 'text_pdf': text_chunks_overall += count
                elif ctype == 'table_pdf' or ctype == 'table_pdf_fragment': table_chunks_overall += count
                elif ctype == 'excel': excel_chunks_overall += count
                # Add other types if they exist and need to be categorized

            # Constructing the stats text
            stats_text = f"""
            ## 📊 ChatBot İstatistikleri
            
            **🤖 LLM Servisi:**
            - Tip: {llm_info['service_type']}
            - Model: {llm_info['model_name']}
            - Durum: {llm_info['status']}
            - Cihaz: {llm_info.get('device', 'N/A')}
            - GPU: {llm_info.get('gpu_info', 'N/A')}
            
            **📚 Doküman Bilgileri (Vector Store Genel):**
            - Toplam Chunk (Vector Store): {vector_store_actual_stats.get('total_documents', 0)}
            - Metin Chunk (Genel Tahmini): {text_chunks_overall}
            - Tablo Chunk (Genel Tahmini): {table_chunks_overall}
            - Excel Chunk (Genel Tahmini): {excel_chunks_overall}
            - Embedding Modeli: {retrieval_stats['embedding_model']}
            - İşlenen Dosyalar: {len(self.processed_files)}
            - Dosya Listesi: {', '.join(self.processed_files) if self.processed_files else 'Henüz dosya yüklenmedi'}
            
            **📄 Son İşlenen Dosya Detayları (TextProcessor):**
            - Toplam Chunk (Son İşlem): {current_processing_stats.get('total_chunks', 'N/A')}
            - Metin Chunk (Son İşlem): {current_processing_stats.get('text_chunks', 'N/A')}
            - Tablo Chunk (Son İşlem): {current_processing_stats.get('table_chunks', 'N/A')}
            - İçerik Dağılımı (Son İşlem): {json.dumps(current_processing_stats.get('content_type_distribution', {}), indent=2, ensure_ascii=False)}

            **🔤 Dil Dağılımı (Vector Store Genel - Örneklemden):**
            {json.dumps(vector_store_actual_stats.get('language_distribution', {}), indent=2, ensure_ascii=False)}
            (Örneklem Boyutu: {vector_store_actual_stats.get('sample_size_for_distributions', 'N/A')})
            
            **📊 İçerik Türü Dağılımı (Vector Store Genel - Örneklemden):**
            {json.dumps(overall_content_distribution, indent=2, ensure_ascii=False)}
            (Örneklem Boyutu: {vector_store_actual_stats.get('sample_size_for_distributions', 'N/A')})
            {timing_info}
            """
            
            return stats_text
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            return f"❌ İstatistik alınamadı: {str(e)}"

def main():
    """Main Streamlit application with GGUF support"""
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'current_query' not in st.session_state:
        st.session_state.current_query = ""
    # <<< Geliştirme Başlangıcı: Zamanlama için Session State Değişkenleri >>>
    if 'model_load_duration_seconds' not in st.session_state:
        st.session_state.model_load_duration_seconds = None
    if 'last_file_processing_duration_seconds' not in st.session_state:
        st.session_state.last_file_processing_duration_seconds = None
    if 'last_query_response_duration_seconds' not in st.session_state:
        st.session_state.last_query_response_duration_seconds = None
    if 'last_llm_inference_duration_seconds' not in st.session_state:
        st.session_state.last_llm_inference_duration_seconds = None
    # <<< Geliştirme Sonu: Zamanlama için Session State Değişkenleri >>>
    
    # Title
    st.title("📄💬 PDF & Excel ChatBot - Local LLM")
    
    st.markdown("""
    **Finansal dokümanlarınızı ve Excel dosyalarınızı yükleyin, Local LLM ile sorularınızı sorun!**
    
    - 🤖 Llama 3.1 8B veya Mistral 7B modelleri
    - 📄 PDF dosyalarınızı yükleyin
    - 📊 Excel dosyalarınızı yükleyin (.xls, .xlsx, .xlsm)  
    - 🧠 Akıllı finansal analiz
    - 🔒 Tamamen local (internet gerekmez)
    """)
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("⚙️ Model Ayarları")
        
        # GGUF Model option
        use_gguf = st.checkbox("🦙 GGUF Model Kullan (Mistral 7B)", value=True)
        
        if use_gguf:
            st.info("📁 GGUF Model Path'i giriniz:")
            default_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            model_path_input = st.text_input(
                "🔗 GGUF Model Path:",
                value=default_path,
                help="Drive'daki .gguf model dosyanızın tam path'i"
            )
            
            actual_model_path_for_llama = model_path_input

            if model_path_input:
                if os.path.exists(model_path_input):
                    st.success(f"✅ GGUF Model bulundu: {model_path_input}")
                    try:
                        file_size = os.path.getsize(model_path_input) / 1e9
                        st.info(f"📊 Dosya boyutu: {file_size:.1f}GB")
                    except:
                        pass
                else:
                    st.error(f"❌ GGUF Model dosyası bulunamadı: {model_path_input}")
            
            if not GGUF_AVAILABLE:
                st.error("""
                ❌ **llama-cpp-python yüklü değil!**
                
                Colab'da çalıştırın:
                ```bash
                !pip install llama-cpp-python
                ```
                """)
            
            if st.button("🚀 GGUF ChatBot'u Başlat", type="primary", disabled=st.session_state.model_initialized):
                if not GGUF_AVAILABLE:
                    st.error("❌ llama-cpp-python kütüphanesi gerekli!")
                elif not model_path_input or not os.path.exists(model_path_input):
                    st.error("❌ Geçerli model path'i girin!")
                else:
                    with st.spinner("GGUF Model yükleniyor... (Bu birkaç dakika sürebilir)"):
                        try:
                            # <<< Geliştirme Başlangıcı: Modeli Colab yerel diskine kopyalama >>>
                            final_model_path_to_load = model_path_input
                            if model_path_input.startswith("/content/drive/"):
                                local_model_dir = "/content/models_colab_local"
                                if not os.path.exists(local_model_dir):
                                    os.makedirs(local_model_dir)
                                model_filename = os.path.basename(model_path_input)
                                local_model_path = os.path.join(local_model_dir, model_filename)
                                
                                if not os.path.exists(local_model_path):
                                    st.info(f"Model Google Drive'da. {local_model_path} adresine kopyalanıyor...")
                                    import shutil
                                    shutil.copy2(model_path_input, local_model_path)
                                    st.info(f"Model kopyalandı: {local_model_path}")
                                    final_model_path_to_load = local_model_path
                                else:
                                    st.info(f"Model zaten yerel olarak mevcut: {local_model_path}")
                                    final_model_path_to_load = local_model_path
                            
                            logger.info(f"Attempting to load GGUF model into PDFChatBot using path: {final_model_path_to_load}")
                            # <<< Geliştirme Sonu: Modeli Colab yerel diskine kopyalama >>>

                            load_start_time = datetime.now()
                            st.session_state.chatbot = PDFChatBot(
                                use_local_llm=True,
                                model_choice="gguf_model",
                                gguf_model_path=final_model_path_to_load
                            )
                            load_end_time = datetime.now()
                            st.session_state.model_load_duration_seconds = (load_end_time - load_start_time).total_seconds()
                            
                            llm_info = st.session_state.chatbot.llm_service.get_service_info()
                            st.session_state.model_initialized = True
                            
                            st.success(f"""
                            ✅ **GGUF ChatBot başarıyla başlatıldı!**
                            
                            **🤖 Model Bilgileri:**
                            - Model: {llm_info.get('model_name', 'Unknown')}
                            - Durum: {llm_info.get('status', 'Unknown')}
                            - Device: {llm_info.get('device', 'Unknown')}
                            
                            💬 Artık PDF yükleyip soru sorabilirsiniz!
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ GGUF ChatBot başlatılamadı: {str(e)}")
        
        else:
            # Original HuggingFace model configuration
            model_path = st.text_input(
                "📁 Model Path:",
                value=os.environ.get("CUSTOM_MODEL_PATH", ""),
                placeholder="/content/drive/MyDrive/model-folder/model.gguf",
                help="Drive'daki model dosyanızın tam path'ini girin"
            )
            
            # Model path kontrolü
            if model_path:
                if os.path.exists(model_path):
                    st.success(f"✅ Model bulundu!")
                    # Model dosya bilgileri
                    try:
                        file_size = os.path.getsize(model_path) / 1e9
                        st.info(f"📊 Dosya boyutu: {file_size:.1f}GB")
                    except:
                        pass
                else:
                    st.error("❌ Model dosyası bulunamadı")
            
            # Initialize button
            if st.button("🚀 ChatBot'u Başlat", type="primary", disabled=st.session_state.model_initialized):
                if not model_path:
                    st.error("❌ Model path'i girin!")
                elif not os.path.exists(model_path):
                    st.error("❌ Model dosyası bulunamadı!")
                else:
                    with st.spinner("Model yükleniyor..."):
                        try:
                            # Set model path environment
                            os.environ["CUSTOM_MODEL_PATH"] = model_path
                            # <<< Geliştirme Başlangıcı: Model Yükleme Zamanlaması (Diğer Model Türü) >>>
                            load_start_time_other = datetime.now()
                            # <<< Geliştirme Sonu: Model Yükleme Zamanlaması (Diğer Model Türü) >>>
                            st.session_state.chatbot = PDFChatBot(
                                use_local_llm=True,
                                model_choice="custom_drive_model" # Bu model_choice'un LocalLLMService tarafından nasıl ele alındığına bağlı
                            )
                            # <<< Geliştirme Başlangıcı: Model Yükleme Zamanlaması (Diğer Model Türü) >>>
                            load_end_time_other = datetime.now()
                            st.session_state.model_load_duration_seconds = (load_end_time_other - load_start_time_other).total_seconds()
                            # <<< Geliştirme Sonu: Model Yükleme Zamanlaması (Diğer Model Türü) >>>
                            
                            llm_info = st.session_state.chatbot.llm_service.get_service_info()
                            st.session_state.model_initialized = True
                            
                            st.success(f"""
                            ✅ **ChatBot başarıyla başlatıldı!**
                            
                            **🤖 Model Bilgileri:**
                            - Path: {model_path}
                            - Durum: {llm_info['status']}
                            
                            💬 Artık PDF yükleyip soru sorabilirsiniz!
                            """)
                            
                        except Exception as e:
                            st.error(f"❌ ChatBot başlatılamadı: {str(e)}")
        
        # PDF Upload section - ORTAK ALAN (her iki model türü için de)
        st.divider()  # Visual separator
        st.header("📁 Dosya Yükle")
        
        # File type selection
        file_type = st.selectbox(
            "📄 Dosya türü seçin:",
            ["PDF", "Excel (XLS/XLSX)"],
            help="Yüklemek istediğiniz dosya türünü seçin"
        )
        
        if file_type == "PDF":
            uploaded_file = st.file_uploader(
                "PDF dosyası seçin:",
                type=['pdf'],
                help="Analiz etmek istediğiniz PDF dosyasını yükleyin"
            )
            
            if st.button("📤 PDF'i Yükle ve İşle", type="primary"):
                if not st.session_state.model_initialized:
                    st.error("❌ Önce ChatBot'u başlatın")
                elif uploaded_file is not None:
                    with st.spinner("PDF işleniyor..."):
                        try:
                            # <<< Geliştirme Başlangıcı: Dosya İşleme Süresini Alma >>>
                            result_summary, duration_secs = st.session_state.chatbot.process_pdf_file(uploaded_file)
                            if duration_secs is not None:
                                st.session_state.last_file_processing_duration_seconds = duration_secs
                            # <<< Geliştirme Sonu: Dosya İşleme Süresini Alma >>>
                            if "✅" in result_summary:
                                if uploaded_file.name not in st.session_state.processed_files:
                                    st.session_state.processed_files.append(uploaded_file.name)
                                st.success(result_summary)
                            else:
                                st.error(result_summary)
                        except Exception as e:
                            st.error(f"PDF işleme hatası: {e}")
                else:
                    st.warning("Lütfen bir PDF dosyası seçin")
        
        else:  # Excel
            uploaded_file = st.file_uploader(
                "Excel dosyası seçin:",
                type=['xls', 'xlsx', 'xlsm'],
                help="Analiz etmek istediğiniz Excel dosyasını yükleyin"
            )
            
            if st.button("📊 Excel'i Yükle ve İşle", type="primary"):
                if not st.session_state.model_initialized:
                    st.error("❌ Önce ChatBot'u başlatın")
                elif uploaded_file is not None:
                    with st.spinner("Excel dosyası işleniyor..."):
                        try:
                            # <<< Geliştirme Başlangıcı: Dosya İşleme Süresini Alma >>>
                            result_summary, duration_secs = st.session_state.chatbot.process_excel_file(uploaded_file)
                            if duration_secs is not None:
                                st.session_state.last_file_processing_duration_seconds = duration_secs
                            # <<< Geliştirme Sonu: Dosya İşleme Süresini Alma >>>
                            if "✅" in result_summary:
                                if uploaded_file.name not in st.session_state.processed_files:
                                    st.session_state.processed_files.append(uploaded_file.name)
                                st.success(result_summary)
                            else:
                                st.error(result_summary)
                        except Exception as e:
                            st.error(f"Excel işleme hatası: {e}")
                else:
                    st.warning("Lütfen bir Excel dosyası seçin")
        
        # Show processed files - ORTAK ALAN
        if st.session_state.processed_files:
            st.header("📚 İşlenen Dosyalar")
            for file_name in st.session_state.processed_files:
                st.write(f"✅ {file_name}")
        
        # Stats section - ORTAK ALAN
        st.header("📊 İstatistikler")
        if st.session_state.chatbot:
            try:
                stats_text = st.session_state.chatbot.get_stats()
                st.markdown(stats_text)
            except:
                st.info("İstatistik alınamadı")
        else:
            st.info("ChatBot başlatılmadı")
    
    # Main content area with simplified tabs
    tab1, tab2 = st.tabs(["💬 Sohbet", "ℹ️ Rehber"])
    
    with tab1:
        # Chat interface
        st.header("💬 Sohbet")
        
        # Show information about loaded data
        if st.session_state.processed_files:
            st.info(f"📚 Yüklenen dosyalar: {', '.join(st.session_state.processed_files)}")
        elif st.session_state.model_initialized:
            st.warning("⚠️ Henüz PDF yüklenmedi. Sol menüden PDF yükleyin.")
        
        # Check vector store status for additional safety
        if st.session_state.chatbot:
            try:
                retrieval_stats = st.session_state.chatbot.retrieval_service.get_retrieval_stats()
                total_docs = retrieval_stats['vector_store_stats'].get('total_documents', 0)
                if total_docs == 0 and st.session_state.processed_files:
                    st.error("⚠️ Veriler kaybolmuş gibi görünüyor. Lütfen PDF'leri tekrar yükleyin.")
                elif total_docs > 0:
                    st.success(f"✅ Vector store'da {total_docs} chunk hazır")
            except:
                pass
        
        # Display chat history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(bot_msg)
        
        # Chat input with proper handling
        query = st.chat_input("Sorunuzu yazın...", key="chat_input")
        
        # Handle query submission
        if query:
            if not st.session_state.model_initialized:
                st.error("❌ Önce ChatBot'u başlatın (Sol menüden)")
            elif not st.session_state.processed_files:
                st.error("❌ Önce PDF dosyası yükleyin (Sol menüden)")
            else:
                try:
                    retrieval_stats = st.session_state.chatbot.retrieval_service.get_retrieval_stats()
                    total_docs = retrieval_stats['vector_store_stats'].get('total_documents', 0)
                    if total_docs == 0:
                        st.error("❌ Vector store'da veri bulunamadı. PDF'leri tekrar yükleyin.")
                        st.stop()
                except:
                    st.error("❌ Vector store durumu kontrol edilemedi.")
                    st.stop()
                
                with st.chat_message("user"):
                    st.write(query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Cevap üretiliyor..."):
                        try:
                            # <<< Geliştirme Başlangıcı: Sorgu ve LLM Sürelerini Alma >>>
                            response_text, total_duration, llm_duration = st.session_state.chatbot.generate_single_response(query)
                            if total_duration is not None:
                                st.session_state.last_query_response_duration_seconds = total_duration
                            if llm_duration is not None:
                                st.session_state.last_llm_inference_duration_seconds = llm_duration
                            # <<< Geliştirme Sonu: Sorgu ve LLM Sürelerini Alma >>>
                            st.write(response_text)
                            
                            st.session_state.chat_history.append([query, response_text])
                            
                        except Exception as e:
                            error_msg = f"❌ Hata: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append([query, error_msg])
        
        # Clear chat button
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.chat_history = []
            st.rerun()
    
    with tab2:
        # Usage guide
        st.header("ℹ️ Kullanım Rehberi")
        
        st.markdown("""
        ## 🚀 Nasıl Kullanılır?
        
        ### 1. Model Seçimi ve Başlatma
        **GGUF Model (Önerilen):**
        - ✅ "GGUF Model Kullan" kutusunu işaretle  
        - 📁 Model path'ini gir: `/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`
        - 🚀 "GGUF ChatBot'u Başlat" butonuna tıkla
        
        **Normal HuggingFace Model:**
        - ❌ "GGUF Model Kullan" kutusunu kaldır
        - 📁 HuggingFace model path'ini gir 
        - 🚀 "ChatBot'u Başlat" butonuna tıkla
        
        ### 2. PDF Yükleme  
        - 📄 "PDF Yükle" bölümünden PDF dosyanızı seçin
        - 📤 "Yükle ve İşle" butonuna tıklayın
        - ⏳ İşlem tamamlanana kadar bekleyin
        - **📊 Tablolar otomatik olarak işlenir!**
        
        ### 2. Dosya Yükleme
        **PDF Dosyaları:**
        - 📄 Dosya türü olarak "PDF" seçin
        - 📁 PDF dosyanızı yükleyin
        - 📤 "PDF'i Yükle ve İşle" butonuna tıklayın
        - **📊 Tablolar otomatik olarak işlenir!**
        
        **Excel Dosyaları (YENİ!):**
        - 📊 Dosya türü olarak "Excel (XLS/XLSX)" seçin
        - 📁 .xls, .xlsx veya .xlsm dosyanızı yükleyin
        - 📤 "Excel'i Yükle ve İşle" butonuna tıklayın
        - **🔢 Tüm sayfalar ve veriler otomatik işlenir!**
        
        ### 3. Soru Sorma
        - 💬 "Sohbet" sekmesinde sorunuzu chat input'a yazın
        - ⏎ Enter'a basın veya gönder butonuna tıklayın
        - **🔍 Hem metin hem de tablo verilerinde arama yapar**
        """)
        
        with st.expander("🦙 GGUF vs Normal Model"):
            st.markdown("""
            **🎯 GGUF Model Avantajları:**
            - ✅ Daha hızlı başlatma
            - ✅ Daha az RAM kullanımı  
            - ✅ GPU acceleration ile hızlı
            - ✅ Quantized (sıkıştırılmış) format
            - ✅ Colab'da stabil çalışır
            
            **📚 Normal HuggingFace Model:**
            - ⚠️ Daha fazla RAM gerekir
            - ⚠️ İlk yükleme uzun sürer
            - ⚠️ HuggingFace token gerekebilir
            - ✅ Daha fazla model seçeneği
            """)
        
        with st.expander("📊 Tablo İşleme Özellikleri"):
            st.markdown("""
            **✅ Desteklenen Tablo İşlemleri:**
            - 🔍 Tablo içeriği otomatik çıkarılır
            - 📝 Tablolar aranabilir metin formatına dönüştürülür
            - 🧠 LLM tablolardan bilgi çıkarabilir
            - 📊 Sayısal verileri analiz edebilir
            
            **📋 Örnek Sorular:**
            - "Tablodaki en yüksek değer nedir?"
            - "Hangi kategoride kaç adet var?"
            - "2023 yılı verilerini göster"
            - "Finansal tablodan gelir kalemlerini listele"
            - "Cari dönem toplam dönen varlıklar ne kadar?"
            
            **⚡ Otomatik İşlenir:**
            - PDF yüklediğinizde tablolar otomatik bulunur
            - Boş tablolar filtrelenir
            - Her tablo ayrı chunk olarak işlenir
            - Tablo metadatası korunur
            """)
        
        with st.expander("📊 Excel İşleme Özellikleri (YENİ!)"):
            st.markdown("""
            **✅ Desteklenen Excel Formatları:**
            - 📄 .xls (Excel 97-2003)
            - 📄 .xlsx (Excel 2007+)
            - 📄 .xlsm (Macro-enabled Excel)
            
            **🔍 Otomatik İşlenen Veriler:**
            - 📊 Tüm sayfalar (sheets) okunur
            - 🔢 Sayısal ve metin verileri ayrı işlenir
            - 📈 Otomatik istatistikler (toplam, ortalama, min, max)
            - 📋 Sütun adları ve metadata korunur
            - 🧹 Boş satır/sütunlar temizlenir
            
            **📋 Excel Sorgu Örnekleri:**
            - "Hangi sayfalarda hangi veriler var?"
            - "Sheet1'deki toplam satır sayısı?"
            - "En yüksek maaş ne kadar?"
            - "Gelir tablosunu özetle"
            - "Satış verilerini analiz et"
            - "Tüm sayfalardaki sayısal özeti ver"
            - "Hangi sütunlarda hangi türde veriler var?"
            
            **⚡ Gelişmiş Özellikler:**
            - Her sayfa ayrı chunk olarak işlenir
            - Sayısal sütunlar için otomatik istatistik
            - Metin formatına dönüştürülmüş aranabilir veri
            - Vector store'da metadata ile arama
            """)

        with st.expander("⚠️ Troubleshooting"):
            st.markdown("""
            **❌ "GGUF Model bulunamadı" hatası:**
            - Drive mount edilmiş mi kontrol edin
            - Path'in doğru olduğunu kontrol edin
            - Dosya izinlerini kontrol edin
            
            **🐌 "Model yavaş yükleniyor":**
            - İlk yükleme 2-5 dakika sürer (normal)
            - GPU memory'yi kontrol edin
            - Colab Pro Plus kullanın
            
            **💾 "Memory error":**
            - Runtime'ı restart edin
            - GPU'yu kontrol edin: Runtime > Change runtime type > GPU
            
            **📊 "PDF yüklenmiyor":**
            - Önce model başlatılmış olmalı
            - PDF dosyası bozuk olmasın
            - Büyük dosyalar uzun sürebilir
            
            **🔍 "Cevap alamıyorum":**
            - Vector store'da veri var mı kontrol edin
            - PDF'yi tekrar yükleyin
            - Farklı soru ifadesi deneyin
            """)

if __name__ == "__main__":
    # Check for required libraries
    try:
        import streamlit
        logger.info("✅ Streamlit library found")
    except ImportError:
        logger.error("❌ Streamlit not installed. Run: pip install streamlit")
        exit(1)
    
    # Run the main app
    main() 