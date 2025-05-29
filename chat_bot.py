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
    page_title="PDF ChatBot - Local LLM",
    page_icon="📄💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules
from pdf_processor import PDFProcessor
from text_processor import TextProcessor
from vector_store import VectorStore, RetrievalService

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
    """Service for GGUF models using llama-cpp-python"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize GGUF model"""
        try:
            logger.info(f"🚀 Loading GGUF model: {self.model_name}")
            
            # Initialize with GPU acceleration
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context window
                n_gpu_layers=32,  # GPU layers for T4
                n_batch=512,  # Batch size
                verbose=False,
                n_threads=4
            )
            
            logger.info("✅ GGUF model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load GGUF model: {e}")
            self.llm = None
    
    def generate_response(self, query: str, context: str, chat_history=None) -> str:
        """Generate response using GGUF model"""
        
        if self.llm is None:
            return self._generate_fallback_response(query, context)
        
        try:
            # Build Mistral-style prompt
            system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. Verilen bağlam bilgilerini kullanarak Türkçe cevap ver.

Kurallar:
- Sadece bağlam bilgilerini kullan
- Eğer bağlamda cevap yoksa "Bu bilgi dokümanlarda bulunmuyor" de
- Finansal terimleri doğru kullan
- Sayısal verileri dikkatli kontrol et
- Kısa ve net cevaplar ver"""

            # Mistral chat format
            prompt = f"<s>[INST] {system_prompt}\n\nBağlam Bilgileri:\n{context}\n\nSoru: {query} [/INST]"
            
            # Generate response
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=["</s>", "[INST]", "<s>"],
                echo=False
            )
            
            generated_text = response['choices'][0]['text'].strip()
            
            # Clean response
            generated_text = generated_text.replace("</s>", "").replace("[INST]", "").strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"GGUF generation failed: {e}")
            return self._generate_fallback_response(query, context)
    
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
    
    def generate_response(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """
        Generate response using available LLM service
        """
        
        # Try GGUF model first
        if self.gguf_service and self.gguf_service.llm:
            try:
                return self.gguf_service.generate_response(query, context, chat_history)
            except Exception as e:
                logger.error(f"GGUF model failed: {e}")
        
        # Try local LLM
        if self.local_service:
            try:
                return self.local_service.generate_response(query, context, chat_history)
            except Exception as e:
                logger.error(f"Local LLM failed: {e}")
        
        # Try OpenAI API
        if self.openai_client:
            try:
                return self._generate_openai_response(query, context, chat_history)
            except Exception as e:
                logger.error(f"OpenAI API failed: {e}")
        
        # Fallback response
        return self._generate_fallback_response(query, context)
    
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
        self.text_processor = TextProcessor(chunk_size=800, overlap_size=150)
        self.vector_store = VectorStore()
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=self.text_processor.embedding_service
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
    
    def process_pdf_file(self, pdf_file) -> str:
        """Process uploaded PDF file and add to vector store"""
        if pdf_file is None:
            return "❌ Lütfen bir PDF dosyası yükleyin."
        
        # Check if file already processed
        if pdf_file.name in self.processed_files:
            return f"ℹ️ {pdf_file.name} zaten işlenmiş. Yeni sorular sorabilirsiniz."
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                temp_path = tmp_file.name
            
            # Process PDF
            logger.info(f"Processing uploaded PDF: {pdf_file.name}")
            pdf_result = self.pdf_processor.process_pdf(temp_path)
            
            # Generate embeddings and add to vector store
            embedded_chunks = self.text_processor.process_document_pages(
                pdf_result.pages, 
                pdf_file.name
            )
            
            # Add to vector store
            self.vector_store.add_documents(embedded_chunks)
            
            # Track processed file
            self.processed_files.append(pdf_file.name)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            # Generate summary
            stats = self.text_processor.get_processing_stats(embedded_chunks)
            
            # Log vector store status
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
            
            return summary
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return f"❌ PDF işlenemedi: {str(e)}"
    
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
            response = self.llm_service.generate_response(
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
    
    def generate_single_response(self, query: str) -> str:
        """
        Generate a single response for the given query
        Optimized version with smart retrieval for large documents
        """
        if not query.strip():
            return "❌ Lütfen bir soru girin."
        
        try:
            # Smart retrieval based on document size
            vector_stats = self.retrieval_service.get_retrieval_stats()
            total_docs = vector_stats['vector_store_stats'].get('total_documents', 0)
            
            # Adjust retrieval parameters based on document size
            if total_docs > 200:  # Large document
                n_results = 8  # More results for better coverage
                logger.info(f"🔍 Large document mode: {total_docs} chunks, retrieving {n_results} results")
            elif total_docs > 100:
                n_results = 6  # Medium document
            else:
                n_results = 5  # Default for small documents
            
            # Retrieve relevant context
            context_result = self.retrieval_service.retrieve_context(
                query=query,
                n_results=n_results
            )
            
            # Check if we have any context
            if not context_result.combined_context.strip():
                return "❌ Bu soruya cevap verebilmek için ilgili bilgi dokümanlarda bulunamadı. Lütfen farklı bir soru deneyin."
            
            # Smart context filtering for large documents
            if total_docs > 200 and len(context_result.combined_context) > 3000:
                # Truncate context to most relevant parts
                context_parts = context_result.combined_context.split('\n\n')
                
                # Keep first 2000 chars of most relevant content
                filtered_context = ""
                for part in context_parts:
                    if len(filtered_context + part) < 2000:
                        filtered_context += part + "\n\n"
                    else:
                        break
                
                if filtered_context:
                    context_result.combined_context = filtered_context.strip()
                    logger.info(f"📝 Context filtered for efficiency: {len(context_result.combined_context)} chars")
            
            # Generate response
            response = self.llm_service.generate_response(
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
            
            # Update internal chat history
            user_msg = ChatMessage(
                role="user",
                content=query,
                timestamp=datetime.now()
            )
            
            assistant_msg = ChatMessage(
                role="assistant", 
                content=response,
                timestamp=datetime.now(),
                sources=[r.source_file for r in context_result.results] if context_result.results else []
            )
            
            self.chat_history.extend([user_msg, assistant_msg])
            
            return response
            
        except Exception as e:
            logger.error(f"Single response generation failed: {e}")
            return f"❌ Cevap oluşturulamadı: {str(e)}\n\nLütfen tekrar deneyin veya farklı bir soru sorun."
    
    def get_stats(self) -> str:
        """Get current chatbot statistics including table information"""
        try:
            retrieval_stats = self.retrieval_service.get_retrieval_stats()
            vector_stats = retrieval_stats['vector_store_stats']
            llm_info = self.llm_service.get_service_info()
            
            # Get text processing stats if available
            try:
                # Get sample embedded chunk to analyze processing stats
                sample_docs = self.vector_store.similarity_search("test", k=1)
                if sample_docs:
                    # This is a workaround to get text processing stats
                    text_stats = {
                        'table_chunks': vector_stats.get('table_chunks', 0),
                        'text_chunks': vector_stats.get('text_chunks', 0),
                        'pages_with_tables': vector_stats.get('pages_with_tables', 0),
                        'content_type_distribution': vector_stats.get('content_type_distribution', {})
                    }
                else:
                    text_stats = {}
            except:
                text_stats = {}
            
            stats_text = f"""
            ## 📊 ChatBot İstatistikleri
            
            **🤖 LLM Servisi:**
            - Tip: {llm_info['service_type']}
            - Model: {llm_info['model_name']}
            - Durum: {llm_info['status']}
            - Device: {llm_info.get('device', 'N/A')}
            - GPU: {llm_info.get('gpu_info', 'N/A')}
            
            **📚 Doküman Bilgileri:**
            - Toplam chunk: {vector_stats.get('total_documents', 0)}
            - Metin chunks: {text_stats.get('text_chunks', 'N/A')}
            - Tablo chunks: {text_stats.get('table_chunks', 'N/A')}
            - Tablolu sayfalar: {text_stats.get('pages_with_tables', 'N/A')}
            - İşlenen dosyalar: {len(self.processed_files)}
            - Dosya listesi: {', '.join(self.processed_files) if self.processed_files else 'Henüz dosya yüklenmedi'}
            
            **💬 Sohbet Bilgileri:**
            - Toplam mesaj: {len(self.chat_history)}
            - Embedding model: {retrieval_stats['embedding_model']}
            
            **🔤 Dil Dağılımı:**
            {json.dumps(vector_stats.get('language_distribution', {}), indent=2)}
            
            **📊 İçerik Türü Dağılımı:**
            {json.dumps(text_stats.get('content_type_distribution', {}), indent=2)}
            """
            
            return stats_text
            
        except Exception as e:
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
    
    # Title
    st.title("📄💬 PDF ChatBot - Local LLM")
    
    st.markdown("""
    **Finansal dokümanlarınızı yükleyin ve Local LLM ile sorularınızı sorun!**
    
    - 🤖 Llama 3.1 8B veya Mistral 7B modelleri
    - 📄 PDF dosyalarınızı yükleyin  
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
            
            # Default path for your specific model
            default_path = "/content/drive/MyDrive/Colab Notebooks/kredi_rag_sistemi/backup/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
            
            model_path = st.text_input(
                "🔗 GGUF Model Path:",
                value=default_path,
                help="Drive'daki .gguf model dosyanızın tam path'i"
            )
            
            # Model path validation
            if model_path:
                if os.path.exists(model_path):
                    st.success(f"✅ GGUF Model bulundu!")
                    try:
                        file_size = os.path.getsize(model_path) / 1e9
                        st.info(f"📊 Dosya boyutu: {file_size:.1f}GB")
                    except:
                        pass
                else:
                    st.error("❌ GGUF Model dosyası bulunamadı")
            
            # GGUF requirements check
            if not GGUF_AVAILABLE:
                st.error("""
                ❌ **llama-cpp-python yüklü değil!**
                
                Colab'da çalıştırın:
                ```bash
                !pip install llama-cpp-python
                ```
                """)
            
            # Initialize GGUF ChatBot
            if st.button("🚀 GGUF ChatBot'u Başlat", type="primary", disabled=st.session_state.model_initialized):
                if not GGUF_AVAILABLE:
                    st.error("❌ llama-cpp-python kütüphanesi gerekli!")
                elif not model_path or not os.path.exists(model_path):
                    st.error("❌ Geçerli model path'i girin!")
                else:
                    with st.spinner("GGUF Model yükleniyor... (Bu birkaç dakika sürebilir)"):
                        try:
                            st.session_state.chatbot = PDFChatBot(
                                use_local_llm=True,
                                model_choice="gguf_model",
                                gguf_model_path=model_path
                            )
                            
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
                            
                            st.session_state.chatbot = PDFChatBot(
                                use_local_llm=True,
                                model_choice="custom_drive_model"
                            )
                            
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
        st.header("📁 PDF Yükle")
        
        uploaded_file = st.file_uploader(
            "PDF dosyası seçin:",
            type=['pdf'],
            help="Analiz etmek istediğiniz PDF dosyasını yükleyin"
        )
        
        if st.button("📤 Yükle ve İşle", type="primary"):
            if not st.session_state.model_initialized:
                st.error("❌ Önce ChatBot'u başlatın")
            elif uploaded_file is not None:
                with st.spinner("PDF işleniyor..."):
                    try:
                        result = st.session_state.chatbot.process_pdf_file(uploaded_file)
                        if "✅" in result:
                            # Update processed files in session state
                            if uploaded_file.name not in st.session_state.processed_files:
                                st.session_state.processed_files.append(uploaded_file.name)
                            st.success(result)
                        else:
                            st.error(result)
                    except Exception as e:
                        st.error(f"PDF işleme hatası: {e}")
            else:
                st.warning("Lütfen bir PDF dosyası seçin")
        
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
                # Additional check - verify vector store has data
                try:
                    retrieval_stats = st.session_state.chatbot.retrieval_service.get_retrieval_stats()
                    total_docs = retrieval_stats['vector_store_stats'].get('total_documents', 0)
                    if total_docs == 0:
                        st.error("❌ Vector store'da veri bulunamadı. PDF'leri tekrar yükleyin.")
                        st.stop()
                except:
                    st.error("❌ Vector store durumu kontrol edilemedi.")
                    st.stop()
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.write(query)
                
                # Generate and display response
                with st.chat_message("assistant"):
                    with st.spinner("Cevap üretiliyor..."):
                        try:
                            # Get response using the improved method
                            response = st.session_state.chatbot.generate_single_response(query)
                            st.write(response)
                            
                            # Add to chat history
                            st.session_state.chat_history.append([query, response])
                            
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