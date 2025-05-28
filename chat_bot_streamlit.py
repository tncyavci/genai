#!/usr/bin/env python3
"""
PDF ChatBot - Streamlit Implementation
Interactive chat interface with PDF upload and question answering
Supports Local LLM models (Llama & Mistral) via HuggingFace
"""

import os
import logging
import streamlit as st
from typing import List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import json
from datetime import datetime
import time

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

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        color: #1e3a8a;
    }
    .status-success {
        padding: 10px;
        border-radius: 5px;
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        margin: 10px 0;
    }
    .status-error {
        padding: 10px;
        border-radius: 5px;
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        margin: 10px 0;
    }
    .status-warning {
        padding: 10px;
        border-radius: 5px;
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e5e7eb;
    }
    .user-message {
        background-color: #dbeafe;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f3f4f6;
        margin-right: 2rem;
    }
</style>
""", unsafe_allow_html=True)

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

@dataclass
class ChatMessage:
    """Container for chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None

class LLMService:
    """
    LLM service supporting both OpenAI API and Local models
    """
    
    def __init__(self, 
                 use_local: bool = True,
                 model_choice: str = "llama_3_1_8b",
                 api_key: Optional[str] = None):
        
        self.use_local = use_local and LOCAL_LLM_AVAILABLE
        self.model_choice = model_choice
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        self.local_service = None
        self.openai_client = None
        
        self._initialize_service()
    
    def _initialize_service(self):
        """Initialize the appropriate LLM service"""
        
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
        """Generate response using available LLM service"""
        
        # Try local LLM first
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
    
    def get_service_info(self) -> dict:
        """Get information about the current LLM service"""
        
        info = {
            "service_type": "unknown",
            "model_name": "none",
            "status": "not_available"
        }
        
        if self.local_service:
            local_info = self.local_service.get_model_info()
            info.update({
                "service_type": "local",
                "model_name": local_info["model_name"],
                "status": "active" if local_info["model_loaded"] else "failed",
                "device": local_info.get("device", "unknown"),
                "gpu_info": local_info.get("gpu_name", "N/A")
            })
        elif self.openai_client:
            info.update({
                "service_type": "openai_api",
                "model_name": "gpt-3.5-turbo",
                "status": "active"
            })
        
        return info

class PDFChatBot:
    """Main ChatBot class with Local LLM support"""
    
    def __init__(self, use_local_llm: bool = True, model_choice: str = "llama_3_1_8b"):
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor()
        self.vector_store = VectorStore()
        self.retrieval_service = RetrievalService(self.vector_store)
        
        # Initialize LLM service
        self.llm_service = LLMService(
            use_local=use_local_llm,
            model_choice=model_choice
        )
        
        # Chat history
        self.chat_history: List[ChatMessage] = []
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "successful_responses": 0,
            "documents_processed": 0,
            "chunks_in_database": 0
        }
        
        logger.info("✅ PDF ChatBot initialized")
    
    def process_pdf_file(self, pdf_file) -> str:
        """Process uploaded PDF file"""
        try:
            if pdf_file is None:
                return "❌ Dosya seçilmedi"
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Process PDF
                logger.info(f"Processing PDF: {tmp_path}")
                
                # Extract text
                pdf_text = self.pdf_processor.extract_text(tmp_path)
                if not pdf_text.strip():
                    return "❌ PDF'den metin çıkarılamadı"
                
                # Process text into chunks
                chunks = self.text_processor.create_chunks(pdf_text)
                logger.info(f"Created {len(chunks)} text chunks")
                
                # Store in vector database
                self.vector_store.add_documents(chunks)
                
                # Update stats
                self.stats["documents_processed"] += 1
                self.stats["chunks_in_database"] = len(chunks)
                
                logger.info("✅ PDF processing completed")
                
                return f"""
                ✅ **PDF başarıyla işlendi!**
                
                📊 **İşlem Detayları:**
                - 📄 Metin uzunluğu: {len(pdf_text)} karakter
                - 🔤 Parça sayısı: {len(chunks)}
                - 💾 Veritabanına eklendi
                
                💬 Artık bu doküman hakkında sorular sorabilirsiniz!
                """
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            return f"❌ PDF işleme hatası: {str(e)}"
    
    def chat(self, query: str) -> str:
        """Process chat query and return response"""
        
        if not query.strip():
            return "❌ Lütfen bir soru yazın"
        
        try:
            self.stats["total_queries"] += 1
            
            # Retrieve relevant context
            relevant_docs = self.retrieval_service.retrieve_relevant_chunks(query, top_k=3)
            context = "\n\n".join(relevant_docs) if relevant_docs else ""
            
            # Generate response
            response = self.llm_service.generate_response(query, context, self.chat_history)
            
            # Add to chat history
            self.chat_history.append(ChatMessage(
                role="user",
                content=query,
                timestamp=datetime.now()
            ))
            
            self.chat_history.append(ChatMessage(
                role="assistant", 
                content=response,
                timestamp=datetime.now(),
                sources=relevant_docs[:2] if relevant_docs else None
            ))
            
            self.stats["successful_responses"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Chat processing failed: {e}")
            return f"❌ Soru işleme hatası: {str(e)}"
    
    def get_stats(self) -> dict:
        """Get chatbot statistics"""
        return self.stats.copy()

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    css_class = "user-message" if is_user else "assistant-message"
    role = "🙋‍♂️ Siz" if is_user else "🤖 ChatBot"
    
    st.markdown(f"""
    <div class="chat-message {css_class}">
        <strong>{role}:</strong><br>
        {message}
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session_state()
    
    # Header
    st.markdown('<div class="main-header">📄💬 PDF ChatBot - Local LLM Destekli</div>', unsafe_allow_html=True)
    
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
        
        # Model selection
        if LOCAL_LLM_AVAILABLE:
            model_choices = list(RECOMMENDED_MODELS.keys())
            default_model = "llama_3_1_8b"
        else:
            model_choices = ["local_not_available"]
            default_model = "local_not_available"
        
        selected_model = st.selectbox(
            "🎯 LLM Model Seçin:",
            model_choices,
            index=model_choices.index(default_model) if default_model in model_choices else 0
        )
        
        # Display model info
        if LOCAL_LLM_AVAILABLE and selected_model in RECOMMENDED_MODELS:
            config = RECOMMENDED_MODELS[selected_model]
            st.markdown(f"""
            **📊 Model Bilgileri:**
            - **İsim:** {config['name']}
            - **Memory:** {config['memory']}
            - **Açıklama:** {config['description']}
            """)
        elif not LOCAL_LLM_AVAILABLE:
            st.error("❌ **Local LLM mevcut değil**\n\nGerekli kütüphaneleri yükleyin:\n`pip install transformers accelerate torch`")
        
        # Initialize button
        if st.button("🚀 ChatBot'u Başlat", type="primary", disabled=st.session_state.model_initialized):
            if not LOCAL_LLM_AVAILABLE and selected_model != "local_not_available":
                st.error("Local LLM servisi mevcut değil!")
            else:
                with st.spinner("Model yükleniyor..."):
                    try:
                        st.session_state.chatbot = PDFChatBot(
                            use_local_llm=LOCAL_LLM_AVAILABLE,
                            model_choice=selected_model if LOCAL_LLM_AVAILABLE else "fallback"
                        )
                        
                        llm_info = st.session_state.chatbot.llm_service.get_service_info()
                        st.session_state.model_initialized = True
                        
                        st.success(f"""
                        ✅ **ChatBot başarıyla başlatıldı!**
                        
                        **🤖 Aktif LLM:**
                        - Tip: {llm_info['service_type']}
                        - Model: {llm_info['model_name']}
                        - Durum: {llm_info['status']}
                        
                        💬 Artık PDF yükleyip soru sorabilirsiniz!
                        """)
                        
                    except Exception as e:
                        st.error(f"❌ ChatBot başlatılamadı: {str(e)}")
        
        # Stats section
        st.header("📊 İstatistikler")
        if st.session_state.chatbot:
            stats = st.session_state.chatbot.get_stats()
            st.metric("Total Queries", stats["total_queries"])
            st.metric("Successful Responses", stats["successful_responses"])
            st.metric("Documents Processed", stats["documents_processed"])
            st.metric("Chunks in Database", stats["chunks_in_database"])
        else:
            st.info("ChatBot başlatılmadı")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        st.header("💬 Sohbet")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:  # User message
                    display_chat_message(message, is_user=True)
                else:  # Assistant message
                    display_chat_message(message, is_user=False)
        
        # Chat input
        user_input = st.text_input(
            "Sorunuzu yazın:",
            placeholder="Örn: Şirketin 2024 yılı net kârı ne kadar?",
            key="user_input"
        )
        
        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("📤 Gönder", type="primary"):
                if not st.session_state.model_initialized:
                    st.error("❌ Önce ChatBot'u başlatın (Sol menüden)")
                elif user_input.strip():
                    with st.spinner("Cevap üretiliyor..."):
                        response = st.session_state.chatbot.chat(user_input)
                        st.session_state.chat_history.extend([user_input, response])
                        st.rerun()
        
        with col_clear:
            if st.button("🗑️ Sohbeti Temizle"):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.chat_history = []
                st.rerun()
    
    with col2:
        # File upload section
        st.header("📁 PDF Dosyası Yükle")
        
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
                    result = st.session_state.chatbot.process_pdf_file(uploaded_file)
                    if "✅" in result:
                        st.success(result)
                    else:
                        st.error(result)
            else:
                st.warning("Lütfen bir PDF dosyası seçin")
        
        # Usage guide
        st.header("ℹ️ Kullanım Rehberi")
        
        with st.expander("🚀 Nasıl Kullanılır?"):
            st.markdown("""
            ### 1. Model Seçimi
            - Sol menüden istediğiniz modeli seçin
            - **Önerilen:** Llama 3.1 8B (dengeli performans)
            - **Hızlı:** Mistral 7B (düşük memory)
            - "🚀 ChatBot'u Başlat" butonuna tıklayın
            
            ### 2. PDF Yükleme  
            - PDF dosyanızı seçin ve "📤 Yükle ve İşle" butonuna tıklayın
            - İşlem tamamlanana kadar bekleyin
            
            ### 3. Soru Sorma
            - Sorunuzu metin kutusuna yazın
            - "📤 Gönder" butonuna tıklayın
            """)
        
        with st.expander("🎯 Model Önerileri"):
            st.markdown(f"""
            **🏆 Llama 3.1 8B** (Önerilen)
            - Memory: ~16GB
            - Türkçe desteği mükemmel
            - Finansal analiz için optimize
            
            **⚡ Mistral 7B** (Hızlı)
            - Memory: ~14GB  
            - Daha hızlı inference
            - Kod üretme konusunda güçlü
            
            **💪 Llama 3.1 70B** (Güçlü)
            - Memory: ~40GB
            - En iyi reasoning
            - Sadece yüksek GPU memory'de çalışır
            
            **Local LLM Mevcut:** {"✅ Evet" if LOCAL_LLM_AVAILABLE else "❌ Hayır"}
            """)

if __name__ == "__main__":
    main() 