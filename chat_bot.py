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
        """
        Generate response using available LLM service
        """
        
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
    """
    Main ChatBot class with Local LLM support
    """
    
    def __init__(self, use_local_llm: bool = True, model_choice: str = "llama_3_1_8b"):
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_processor = TextProcessor(chunk_size=800, overlap_size=150)
        self.vector_store = VectorStore()
        self.retrieval_service = RetrievalService(
            vector_store=self.vector_store,
            embedding_service=self.text_processor.embedding_service
        )
        
        # Initialize LLM service
        self.llm_service = LLMService(
            use_local=use_local_llm,
            model_choice=model_choice
        )
        
        # Chat state
        self.chat_history: List[ChatMessage] = []
        self.processed_files: List[str] = []
        
        logger.info("✅ PDF ChatBot initialized")
    
    def process_pdf_file(self, pdf_file) -> str:
        """Process uploaded PDF file and add to vector store"""
        if pdf_file is None:
            return "❌ Lütfen bir PDF dosyası yükleyin."
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.read())
                temp_path = tmp_file.name
            
            # Process PDF
            logger.info(f"Processing uploaded PDF...")
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
            
            summary = f"""
            ✅ **PDF başarıyla işlendi!**
            
            📊 **İstatistikler:**
            - 📄 Sayfa sayısı: {pdf_result.total_pages}
            - 🧩 Chunk sayısı: {stats['total_chunks']}
            - 📝 Toplam karakter: {stats['total_characters']:,}
            - 🌍 Dil: {', '.join(stats['language_distribution'].keys())}
            
            💬 Artık bu dokümana dair sorular sorabilirsiniz!
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
    
    def get_stats(self) -> str:
        """Get current chatbot statistics"""
        try:
            retrieval_stats = self.retrieval_service.get_retrieval_stats()
            vector_stats = retrieval_stats['vector_store_stats']
            llm_info = self.llm_service.get_service_info()
            
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
            - İşlenen dosyalar: {len(self.processed_files)}
            - Dosya listesi: {', '.join(self.processed_files) if self.processed_files else 'Henüz dosya yüklenmedi'}
            
            **💬 Sohbet Bilgileri:**
            - Toplam mesaj: {len(self.chat_history)}
            - Embedding model: {retrieval_stats['embedding_model']}
            
            **🔤 Dil Dağılımı:**
            {json.dumps(vector_stats.get('language_distribution', {}), indent=2)}
            """
            
            return stats_text
            
        except Exception as e:
            return f"❌ İstatistik alınamadı: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_initialized' not in st.session_state:
        st.session_state.model_initialized = False
    
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
            try:
                stats_text = st.session_state.chatbot.get_stats()
                st.markdown(stats_text)
            except:
                st.info("İstatistik alınamadı")
        else:
            st.info("ChatBot başlatılmadı")
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["💬 Sohbet", "📁 PDF Yükle", "ℹ️ Rehber"])
    
    with tab1:
        # Chat interface
        st.header("💬 Sohbet")
        
        # Display chat history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(bot_msg)
        
        # Chat input
        if prompt := st.chat_input("Sorunuzu yazın..."):
            if not st.session_state.model_initialized:
                st.error("❌ Önce ChatBot'u başlatın (Sol menüden)")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append([prompt, ""])
                
                with st.spinner("Cevap üretiliyor..."):
                    try:
                        # Get response
                        _, updated_history = st.session_state.chatbot.chat(prompt, st.session_state.chat_history)
                        st.session_state.chat_history = updated_history
                        st.rerun()
                    except Exception as e:
                        st.error(f"Hata: {e}")
        
        # Clear chat button
        if st.button("🗑️ Sohbeti Temizle"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.chat_history = []
            st.rerun()
    
    with tab2:
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
                    try:
                        result = st.session_state.chatbot.process_pdf_file(uploaded_file)
                        if "✅" in result:
                            st.success(result)
                        else:
                            st.error(result)
                    except Exception as e:
                        st.error(f"PDF işleme hatası: {e}")
            else:
                st.warning("Lütfen bir PDF dosyası seçin")
    
    with tab3:
        # Usage guide
        st.header("ℹ️ Kullanım Rehberi")
        
        st.markdown("""
        ## 🚀 Nasıl Kullanılır?
        
        ### 1. Model Seçimi
        - Sol menüden istediğiniz modeli seçin
        - **Önerilen:** Llama 3.1 8B (dengeli performans)
        - **Hızlı:** Mistral 7B (düşük memory)
        - "🚀 ChatBot'u Başlat" butonuna tıklayın
        
        ### 2. PDF Yükleme  
        - "📁 PDF Yükle" sekmesinden PDF dosyanızı seçin
        - "📤 Yükle ve İşle" butonuna tıklayın
        - İşlem tamamlanana kadar bekleyin
        
        ### 3. Soru Sorma
        - "💬 Sohbet" sekmesine gidin
        - Sorunuzu chat input'a yazın
        - Enter'a basın veya gönder butonuna tıklayın
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
    # Check for required libraries
    try:
        import streamlit
        logger.info("✅ Streamlit library found")
    except ImportError:
        logger.error("❌ Streamlit not installed. Run: pip install streamlit")
        exit(1)
    
    # Run the main app
    main() 