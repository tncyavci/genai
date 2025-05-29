# 📄💬 PDF ChatBot - Finansal Doküman Analizi

Finansal PDF dokümanlarını yükleyip, AI ile doğal dil sorguları yapabileceğiniz akıllı chatbot uygulaması. Google Colab Pro Plus ile optimize edilmiş RAG (Retrieval Augmented Generation) mimarisi.

## 🌟 Özellikler

### 📋 PDF İşleme
- **Çoklu Format Desteği**: pdfplumber ve PyPDF2 ile güçlü extraction
- **Tablo Algılama**: Finansal tablolar otomatik olarak tespit edilir
- **Türkçe Desteği**: Türkçe karakterler ve finansal terimler optimize edilmiş
- **Metadata Tracking**: Sayfa numaraları ve kaynak takibi

### 🧠 Akıllı Text Processing
- **Smart Chunking**: Cümle sınırlarını dikkate alan intelligent chunking
- **Embedding Generation**: Çok dilli sentence transformers
- **Overlap Strategy**: Context korunumu için overlap tekniği
- **Language Detection**: Türkçe/İngilizce otomatik tanıma

### 🔍 Vector Search
- **ChromaDB Integration**: Persistent vector database
- **Similarity Search**: Cosine similarity ile relevance scoring
- **Metadata Filtering**: Dil, sayfa, dosya bazlı filtreleme
- **Batch Processing**: Büyük dokümanlar için optimize edilmiş

### 💬 Chat Interface
- **Gradio UI**: Modern, responsive web interface
- **Chat History**: Conversation context tracking
- **Source References**: Cevaplarda kaynak belirtimi
- **Real-time Stats**: Doküman ve chat istatistikleri

## 🏗️ Mimari

```
📁 PDF ChatBot Project
├── 📄 pdf_processor.py      # PDF extraction & table detection
├── 🔤 text_processor.py     # Chunking & embedding generation  
├── 🗃️ vector_store.py       # ChromaDB vector operations
├── 🤖 chat_bot.py          # Gradio UI & chat logic
├── ⚙️ setup_requirements.py # Dependency installer
└── 📋 README.md            # Documentation
```

### Clean Architecture Principles
- **Separation of Concerns**: Her modül tek sorumluluğa sahip
- **Dependency Injection**: Loosely coupled components
- **Interface Segregation**: Clear, focused interfaces
- **Single Responsibility**: Her sınıf tek bir görevi yerine getirir

## 🚀 Kurulum

### 1. Depo Klonlama
```bash
git clone <repository-url>
cd pdf-chatbot
```

### 2. Dependencies Yükleme
```bash
# Otomatik kurulum
python setup_requirements.py

# Manuel kurulum
pip install pdfplumber PyPDF2 pandas numpy
pip install sentence-transformers chromadb
pip install gradio openai
```

### 3. Environment Setup (İsteğe Bağlı)
```bash
# OpenAI API key için .env dosyası oluştur
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

## 📱 Kullanım

### 1. ChatBot Başlatma
```bash
python chat_bot.py
```

Interface `http://localhost:7860` adresinde açılacaktır.

### 2. PDF Yükleme
- "📁 Dosya Yükle" sekmesine gidin
- PDF dosyanızı seçin
- "📤 Yükle ve İşle" butonuna tıklayın
- İşlem tamamlanana kadar bekleyin

### 3. Soru Sorma
- "💬 Sohbet" sekmesine geçin
- Sorunuzu yazın ve "📤 Gönder" butonuna tıklayın
- AI cevabı kaynaklarıyla birlikte görüntüleyecektir

## 🔧 Google Colab Entegrasyonu

### Colab Notebook Setup
```python
# 1. Repository clone
!git clone <your-repo-url>
%cd pdf-chatbot

# 2. Dependencies install
!pip install -r requirements.txt

# 3. Start chatbot with public URL
import subprocess
import threading

def start_chatbot():
    subprocess.run(["python", "chat_bot.py"])

# Background thread'de başlat
thread = threading.Thread(target=start_chatbot)
thread.daemon = True
thread.start()

# Gradio public URL ile erişim
# Gradio interface'de share=True yapın
```

### Colab Pro Plus Optimizasyonları
```python
# GPU kullanımı için
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Memory optimization
torch.cuda.empty_cache()

# Büyük dosyalar için batch processing
chunk_size = 1500  # Colab Pro Plus için optimize
overlap_size = 300
```

## 📊 Test ve Performans

### Modül Test Etme
```bash
# PDF processor test
python pdf_processor.py

# Text processing test  
python text_processor.py

# Vector store test
python vector_store.py
```

### Performance Metrics
- **PDF Processing**: ~30 sayfa/dakika
- **Embedding Generation**: ~100 chunk/dakika  
- **Search Response**: <2 saniye
- **Memory Usage**: ~500MB (orta büyüklük dokümanlar)

## 🛠️ Konfigürasyon

### Embedding Model Değiştirme
```python
# text_processor.py içinde
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Alternatifler:
# "sentence-transformers/all-MiniLM-L6-v2"  # Hızlı
# "sentence-transformers/all-mpnet-base-v2" # Yüksek kalite
```

### Chunk Size Optimization
```python
# Farklı doküman türleri için
processor = TextProcessor(
    chunk_size=800,    # Finansal raporlar için
    overlap_size=150   # Context korunumu
)
```

### LLM Provider Değiştirme
```python
# chat_bot.py içinde OpenAI yerine
# Anthropic Claude, Google Gemini, veya local models kullanılabilir
```

## 🔍 Troubleshooting

### Yaygın Sorunlar

**PDF İşlenemiyor**
```python
# Encoding problems için
pdf_result = processor.process_pdf(path, method='hybrid')
```

**Embedding Model Yüklenmiyor**
```bash
# Model cache temizleme
rm -rf ~/.cache/huggingface/transformers/
```

**ChromaDB Bağlantı Hatası**
```python
# Database reset
vector_store.clear_collection()
```

**Memory Issues**
```python
# Smaller chunks için
processor = TextProcessor(chunk_size=500, overlap_size=100)
```

## 📈 İleri Seviye Kullanım

### Custom Embedding Functions
```python
class CustomEmbeddingService(EmbeddingService):
    def generate_embedding(self, text: str):
        # Custom embedding logic
        return custom_embedding_vector
```

### Multi-Document Search
```python
# Specific document filtering
context = retrieval_service.retrieve_context(
    query="finansal performans",
    filter_by_source="annual_report_2024.pdf",
    n_results=10
)
```

### Advanced Prompting
```python
system_prompt = """
Sen uzman bir finansal analistisin.
- Sadece verilen bağlamı kullan
- Sayısal verileri doğrula
- Risk faktörlerini belirt
- Trend analizleri yap
"""
```

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 License

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🔗 Faydalı Linkler

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Gradio Documentation](https://gradio.app/docs/)
- [pdfplumber Guide](https://github.com/jsvine/pdfplumber)

## 📞 Destek

Sorunlarınız için:
- GitHub Issues açın
- Documentation'ı kontrol edin
- Community discussions'a katılın

---

**⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!** 

# PDF & Excel ChatBot - Local LLM

**Finansal dokümanlar ve Excel dosyaları için Local LLM ile RAG sistemi**

Bu proje PDF dosyalarına ve Excel tablolarına (XLS/XLSX) sorular sorabileceğiniz, tamamen local çalışan bir ChatBot sistemidir.

## 🚀 Yeni Özellik: Excel Desteği!

Artık Excel dosyalarınızı da yükleyip analiz edebilirsiniz:
- ✅ .xls, .xlsx, .xlsm formatları
- ✅ Çoklu sayfa desteği  
- ✅ Otomatik sayısal analiz
- ✅ Aranabilir metin formatına dönüştürme

## ✨ Özellikler

### 📄 Desteklenen Dosya Formatları
- **PDF**: Metin ve tablo çıkarma
- **Excel**: XLS, XLSX, XLSM dosyaları
- **Çoklu sayfa**: Tüm sayfalar otomatik işlenir

### 🤖 LLM Desteği  
- **GGUF Modeller**: Mistral 7B (önerilen)
- **HuggingFace Modeller**: Llama 3.1 8B
- **OpenAI API**: Yedek seçenek

### 🧠 RAG Sistemi
- **Akıllı metin parçalama**: Overlap ile chunking
- **Vector Store**: ChromaDB ile embedding storage  
- **Çokdilli**: Türkçe ve İngilizce desteği
- **Metadata**: Kaynak dosya ve sayfa takibi

## 📊 Excel İşleme Özellikleri

- **Tüm sayfaları okur**: Multi-sheet Excel dosyaları
- **Sayısal analiz**: Otomatik istatistikler (toplam, ortalama, min, max)
- **Metin dönüştürme**: Aranabilir format
- **Metadata korunması**: Sütun adları ve veri tipleri
- **Temizleme**: Boş satır/sütunları otomatik kaldırır 