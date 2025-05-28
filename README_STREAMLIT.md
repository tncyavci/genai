# 📄💬 PDF ChatBot - Streamlit Version

**Local LLM Destekli PDF Analiz ve Soru-Cevap Sistemi**

## 🎉 Gradio'dan Streamlit'e Geçiş

Gradio'daki sürekli dependency conflict sorunları nedeniyle **Streamlit**'e geçtik!

### ✅ Streamlit Avantajları:
- **No dependency conflicts** (en büyük avantaj!)
- Daha stabil Google Colab desteği
- Daha iyi dosya upload yönetimi
- Temiz ve modern UI
- Daha iyi session management
- Responsive tasarım

## 🚀 Hızlı Başlangıç

### 1. **Local Kurulum**
```bash
# Gereksinimler
pip install -r requirements_colab.txt

# Başlat
python start_streamlit.py
# veya
streamlit run chat_bot_streamlit.py
```

### 2. **Google Colab**
```python
# Setup çalıştır
!python colab_setup.py

# App'i başlat
!streamlit run chat_bot_streamlit.py &
```

## 📁 Proje Dosyaları

### 🆕 Streamlit Versiyonu:
- `chat_bot_streamlit.py` - Ana Streamlit uygulaması
- `start_streamlit.py` - Kolay başlatma scripti
- `requirements_colab.txt` - Streamlit requirements (gradio kaldırıldı)
- `colab_setup.py` - Streamlit için güncellenmiş setup
- `README_STREAMLIT.md` - Bu dosya

### 📚 Core Modüller:
- `pdf_processor.py` - PDF metin çıkarma
- `text_processor.py` - Metin işleme ve chunking
- `vector_store.py` - Vector database (ChromaDB)
- `llm_service_local.py` - Local LLM servisi

### 🗑️ Kaldırılan (Gradio):
- ~~`chat_bot.py`~~ - Gradio versiyonu (artık kullanmıyoruz)
- ~~`gradio_emergency_fix.py`~~ - Artık gerekmiyor!

## 🎯 Desteklenen Modeller

### **🏆 Llama 3.1 8B** (Önerilen)
- Memory: ~16GB
- Türkçe desteği mükemmel
- Finansal analiz için optimize
- HF Model: `meta-llama/Meta-Llama-3.1-8B-Instruct`

### **⚡ Mistral 7B** (Hızlı)
- Memory: ~14GB  
- Daha hızlı inference
- Avrupa dilleri desteği güçlü
- HF Model: `mistralai/Mistral-7B-Instruct-v0.3`

### **💪 Llama 3.1 70B** (En Güçlü)
- Memory: ~40GB
- En iyi reasoning yeteneği
- Sadece yüksek GPU memory'de çalışır
- HF Model: `meta-llama/Meta-Llama-3.1-70B-Instruct`

## 💻 Kullanım

### 1. **Model Seçimi**
- Sol sidebar'dan model seçin
- "🚀 ChatBot'u Başlat" butonuna tıklayın
- Model yüklenene kadar bekleyin (2-5 dakika)

### 2. **PDF Yükleme**
- Sağ panelden PDF dosyasını seçin
- "📤 Yükle ve İşle" butonuna tıklayın
- İşlem tamamlanana kadar bekleyin

### 3. **Soru Sorma**
- Ana chat alanından sorunuzu yazın
- "📤 Gönder" butonuna tıklayın
- AI cevabını bekleyin

## 🔧 Teknik Detaylar

### **Mimari:**
```
Frontend (Streamlit) 
    ↓
Main App (chat_bot_streamlit.py)
    ↓
LLM Service (llm_service_local.py)
    ↓
Vector Store (ChromaDB) + PDF Processing
```

### **Memory Gereksinimleri:**
- **Minimum:** 8GB RAM, 4GB GPU
- **Önerilen:** 16GB RAM, 16GB GPU
- **Optimal:** 32GB RAM, 24GB+ GPU

### **Desteklenen Formatlar:**
- PDF dosyaları (metin içerikli)
- Türkçe ve İngilizce içerik

## 🐛 Troubleshooting

### **Streamlit Başlamıyor:**
```python
# Port kontrolü
!lsof -i :8501

# Streamlit restart
!pkill -f streamlit
!streamlit run chat_bot_streamlit.py &
```

### **Model Yüklenmiyor:**
```python
# GPU kontrol
import torch
print(torch.cuda.is_available())

# Memory temizle
torch.cuda.empty_cache()
```

### **HuggingFace Hatası:**
```python
# Token gir
from huggingface_hub import login
login(token="your_token_here")
```

## 🔄 Gradio'dan Migration

Eğer eski gradio versiyonunu kullanıyordunuz:

1. **Streamlit'e geç:**
   ```bash
   pip uninstall gradio gradio-client
   pip install streamlit>=1.28.0
   ```

2. **Yeni app'i başlat:**
   ```bash
   streamlit run chat_bot_streamlit.py
   ```

3. **Avantajları:**
   - ✅ No more `handle_file` errors!
   - ✅ No more version conflicts!
   - ✅ Better UX and performance
   - ✅ More stable in Colab

## 📊 Performance

### **Benchmark (Google Colab Pro Plus):**

| Model | Loading Time | Query Response | Memory Usage |
|-------|-------------|----------------|--------------|
| Llama 3.1 8B | 3-5 min | 2-5 sec | ~16GB |
| Mistral 7B | 2-4 min | 1-3 sec | ~14GB |
| Llama 3.1 70B | 8-12 min | 5-10 sec | ~40GB |

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun
3. Streamlit best practices'i takip edin
4. Pull request gönderin

## 📄 Lisans

MIT License

## 🆘 Destek

Sorunlar için GitHub Issues kullanın:
- Streamlit specific sorunlar
- Model yükleme problemleri  
- PDF processing hataları
- UI/UX geliştirme önerileri

---

**🎉 Streamlit ile artık stable ve hızlı PDF chatbot deneyimi!** 