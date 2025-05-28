#!/usr/bin/env python3
"""
Local LLM Service for PDF ChatBot
Supports Llama and Mistral models via HuggingFace Transformers
Optimized for Google Colab Pro Plus
"""

import logging
import torch
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Container for chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[str]] = None

class LocalLLMService:
    """
    Local LLM service supporting Llama and Mistral models
    Optimized for financial document analysis in Turkish
    """
    
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 device: str = "auto",
                 max_new_tokens: int = 512,
                 temperature: float = 0.7):
        
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the local LLM model"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            
            logger.info(f"🚀 Loading model: {self.model_name}")
            
            # Check GPU availability
            if torch.cuda.is_available():
                device_map = "auto"
                torch_dtype = torch.float16
                logger.info(f"✅ Using GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                device_map = "mps"
                torch_dtype = torch.float16
                logger.info("✅ Using Apple Silicon MPS")
            else:
                device_map = "cpu"
                torch_dtype = torch.float32
                logger.info("⚠️ Using CPU (will be slow)")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=False,  # 8-bit quantization if memory limited
                load_in_4bit=False   # 4-bit quantization for very limited memory
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("✅ Local LLM model loaded successfully")
            
        except ImportError:
            logger.error("❌ Transformers library not installed. Run: pip install transformers accelerate")
            self.pipeline = None
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.pipeline = None
    
    def generate_response(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """
        Generate response using local LLM
        """
        if self.pipeline is None:
            return self._generate_fallback_response(query, context)
        
        try:
            # Build prompt based on model type
            if "llama" in self.model_name.lower():
                prompt = self._build_llama_prompt(query, context, chat_history)
            elif "mistral" in self.model_name.lower():
                prompt = self._build_mistral_prompt(query, context, chat_history)
            else:
                prompt = self._build_generic_prompt(query, context, chat_history)
            
            # Generate response
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Clean response (remove prompt part)
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # Post-process response
            response = self._post_process_response(response)
            
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._generate_fallback_response(query, context)
    
    def _build_llama_prompt(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """Build prompt for Llama models"""
        
        system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. Sana verilen bağlam bilgilerini kullanarak kullanıcının sorularını Türkçe olarak cevapla.

Kurallar:
- Sadece verilen bağlam bilgilerini kullan
- Eğer bağlamda cevap yoksa "Bu bilgi dokümanlarda bulunmuyor" de
- Finansal terimleri doğru kullan
- Sayısal verileri dikkatli kontrol et
- Kısa ve net cevaplar ver
- Kaynak belirtmeye gerek yok (otomatik eklenecek)"""

        # Llama 3.1 format
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history
        if chat_history:
            for msg in chat_history[-3:]:  # Last 3 messages
                messages.append({
                    "role": msg.role, 
                    "content": msg.content
                })
        
        # Add current query with context
        user_message = f"""Bağlam Bilgileri:
{context}

Soru: {query}"""
        
        messages.append({"role": "user", "content": user_message})
        
        # Format as Llama prompt
        prompt = "<|begin_of_text|>"
        for message in messages:
            prompt += f"<|start_header_id|>{message['role']}<|end_header_id|>\n\n{message['content']}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return prompt
    
    def _build_mistral_prompt(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """Build prompt for Mistral models"""
        
        system_prompt = """Sen finansal dokümanları analiz eden uzman bir asistansın. Verilen bağlam bilgilerini kullanarak Türkçe cevap ver.

Kurallar:
- Sadece bağlam bilgilerini kullan
- Finansal terimleri doğru kullan
- Kısa ve net ol"""

        # Mistral format
        prompt = f"<s>[INST] {system_prompt}\n\nBağlam:\n{context}\n\nSoru: {query} [/INST]"
        
        return prompt
    
    def _build_generic_prompt(self, query: str, context: str, chat_history: List[ChatMessage] = None) -> str:
        """Generic prompt builder"""
        
        prompt = f"""Sen finansal doküman uzmanısın. Bağlam bilgilerini kullanarak Türkçe cevap ver.

Bağlam:
{context}

Soru: {query}

Cevap:"""
        
        return prompt
    
    def _post_process_response(self, response: str) -> str:
        """Post-process the generated response"""
        
        # Remove common artifacts
        response = response.replace("<|eot_id|>", "")
        response = response.replace("</s>", "")
        response = response.replace("[/INST]", "")
        
        # Clean up extra whitespace
        response = response.strip()
        
        # Limit length
        if len(response) > 1000:
            sentences = response.split('.')
            limited_response = ""
            for sentence in sentences:
                if len(limited_response + sentence) < 900:
                    limited_response += sentence + "."
                else:
                    break
            response = limited_response
        
        return response
    
    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Fallback response when model fails"""
        if context:
            return f"""
**[Local Model Mevcut Değil]**

Sorunuz: {query}

Bulunan İlgili Bilgiler:
{context[:500]}...

💡 Local model yüklendiğinde bu bilgileri analiz ederek detaylı cevap verebilirim.
"""
        else:
            return "Bu konuda dokümanlarda ilgili bilgi bulunamadı."
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        
        info = {
            "model_name": self.model_name,
            "model_loaded": self.pipeline is not None,
            "device": str(self.device),
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature
        }
        
        if torch.cuda.is_available():
            info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            info["gpu_name"] = torch.cuda.get_device_name()
        
        return info

# Model configurations
RECOMMENDED_MODELS = {
    "llama_3_1_8b": {
        "name": "meta-llama/Llama-3.1-8B-Instruct", 
        "memory": "~16GB",
        "description": "En dengeli seçim - Türkçe ve finansal analiz için optimize (HF Token gerekli)"
    },
    "mistral_7b": {
        "name": "mistralai/Mistral-7B-Instruct-v0.3",
        "memory": "~14GB", 
        "description": "Hızlı ve efficient - Düşük memory kullanımı"
    },
    "llama_3_2_3b": {
        "name": "meta-llama/Llama-3.2-3B-Instruct",
        "memory": "~6GB",
        "description": "Hafif model - Düşük GPU memory için (HF Token gerekli)"
    },
    "phi_3_mini": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "memory": "~8GB",
        "description": "Microsoft'un compact modeli - Hızlı ve verimli"
    },
    "gpt2_turkish": {
        "name": "redrussianarmy/gpt2-turkish-cased",
        "memory": "~2GB",
        "description": "Türkçe GPT-2 modeli - Çok hafif ama basit"
    }
}

def create_optimized_llm_service(model_choice: str = "llama_3_1_8b") -> LocalLLMService:
    """
    Create optimized LLM service based on choice
    """
    
    if model_choice not in RECOMMENDED_MODELS:
        logger.warning(f"Unknown model choice: {model_choice}. Using default.")
        model_choice = "llama_3_1_8b"
    
    model_config = RECOMMENDED_MODELS[model_choice]
    
    logger.info(f"🎯 Creating LLM service with {model_config['name']}")
    logger.info(f"📊 Expected memory usage: {model_config['memory']}")
    logger.info(f"📝 Description: {model_config['description']}")
    
    return LocalLLMService(
        model_name=model_config["name"],
        max_new_tokens=512,
        temperature=0.7
    )

if __name__ == "__main__":
    # Test the service
    logger.info("🧪 Testing Local LLM Service...")
    
    # Test with Llama 3.1 8B
    llm_service = create_optimized_llm_service("llama_3_1_8b")
    
    # Test query
    test_context = "Şirketin 2024 yılı net kârı 1.2 milyar TL olarak gerçekleşmiştir."
    test_query = "Şirketin 2024 performansı nasıl?"
    
    response = llm_service.generate_response(test_query, test_context)
    
    print(f"\n📝 Test Query: {test_query}")
    print(f"🤖 Response: {response}")
    
    # Model info
    info = llm_service.get_model_info()
    print(f"\n📊 Model Info: {info}") 