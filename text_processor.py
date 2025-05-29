#!/usr/bin/env python3
"""
Text Processing Module
Handles text chunking and embedding generation for RAG pipeline
Optimized for Turkish and English content processing
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Container for text chunks with metadata"""
    content: str
    chunk_id: str
    source_file: str
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict

@dataclass
class EmbeddedChunk:
    """Text chunk with embedding vector"""
    chunk: TextChunk
    embedding: np.ndarray
    embedding_model: str

class TextChunker:
    """
    Intelligent text chunking with overlap strategy
    Optimized for financial documents and Turkish language
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_size: int = 200,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Turkish sentence endings
        self.sentence_endings = r'[.!?]+\s+'
        self.paragraph_break = r'\n\s*\n'
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content
        """
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove encoding artifacts (common in PDF extraction)
        text = re.sub(r'\(cid:\d+\)', '', text)
        
        # Fix common PDF extraction issues
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('ı', 'ı').replace('İ', 'İ')  # Turkish specific
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        
        return text.strip()
    
    def find_sentence_boundaries(self, text: str) -> List[int]:
        """
        Find sentence boundaries for intelligent chunking
        """
        boundaries = []
        
        # Find sentence endings
        for match in re.finditer(self.sentence_endings, text):
            boundaries.append(match.end())
        
        # Find paragraph breaks
        for match in re.finditer(self.paragraph_break, text):
            boundaries.append(match.start())
            
        return sorted(set(boundaries))
    
    def chunk_text(self, text: str, source_file: str, page_number: int) -> List[TextChunk]:
        """
        Split text into overlapping chunks with intelligent boundaries
        """
        cleaned_text = self.clean_text(text)
        if len(cleaned_text) < self.min_chunk_size:
            return []
            
        chunks = []
        sentence_boundaries = self.find_sentence_boundaries(cleaned_text)
        
        start_pos = 0
        chunk_counter = 0
        
        while start_pos < len(cleaned_text):
            # Calculate end position
            end_pos = min(start_pos + self.chunk_size, len(cleaned_text))
            
            # Try to end at sentence boundary if possible
            if end_pos < len(cleaned_text):
                suitable_boundaries = [b for b in sentence_boundaries 
                                     if start_pos + self.chunk_size // 2 <= b <= end_pos]
                if suitable_boundaries:
                    end_pos = suitable_boundaries[-1]
            
            # Extract chunk content
            chunk_content = cleaned_text[start_pos:end_pos].strip()
            
            if len(chunk_content) >= self.min_chunk_size:
                chunk_id = f"{source_file}_p{page_number}_c{chunk_counter}"
                
                chunk = TextChunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_number=page_number,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={
                        'chunk_length': len(chunk_content),
                        'word_count': len(chunk_content.split()),
                        'language': self.detect_language(chunk_content)
                    }
                )
                chunks.append(chunk)
                chunk_counter += 1
            
            # Move to next position with overlap
            if end_pos >= len(cleaned_text):
                break
                
            # Calculate next start position with overlap
            next_start = end_pos - self.overlap_size
            
            # Find suitable overlap boundary
            overlap_boundaries = [b for b in sentence_boundaries 
                                if end_pos - self.overlap_size * 2 <= b <= next_start]
            if overlap_boundaries:
                next_start = overlap_boundaries[-1]
                
            start_pos = max(next_start, start_pos + self.min_chunk_size)
        
        logger.info(f"Created {len(chunks)} chunks for page {page_number}")
        return chunks
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection for Turkish vs English
        """
        turkish_chars = 'çğıöşüÇĞIÖŞÜ'
        turkish_count = sum(1 for char in text if char in turkish_chars)
        
        # If more than 1% Turkish characters, assume Turkish
        if len(text) > 0 and (turkish_count / len(text)) > 0.01:
            return 'tr'
        return 'en'

class EmbeddingService:
    """
    Embedding generation service supporting multiple models
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """
        Load the embedding model
        """
        try:
            # Try to import sentence transformers
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("✅ Embedding model loaded successfully")
            
        except ImportError:
            logger.warning("SentenceTransformers not available, using dummy embeddings")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text
        """
        if self.model is None:
            # Return dummy embedding for testing
            logger.warning("Using dummy embedding")
            return np.random.rand(384).astype(np.float32)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.random.rand(384).astype(np.float32)
    
    def embed_chunks(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        """
        Generate embeddings for a list of text chunks
        """
        embedded_chunks = []
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        for chunk in chunks:
            try:
                embedding = self.generate_embedding(chunk.content)
                
                embedded_chunk = EmbeddedChunk(
                    chunk=chunk,
                    embedding=embedding,
                    embedding_model=self.model_name
                )
                embedded_chunks.append(embedded_chunk)
                
            except Exception as e:
                logger.error(f"Failed to embed chunk {chunk.chunk_id}: {e}")
                continue
        
        logger.info(f"✅ Generated {len(embedded_chunks)} embeddings")
        return embedded_chunks

class TextProcessor:
    """
    Main text processing pipeline combining chunking and embedding
    """
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 overlap_size: int = 200,
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        
        self.chunker = TextChunker(chunk_size=chunk_size, overlap_size=overlap_size)
        self.embedding_service = EmbeddingService(model_name=embedding_model)
    
    def process_document_pages(self, pages: List, source_file: str) -> List[EmbeddedChunk]:
        """
        Process all pages from a document and return embedded chunks
        """
        all_chunks = []
        
        for page_content in pages:
            # Extract text content (handling different input formats)
            if hasattr(page_content, 'text'):
                text = page_content.text
                page_num = page_content.page_number
            else:
                text = str(page_content)
                page_num = 1
            
            # Create chunks
            page_chunks = self.chunker.chunk_text(text, source_file, page_num)
            all_chunks.extend(page_chunks)
        
        # Generate embeddings
        embedded_chunks = self.embedding_service.embed_chunks(all_chunks)
        
        return embedded_chunks
    
    def process_excel_sheets(self, sheets: List, source_file: str) -> List[EmbeddedChunk]:
        """
        Process Excel sheets and return embedded chunks
        Each sheet becomes a separate 'page' for chunking
        """
        all_chunks = []
        
        for sheet_idx, sheet_content in enumerate(sheets, 1):
            # Handle ExcelSheetContent objects
            if hasattr(sheet_content, 'text_content'):
                text = sheet_content.text_content
                sheet_name = sheet_content.sheet_name
                page_num = sheet_idx
            else:
                # Fallback for other formats
                text = str(sheet_content)
                sheet_name = f"Sheet_{sheet_idx}"
                page_num = sheet_idx
            
            # Create chunks with Excel-specific metadata
            sheet_chunks = self.chunker.chunk_text(text, source_file, page_num)
            
            # Add Excel-specific metadata to chunks
            for chunk in sheet_chunks:
                chunk.metadata.update({
                    'content_type': 'excel',
                    'sheet_name': sheet_name,
                    'sheet_index': sheet_idx
                })
            
            all_chunks.extend(sheet_chunks)
        
        # Generate embeddings
        embedded_chunks = self.embedding_service.embed_chunks(all_chunks)
        
        logger.info(f"Created {len(embedded_chunks)} embedded chunks from {len(sheets)} Excel sheets")
        return embedded_chunks
    
    def get_processing_stats(self, embedded_chunks: List[EmbeddedChunk]) -> Dict:
        """
        Generate processing statistics
        """
        if not embedded_chunks:
            return {}
        
        total_chunks = len(embedded_chunks)
        total_characters = sum(len(chunk.chunk.content) for chunk in embedded_chunks)
        avg_chunk_size = total_characters / total_chunks if total_chunks > 0 else 0
        
        # Language distribution
        languages = [chunk.chunk.metadata.get('language', 'unknown') 
                    for chunk in embedded_chunks]
        lang_dist = {lang: languages.count(lang) for lang in set(languages)}
        
        return {
            'total_chunks': total_chunks,
            'total_characters': total_characters,
            'avg_chunk_size': avg_chunk_size,
            'language_distribution': lang_dist,
            'embedding_dimension': embedded_chunks[0].embedding.shape[0] if embedded_chunks else 0
        }

def test_text_processor():
    """
    Test the text processing pipeline
    """
    # Import PDF processor
    from pdf_processor import PDFProcessor
    
    processor = TextProcessor(chunk_size=800, overlap_size=150)
    pdf_processor = PDFProcessor()
    
    # Test with one of our sample PDFs
    test_pdf = "pdf/KAP - Pegasus Özel Finansal Bilgiler.pdf"
    
    try:
        logger.info(f"Testing with: {test_pdf}")
        
        # Process PDF
        pdf_result = pdf_processor.process_pdf(test_pdf)
        logger.info(f"PDF processed: {len(pdf_result.pages)} pages")
        
        # Process first few pages for testing
        test_pages = pdf_result.pages[:3]  # Test with first 3 pages
        
        # Generate embedded chunks
        embedded_chunks = processor.process_document_pages(test_pages, test_pdf)
        
        # Get statistics
        stats = processor.get_processing_stats(embedded_chunks)
        
        print(f"\n📊 Text Processing Results:")
        print(f"   📄 Pages processed: {len(test_pages)}")
        print(f"   🧩 Total chunks: {stats['total_chunks']}")
        print(f"   📝 Total characters: {stats['total_characters']:,}")
        print(f"   📏 Average chunk size: {stats['avg_chunk_size']:.1f}")
        print(f"   🌍 Languages: {stats['language_distribution']}")
        print(f"   🔢 Embedding dimension: {stats['embedding_dimension']}")
        
        # Show sample chunk
        if embedded_chunks:
            sample_chunk = embedded_chunks[0]
            print(f"\n📝 Sample chunk:")
            print(f"   ID: {sample_chunk.chunk.chunk_id}")
            print(f"   Length: {len(sample_chunk.chunk.content)} chars")
            print(f"   Content preview: {sample_chunk.chunk.content[:150]}...")
            print(f"   Embedding shape: {sample_chunk.embedding.shape}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")

if __name__ == "__main__":
    test_text_processor() 