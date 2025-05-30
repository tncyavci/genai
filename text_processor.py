#!/usr/bin/env python3
"""
Text Processing Module
Handles text chunking, embedding, and metadata enrichment
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Container for text chunk data"""
    chunk_id: str
    content: str
    source_file: str
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]

@dataclass
class EmbeddedChunk:
    """Container for embedded chunk data"""
    chunk: TextChunk
    embedding: np.ndarray
    embedding_model: str

class TextProcessor:
    """Class for processing text and creating embeddings"""
    
    def __init__(self,
                 chunk_size: int = 300,  # Reduced from 500
                 overlap_size: int = 50,  # Reduced from 100
                 min_chunk_size: int = 50,  # New parameter
                 embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Initialize text processor
        
        Args:
            chunk_size: Maximum size of text chunks
            overlap_size: Size of overlap between chunks
            min_chunk_size: Minimum size for a chunk
            embedding_model: Name of the embedding model to use
        """
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model
        
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            
            # Initialize embedding model
            self.model = SentenceTransformer(embedding_model)
            
            logger.info(f"✅ Text processor initialized with model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to initialize text processor: {e}")
            raise
    
    def process_document_pages(self, pages: List[Dict], source_file: str) -> List[EmbeddedChunk]:
        """
        Process document pages and create embedded chunks
        
        Args:
            pages: List of document pages
            source_file: Source file name
            
        Returns:
            List of embedded chunks
        """
        try:
            embedded_chunks = []
            
            for page in pages:
                # Process text content
                text_chunks = self._create_text_chunks(
                    text=page['text'],
                    page_number=page['page_number'],
                    source_file=source_file
                )
                
                # Process tables if present
                if 'tables' in page:
                    table_chunks = self._process_tables(
                        tables=page['tables'],
                        page_number=page['page_number'],
                        source_file=source_file
                    )
                    text_chunks.extend(table_chunks)
                
                # Create embeddings for chunks
                for chunk in text_chunks:
                    embedding = self._create_embedding(chunk.content)
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=self.embedding_model
                    )
                    embedded_chunks.append(embedded_chunk)
            
            logger.info(f"✅ Processed {len(pages)} pages into {len(embedded_chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to process document pages: {e}")
            return []
    
    def _create_text_chunks(self, text: str, page_number: int, source_file: str) -> List[TextChunk]:
        """
        Create text chunks using improved chunking strategy
        
        Args:
            text: Text to chunk
            page_number: Page number
            source_file: Source file name
            
        Returns:
            List of text chunks
        """
        try:
            chunks = []
            current_pos = 0
            
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence_length = len(sentence)
                
                # Check if adding this sentence would exceed chunk size
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    # Create chunk if it meets minimum size
                    if current_length >= self.min_chunk_size:
                        chunk_content = ' '.join(current_chunk)
                        chunk = self._create_chunk(
                            content=chunk_content,
                            start_pos=current_pos,
                            end_pos=current_pos + len(chunk_content),
                            page_number=page_number,
                            source_file=source_file,
                            chunk_type='text'
                        )
                        chunks.append(chunk)
                    
                    # Keep overlap for context
                    overlap_start = max(0, len(current_chunk) - self.overlap_size)
                    current_chunk = current_chunk[overlap_start:]
                    current_length = sum(len(s) for s in current_chunk)
                    current_pos += len(' '.join(current_chunk[:overlap_start]))
                
                current_chunk.append(sentence)
                current_length += sentence_length
            
            # Add remaining text as final chunk
            if current_chunk:
                chunk_content = ' '.join(current_chunk)
                if len(chunk_content) >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        content=chunk_content,
                        start_pos=current_pos,
                        end_pos=current_pos + len(chunk_content),
                        page_number=page_number,
                        source_file=source_file,
                        chunk_type='text'
                    )
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to create text chunks: {e}")
            return []
    
    def _process_tables(self, tables: List[Dict], page_number: int, source_file: str) -> List[TextChunk]:
        """
        Process tables and create chunks
        
        Args:
            tables: List of tables
            page_number: Page number
            source_file: Source file name
            
        Returns:
            List of text chunks
        """
        try:
            chunks = []
            
            for i, table in enumerate(tables):
                # Process table header
                if 'header' in table:
                    header_chunk = self._create_chunk(
                        content=table['header'],
                        start_pos=0,
                        end_pos=len(table['header']),
                        page_number=page_number,
                        source_file=source_file,
                        chunk_type='table_header',
                        table_id=f"table_{i}",
                        is_table_fragment=True
                    )
                    chunks.append(header_chunk)
                
                # Process table rows
                if 'rows' in table:
                    for j, row in enumerate(table['rows']):
                        row_chunk = self._create_chunk(
                            content=str(row),
                            start_pos=0,
                            end_pos=len(str(row)),
                            page_number=page_number,
                            source_file=source_file,
                            chunk_type='table_row',
                            table_id=f"table_{i}",
                            row_index=j,
                            is_table_fragment=True
                        )
                        chunks.append(row_chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to process tables: {e}")
            return []
    
    def _create_chunk(self,
                     content: str,
                     start_pos: int,
                     end_pos: int,
                     page_number: int,
                     source_file: str,
                     chunk_type: str,
                     **kwargs) -> TextChunk:
        """
        Create a text chunk with enhanced metadata
        
        Args:
            content: Chunk content
            start_pos: Start position in original text
            end_pos: End position in original text
            page_number: Page number
            source_file: Source file name
            chunk_type: Type of chunk (text/table_header/table_row)
            **kwargs: Additional metadata
            
        Returns:
            TextChunk object
        """
        # Generate chunk ID
        chunk_id = f"{source_file}_p{page_number}_{start_pos}_{end_pos}"
        
        # Create base metadata
        metadata = {
            'chunk_type': chunk_type,
            'chunk_length': len(content),
            'word_count': len(content.split()),
            'language': self._detect_language(content),
            'content_type': self._detect_content_type(content),
            'has_numbers': bool(re.search(r'\d', content)),
            'has_currency': bool(re.search(r'[₺$€£]', content)),
            'has_percentage': bool(re.search(r'%', content))
        }
        
        # Add additional metadata
        metadata.update(kwargs)
        
        return TextChunk(
            chunk_id=chunk_id,
            content=content,
            source_file=source_file,
            page_number=page_number,
            start_char=start_pos,
            end_char=end_pos,
            metadata=metadata
        )
    
    def _create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            return self.model.encode(text)
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return np.zeros(384)  # Default embedding size for all-MiniLM-L6-v2
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code
        """
        # Simple language detection based on common words
        tr_words = ['ve', 'ile', 'bu', 'bir', 'de', 'da']
        en_words = ['and', 'the', 'this', 'that', 'in', 'on']
        
        tr_count = sum(1 for word in tr_words if word in text.lower())
        en_count = sum(1 for word in en_words if word in text.lower())
        
        return 'tr' if tr_count > en_count else 'en'
    
    def _detect_content_type(self, text: str) -> str:
        """
        Detect content type of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Content type
        """
        if re.search(r'\d+[.,]\d+\s*[₺$€£]', text):
            return 'financial_value'
        elif re.search(r'\d+%', text):
            return 'percentage'
        elif re.search(r'\d{4}-\d{2}-\d{2}', text):
            return 'date'
        elif re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:', text):
            return 'header'
        else:
            return 'text'
    
    def process_excel_sheets(self, sheets: List[Dict], source_file: str) -> List[EmbeddedChunk]:
        """
        Process Excel sheets and create embedded chunks
        
        Args:
            sheets: List of Excel sheets
            source_file: Source file name
            
        Returns:
            List of embedded chunks
        """
        try:
            embedded_chunks = []
            
            for sheet in sheets:
                # Process text content
                text_chunks = self._create_text_chunks(
                    text=sheet['text_content'],
                    page_number=sheet['sheet_index'],
                    source_file=source_file
                )
                
                # Create embeddings for chunks
                for chunk in text_chunks:
                    # Update chunk type for Excel
                    chunk.metadata['content_type'] = 'excel'
                    chunk.metadata['sheet_name'] = sheet['sheet_name']
                    
                    embedding = self._create_embedding(chunk.content)
                    embedded_chunk = EmbeddedChunk(
                        chunk=chunk,
                        embedding=embedding,
                        embedding_model=self.embedding_model
                    )
                    embedded_chunks.append(embedded_chunk)
            
            logger.info(f"✅ Processed {len(sheets)} Excel sheets into {len(embedded_chunks)} chunks")
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Failed to process Excel sheets: {e}")
            return []
    
    def get_processing_stats(self, embedded_chunks: List[EmbeddedChunk] = None) -> Dict:
        """
        Get processing statistics
        
        Args:
            embedded_chunks: List of embedded chunks to analyze
            
        Returns:
            Statistics dictionary
        """
        if not embedded_chunks:
            return {
                'total_chunks': 0,
                'text_chunks': 0,
                'table_chunks': 0,
                'total_characters': 0,
                'language_distribution': {},
                'content_type_distribution': {},
                'chunk_type_distribution': {}
            }
        
        stats = {
            'total_chunks': len(embedded_chunks),
            'text_chunks': 0,
            'table_chunks': 0,
            'total_characters': 0,
            'language_distribution': {},
            'content_type_distribution': {},
            'chunk_type_distribution': {},
            'pages_with_tables': 0
        }
        
        pages_with_tables = set()
        
        for embedded_chunk in embedded_chunks:
            chunk = embedded_chunk.chunk
            
            # Count characters
            stats['total_characters'] += len(chunk.content)
            
            # Count chunk types
            chunk_type = chunk.metadata.get('chunk_type', 'unknown')
            if chunk_type.startswith('table'):
                stats['table_chunks'] += 1
                pages_with_tables.add(chunk.page_number)
            else:
                stats['text_chunks'] += 1
            
            # Language distribution
            language = chunk.metadata.get('language', 'unknown')
            stats['language_distribution'][language] = stats['language_distribution'].get(language, 0) + 1
            
            # Content type distribution  
            content_type = chunk.metadata.get('content_type', 'unknown')
            stats['content_type_distribution'][content_type] = stats['content_type_distribution'].get(content_type, 0) + 1
            
            # Chunk type distribution
            stats['chunk_type_distribution'][chunk_type] = stats['chunk_type_distribution'].get(chunk_type, 0) + 1
        
        stats['pages_with_tables'] = len(pages_with_tables)
        
        return stats

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
        print(f"   📏 Average chunk size: {stats['total_characters'] / stats['total_chunks']:.1f}")
        print(f"   🌍 Languages: {stats['language_distribution']}")
        print(f"   🔢 Embedding dimension: {processor.model.get_sentence_embedding_dimension()}")
        
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