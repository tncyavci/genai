#!/usr/bin/env python3
"""
Vector Store and Retrieval Service
Handles document storage and similarity search using ChromaDB
Optimized for financial document retrieval and chat functionality
"""

import os
import logging
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results"""
    content: str
    chunk_id: str
    source_file: str
    page_number: int
    score: float
    metadata: Dict

@dataclass
class RetrievalContext:
    """Context for RAG pipeline"""
    query: str
    results: List[SearchResult]
    total_results: int
    combined_context: str

class VectorStore:
    """
    ChromaDB-based vector store for document chunks
    Handles storage, indexing, and similarity search
    """
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "pdf_documents"):
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()
    
    def _initialize_store(self):
        """
        Initialize ChromaDB client and collection
        """
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"✅ Loaded existing collection: {self.collection_name}")
                logger.info(f"   📊 Documents in collection: {self.collection.count()}")
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "PDF document chunks for RAG chatbot"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def add_documents(self, embedded_chunks):
        """
        Add embedded document chunks to the vector store
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks to add")
            return
        
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for embedded_chunk in embedded_chunks:
                chunk = embedded_chunk.chunk
                
                # Generate unique ID
                chunk_id = f"{chunk.chunk_id}_{uuid.uuid4().hex[:8]}"
                ids.append(chunk_id)
                
                # Add embedding
                embeddings.append(embedded_chunk.embedding.tolist())
                
                # Add document content
                documents.append(chunk.content)
                
                # Add metadata
                metadata = {
                    'source_file': chunk.source_file,
                    'page_number': chunk.page_number,
                    'chunk_id': chunk.chunk_id,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'chunk_length': chunk.metadata.get('chunk_length', 0),
                    'word_count': chunk.metadata.get('word_count', 0),
                    'language': chunk.metadata.get('language', 'unknown'),
                    'content_type': chunk.metadata.get('content_type', 'unknown'),
                    'chunk_type': chunk.metadata.get('chunk_type', 'unknown'),
                    'has_numbers': chunk.metadata.get('has_numbers', False),
                    'has_currency': chunk.metadata.get('has_currency', False),
                    'has_percentage': chunk.metadata.get('has_percentage', False),
                    'embedding_model': embedded_chunk.embedding_model
                }
                
                # Add table-specific metadata
                if 'table_id' in chunk.metadata:
                    metadata['table_id'] = chunk.metadata['table_id']
                if 'is_table_fragment' in chunk.metadata:
                    metadata['is_table_fragment'] = chunk.metadata['is_table_fragment']
                if 'row_index' in chunk.metadata:
                    metadata['row_index'] = chunk.metadata['row_index']
                
                metadatas.append(metadata)
            
            # Add to collection in batches
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
            
            logger.info(f"✅ Added {len(embedded_chunks)} chunks to vector store")
            logger.info(f"   📊 Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    def search(self, 
               query_embedding: np.ndarray,
               n_results: int = 10,  # Increased from 5
               min_score: float = 0.6,  # Restored to reasonable level
               filter_metadata: Optional[Dict] = None) -> List[SearchResult]:
        """
        Perform similarity search in the vector store
        """
        try:
            # Convert embedding to list if it's numpy array
            if isinstance(query_embedding, np.ndarray):
                query_embedding = query_embedding.tolist()
            
            # Perform search with increased results
            search_kwargs = {
                "query_embeddings": [query_embedding],
                "n_results": n_results * 2  # Get more results for filtering
            }
            
            # Add metadata filter if provided
            if filter_metadata:
                search_kwargs["where"] = filter_metadata
            
            results = self.collection.query(**search_kwargs)
            
            # Convert to SearchResult objects and filter by score
            search_results = []
            
            for i in range(len(results['ids'][0])):
                score = 1.0 - results['distances'][0][i]  # Convert distance to similarity
                
                # Apply minimum score threshold
                if score >= min_score:
                    result = SearchResult(
                        content=results['documents'][0][i],
                        chunk_id=results['metadatas'][0][i]['chunk_id'],
                        source_file=results['metadatas'][0][i]['source_file'],
                        page_number=results['metadatas'][0][i]['page_number'],
                        score=score,
                        metadata=results['metadatas'][0][i]
                    )
                    search_results.append(result)
            
            # Sort by score and limit to requested number of results
            search_results.sort(key=lambda x: x.score, reverse=True)
            search_results = search_results[:n_results]
            
            logger.info(f"Found {len(search_results)} results for query (min_score={min_score})")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the collection
        """
        try:
            count = self.collection.count()
            if count == 0:
                return {
                    'total_documents': 0,
                    'unique_source_files': 0,
                    'language_distribution': {},
                    'content_type_distribution': {},
                    'chunk_type_distribution': {},
                    'sample_size': 0,
                    'avg_chunk_length': 0
                }

            # Get sample of documents to analyze metadata
            limit_for_stats = min(count, 500)
            sample = self.collection.get(limit=limit_for_stats, include=["metadatas"])
            
            # Analyze metadata distributions
            languages = {}
            content_types = {}
            chunk_types = {}
            source_files = set()
            total_chars = 0
            actual_sample_size = 0

            if sample and sample['metadatas']:
                actual_sample_size = len(sample['metadatas'])
                for metadata_item in sample['metadatas']:
                    # Language distribution
                    lang = metadata_item.get('language', 'unknown')
                    languages[lang] = languages.get(lang, 0) + 1
                    
                    # Content type distribution
                    ctype = metadata_item.get('content_type', 'unknown')
                    content_types[ctype] = content_types.get(ctype, 0) + 1
                    
                    # Chunk type distribution
                    chunk_type = metadata_item.get('chunk_type', 'unknown')
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                    
                    # Source files
                    source_files.add(metadata_item.get('source_file', 'unknown'))
                    
                    # Character count
                    total_chars += metadata_item.get('chunk_length', 0)
            
            return {
                'total_documents': count,
                'unique_source_files': len(source_files),
                'language_distribution': languages,
                'content_type_distribution': content_types,
                'chunk_type_distribution': chunk_types,
                'sample_size_for_distributions': actual_sample_size,
                'avg_chunk_length_in_sample': total_chars / actual_sample_size if actual_sample_size > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}", exc_info=True)
            return {
                'total_documents': 0,
                'unique_source_files': 0,
                'language_distribution': {},
                'content_type_distribution': {},
                'chunk_type_distribution': {},
                'sample_size_for_distributions': 0,
                'avg_chunk_length_in_sample': 0,
                'error': str(e)
            }
    
    def clear_collection(self):
        """
        Clear all documents from the collection
        """
        try:
            # Delete the collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document chunks for RAG chatbot"}
            )
            
            logger.info(f"✅ Cleared collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

class RetrievalService:
    """
    High-level retrieval service combining embedding and vector search
    """
    
    def __init__(self, vector_store: VectorStore, embedding_service):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def retrieve_context(self, 
                        query: str,
                        n_results: int = 10,  # Increased from 5 to 10
                        min_score: float = 0.6,  # Restored to reasonable level
                        filter_by_language: Optional[str] = None,
                        filter_by_source: Optional[str] = None,
                        filter_by_content_type: Optional[str] = None) -> RetrievalContext:
        """
        Retrieve relevant context for a query
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Prepare metadata filter
            metadata_filter = {}
            if filter_by_language:
                metadata_filter['language'] = filter_by_language
            if filter_by_source:
                metadata_filter['source_file'] = filter_by_source
            if filter_by_content_type:
                metadata_filter['content_type'] = filter_by_content_type
            
            # Search for relevant chunks
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
                min_score=min_score,
                filter_metadata=metadata_filter if metadata_filter else None
            )
            
            # Combine contexts with some overlap handling
            combined_context = self._combine_contexts(search_results)
            
            return RetrievalContext(
                query=query,
                results=search_results,
                total_results=len(search_results),
                combined_context=combined_context
            )
            
        except Exception as e:
            logger.error(f"Context retrieval failed: {e}")
            return RetrievalContext(
                query=query,
                results=[],
                total_results=0,
                combined_context=""
            )
    
    def _combine_contexts(self, search_results: List[SearchResult]) -> str:
        """
        Intelligently combine search results into coherent context
        """
        if not search_results:
            return ""
        
        # Group by source file and page for better organization
        grouped_results = {}
        for result in search_results:
            key = f"{result.source_file}_page_{result.page_number}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        # Build combined context
        context_parts = []
        
        for file_page, results in grouped_results.items():
            # Sort by score (relevance)
            results.sort(key=lambda x: x.score, reverse=True)
            
            # Add source information
            source_info = f"\n**Kaynak: {results[0].source_file}, Sayfa {results[0].page_number}**\n"
            context_parts.append(source_info)
            
            # Add content
            for result in results:
                # Add metadata hints for better context
                metadata_hints = []
                if result.metadata.get('has_numbers'):
                    metadata_hints.append("📊")
                if result.metadata.get('has_currency'):
                    metadata_hints.append("💰")
                if result.metadata.get('has_percentage'):
                    metadata_hints.append("📈")
                
                hint_str = " ".join(metadata_hints)
                context_parts.append(f"{hint_str} {result.content}\n")
        
        return "\n".join(context_parts)
    
    def get_retrieval_stats(self) -> Dict:
        """
        Get retrieval service statistics
        """
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'vector_store_stats': vector_stats,
            'embedding_model': self.embedding_service.model_name,
            'service_status': 'active'
        }

def test_vector_store():
    """
    Test the vector store and retrieval service
    """
    from text_processor import TextProcessor
    from pdf_processor import PDFProcessor
    
    # Initialize components
    vector_store = VectorStore()
    text_processor = TextProcessor(chunk_size=300, overlap_size=50)
    pdf_processor = PDFProcessor()
    
    # Test with sample PDF
    test_pdf = "pdf/KAP - Pegasus Özel Finansal Bilgiler.pdf"
    
    try:
        logger.info("🧪 Testing Vector Store...")
        
        # Process PDF and create embeddings
        pdf_result = pdf_processor.process_pdf(test_pdf)
        embedded_chunks = text_processor.process_document_pages(
            pdf_result.pages[:2], test_pdf  # First 2 pages for testing
        )
        
        # Add to vector store
        vector_store.add_documents(embedded_chunks)
        
        # Initialize retrieval service
        class EmbeddingServiceWrapper:
            def __init__(self, model):
                self.model_name = model.model_name if hasattr(model, 'model_name') else str(model)
                self.model = model
            
            def generate_embedding(self, text: str):
                return self.model.encode(text)
        
        embedding_wrapper = EmbeddingServiceWrapper(text_processor.model)
        retrieval_service = RetrievalService(
            vector_store=vector_store,
            embedding_service=embedding_wrapper
        )
        
        # Test retrieval
        test_queries = [
            "finansal performans",
            "gelir tablosu",
            "Pegasus hakkında bilgi ver"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            
            context = retrieval_service.retrieve_context(
                query=query,
                n_results=3,
                min_score=0.7
            )
            
            print(f"   📊 Results found: {context.total_results}")
            
            for i, result in enumerate(context.results, 1):
                print(f"   {i}. Score: {result.score:.3f} | Page: {result.page_number}")
                print(f"      Preview: {result.content[:100]}...")
        
        # Get statistics
        stats = retrieval_service.get_retrieval_stats()
        print(f"\n📊 Collection Statistics:")
        print(f"   📄 Total documents: {stats['vector_store_stats']['total_documents']}")
        print(f"   📁 Source files: {stats['vector_store_stats']['unique_source_files']}")
        print(f"   🌍 Languages: {stats['vector_store_stats']['language_distribution']}")
        print(f"   📑 Content types: {stats['vector_store_stats']['content_type_distribution']}")
        print(f"   📋 Chunk types: {stats['vector_store_stats']['chunk_type_distribution']}")
        
    except Exception as e:
        logger.error(f"Vector store test failed: {e}")

if __name__ == "__main__":
    test_vector_store() 