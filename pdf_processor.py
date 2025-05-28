#!/usr/bin/env python3
"""
PDF Processor Module
Handles text extraction and table detection from PDF files
Supports multiple extraction methods for robust parsing
"""

import os
import pandas as pd
import pdfplumber
import PyPDF2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Container for extracted PDF content"""
    text: str
    tables: List[pd.DataFrame]
    metadata: Dict
    page_number: int
    
@dataclass 
class ProcessingResult:
    """Complete processing result for a PDF"""
    pages: List[ExtractedContent]
    total_pages: int
    file_path: str
    processing_method: str

class PDFProcessor:
    """
    Advanced PDF processor supporting multiple extraction methods
    Handles both text extraction and table detection
    """
    
    def __init__(self):
        self.supported_methods = ['pdfplumber', 'pypdf2', 'hybrid']
        
    def extract_with_pdfplumber(self, pdf_path: str) -> ProcessingResult:
        """
        Extract content using pdfplumber (recommended for tables)
        """
        pages = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages with pdfplumber...")
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    text = page.extract_text() or ""
                    
                    # Extract tables
                    tables = []
                    try:
                        extracted_tables = page.extract_tables()
                        for table_data in extracted_tables:
                            if table_data and len(table_data) > 1:  # Skip empty or single-row tables
                                df = pd.DataFrame(table_data[1:], columns=table_data[0])
                                df = df.dropna(how='all').dropna(axis=1, how='all')  # Clean empty rows/cols
                                if not df.empty:
                                    tables.append(df)
                    except Exception as e:
                        logger.warning(f"Table extraction failed on page {page_num}: {e}")
                    
                    # Create metadata
                    metadata = {
                        'page_width': page.width,
                        'page_height': page.height,
                        'char_count': len(text),
                        'table_count': len(tables)
                    }
                    
                    content = ExtractedContent(
                        text=text,
                        tables=tables,
                        metadata=metadata,
                        page_number=page_num
                    )
                    pages.append(content)
                    
        except Exception as e:
            logger.error(f"PDFPlumber extraction failed: {e}")
            raise
            
        return ProcessingResult(
            pages=pages,
            total_pages=total_pages,
            file_path=pdf_path,
            processing_method='pdfplumber'
        )
    
    def extract_with_pypdf2(self, pdf_path: str) -> ProcessingResult:
        """
        Extract content using PyPDF2 (fallback for text-only)
        """
        pages = []
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                logger.info(f"Processing {total_pages} pages with PyPDF2...")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text() or ""
                    
                    metadata = {
                        'char_count': len(text),
                        'table_count': 0  # PyPDF2 doesn't extract tables
                    }
                    
                    content = ExtractedContent(
                        text=text,
                        tables=[],  # No table extraction with PyPDF2
                        metadata=metadata,
                        page_number=page_num
                    )
                    pages.append(content)
                    
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
            
        return ProcessingResult(
            pages=pages,
            total_pages=total_pages,
            file_path=pdf_path,
            processing_method='pypdf2'
        )
    
    def process_pdf(self, pdf_path: str, method: str = 'pdfplumber') -> ProcessingResult:
        """
        Main processing method with fallback strategy
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        if method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {method}. Use one of {self.supported_methods}")
        
        logger.info(f"Starting PDF processing: {os.path.basename(pdf_path)}")
        
        try:
            if method == 'pdfplumber':
                return self.extract_with_pdfplumber(pdf_path)
            elif method == 'pypdf2':
                return self.extract_with_pypdf2(pdf_path)
            elif method == 'hybrid':
                # Try pdfplumber first, fallback to PyPDF2
                try:
                    return self.extract_with_pdfplumber(pdf_path)
                except Exception as e:
                    logger.warning(f"PDFPlumber failed, falling back to PyPDF2: {e}")
                    return self.extract_with_pypdf2(pdf_path)
        except Exception as e:
            logger.error(f"All extraction methods failed for {pdf_path}: {e}")
            raise
    
    def get_text_summary(self, result: ProcessingResult) -> Dict:
        """
        Generate summary statistics for extracted content
        """
        total_text_length = sum(len(page.text) for page in result.pages)
        total_tables = sum(len(page.tables) for page in result.pages)
        pages_with_tables = sum(1 for page in result.pages if page.tables)
        
        return {
            'file_name': os.path.basename(result.file_path),
            'total_pages': result.total_pages,
            'total_text_length': total_text_length,
            'total_tables': total_tables,
            'pages_with_tables': pages_with_tables,
            'processing_method': result.processing_method,
            'avg_text_per_page': total_text_length / result.total_pages if result.total_pages > 0 else 0
        }

def test_pdf_processor():
    """
    Test function to validate PDF processing
    """
    processor = PDFProcessor()
    
    # Test with sample PDF files
    pdf_directory = "pdf"
    if not os.path.exists(pdf_directory):
        print("❌ PDF directory not found!")
        return
    
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDF files found!")
        return
    
    print(f"🔍 Found {len(pdf_files)} PDF files for testing")
    
    for pdf_file in pdf_files[:2]:  # Test first 2 files
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"\n📄 Testing: {pdf_file}")
        
        try:
            # Process with pdfplumber
            result = processor.process_pdf(pdf_path, method='pdfplumber')
            summary = processor.get_text_summary(result)
            
            print(f"✅ Processing successful!")
            print(f"   📊 Pages: {summary['total_pages']}")
            print(f"   📝 Text length: {summary['total_text_length']:,} chars")
            print(f"   📋 Tables found: {summary['total_tables']}")
            print(f"   🗂️  Pages with tables: {summary['pages_with_tables']}")
            
            # Show sample text from first page
            if result.pages and result.pages[0].text:
                sample_text = result.pages[0].text[:200].replace('\n', ' ')
                print(f"   📖 Sample text: {sample_text}...")
            
            # Show table info if any
            if result.pages[0].tables:
                table = result.pages[0].tables[0]
                print(f"   📊 First table shape: {table.shape}")
                print(f"   📊 Table columns: {list(table.columns)[:3]}...")
                
        except Exception as e:
            print(f"❌ Processing failed: {e}")

if __name__ == "__main__":
    test_pdf_processor() 