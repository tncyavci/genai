#!/usr/bin/env python3
"""
Excel Processor Module
Handles text extraction and data processing from Excel files (XLS/XLSX)
Supports multiple sheets and data analysis
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExcelSheetContent:
    """Container for Excel sheet content"""
    sheet_name: str
    data: pd.DataFrame
    text_content: str
    metadata: Dict
    
@dataclass 
class ExcelProcessingResult:
    """Complete processing result for an Excel file"""
    sheets: List[ExcelSheetContent]
    total_sheets: int
    file_path: str
    processing_method: str

class ExcelProcessor:
    """
    Advanced Excel processor supporting XLS and XLSX files
    Converts data to searchable text format for RAG pipeline
    """
    
    def __init__(self):
        self.supported_extensions = ['.xls', '.xlsx', '.xlsm']
        
    def read_excel_file(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """
        Read all sheets from Excel file
        """
        try:
            # Read all sheets
            excel_data = pd.read_excel(
                file_path, 
                sheet_name=None,  # Read all sheets
                engine='openpyxl' if file_path.endswith('.xlsx') else 'xlrd'
            )
            
            logger.info(f"Read {len(excel_data)} sheets from {os.path.basename(file_path)}")
            return excel_data
            
        except Exception as e:
            logger.error(f"Failed to read Excel file {file_path}: {e}")
            raise
    
    def dataframe_to_text(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        Convert DataFrame to searchable text format
        """
        if df.empty:
            return f"Sheet '{sheet_name}' is empty."
        
        text_parts = []
        
        # Add sheet header
        text_parts.append(f"=== SAYFA: {sheet_name} ===\n")
        
        # Add basic info
        rows, cols = df.shape
        text_parts.append(f"Boyut: {rows} satır x {cols} sütun\n")
        
        # Add column headers
        if not df.columns.empty:
            headers = [str(col) for col in df.columns]
            text_parts.append(f"Sütunlar: {', '.join(headers)}\n")
        
        # Process data row by row
        for idx, row in df.iterrows():
            row_texts = []
            for col_name, value in row.items():
                if pd.notna(value) and str(value).strip():
                    # Format value based on type
                    if isinstance(value, (int, float)):
                        if isinstance(value, float) and value.is_integer():
                            formatted_value = f"{int(value):,}"
                        else:
                            formatted_value = f"{value:,.2f}"
                    else:
                        formatted_value = str(value).strip()
                    
                    row_texts.append(f"{col_name}: {formatted_value}")
            
            if row_texts:
                text_parts.append(f"Satır {idx + 1}: {' | '.join(row_texts)}")
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            text_parts.append("\n=== SAYISAL ÖZET ===")
            for col in numeric_cols:
                if not df[col].empty and df[col].notna().any():
                    stats = df[col].describe()
                    text_parts.append(
                        f"{col}: Toplam={df[col].sum():,.2f}, "
                        f"Ortalama={stats['mean']:,.2f}, "
                        f"Min={stats['min']:,.2f}, "
                        f"Max={stats['max']:,.2f}"
                    )
        
        return "\n".join(text_parts)
    
    def extract_metadata(self, df: pd.DataFrame, sheet_name: str) -> Dict:
        """
        Extract metadata from DataFrame
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        
        metadata = {
            'sheet_name': sheet_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'text_columns': len(text_cols),
            'empty_cells': df.isnull().sum().sum(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        # Add column names
        metadata['column_names'] = df.columns.tolist()
        
        # Add data ranges for numeric columns
        if len(numeric_cols) > 0:
            metadata['numeric_ranges'] = {}
            for col in numeric_cols:
                if df[col].notna().any():
                    metadata['numeric_ranges'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean())
                    }
        
        return metadata
    
    def process_excel(self, file_path: str) -> ExcelProcessingResult:
        """
        Main processing method for Excel files
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Excel file not found: {file_path}")
            
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_ext}")
        
        logger.info(f"Starting Excel processing: {os.path.basename(file_path)}")
        
        try:
            # Read Excel file
            excel_data = self.read_excel_file(file_path)
            
            sheets = []
            for sheet_name, df in excel_data.items():
                # Clean DataFrame
                df_cleaned = self.clean_dataframe(df)
                
                # Convert to text
                text_content = self.dataframe_to_text(df_cleaned, sheet_name)
                
                # Extract metadata
                metadata = self.extract_metadata(df_cleaned, sheet_name)
                
                sheet_content = ExcelSheetContent(
                    sheet_name=sheet_name,
                    data=df_cleaned,
                    text_content=text_content,
                    metadata=metadata
                )
                sheets.append(sheet_content)
            
            return ExcelProcessingResult(
                sheets=sheets,
                total_sheets=len(sheets),
                file_path=file_path,
                processing_method='pandas'
            )
            
        except Exception as e:
            logger.error(f"Excel processing failed for {file_path}: {e}")
            raise
    
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare DataFrame
        """
        # Remove completely empty rows and columns
        df_cleaned = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Reset index
        df_cleaned = df_cleaned.reset_index(drop=True)
        
        # Clean column names
        df_cleaned.columns = [
            str(col).strip() if pd.notna(col) else f"Column_{i}" 
            for i, col in enumerate(df_cleaned.columns)
        ]
        
        return df_cleaned
    
    def get_excel_summary(self, result: ExcelProcessingResult) -> Dict:
        """
        Generate summary statistics for Excel content
        """
        total_rows = sum(len(sheet.data) for sheet in result.sheets)
        total_columns = sum(len(sheet.data.columns) for sheet in result.sheets)
        total_numeric_cols = sum(sheet.metadata.get('numeric_columns', 0) for sheet in result.sheets)
        total_text_length = sum(len(sheet.text_content) for sheet in result.sheets)
        
        return {
            'file_name': os.path.basename(result.file_path),
            'total_sheets': result.total_sheets,
            'total_rows': total_rows,
            'total_columns': total_columns,
            'total_numeric_columns': total_numeric_cols,
            'total_text_length': total_text_length,
            'processing_method': result.processing_method,
            'sheet_names': [sheet.sheet_name for sheet in result.sheets],
            'avg_rows_per_sheet': total_rows / result.total_sheets if result.total_sheets > 0 else 0
        }

def test_excel_processor():
    """
    Test function to validate Excel processing
    """
    processor = ExcelProcessor()
    
    # Test with sample Excel files
    excel_directory = "excel"
    if not os.path.exists(excel_directory):
        print("❌ Excel directory not found!")
        return
    
    excel_files = [f for f in os.listdir(excel_directory) 
                   if any(f.endswith(ext) for ext in processor.supported_extensions)]
    
    if not excel_files:
        print("❌ No Excel files found!")
        return
    
    for excel_file in excel_files[:2]:  # Test first 2 files
        file_path = os.path.join(excel_directory, excel_file)
        print(f"\n🧪 Testing: {excel_file}")
        
        try:
            result = processor.process_excel(file_path)
            summary = processor.get_excel_summary(result)
            
            print(f"✅ Processed successfully!")
            print(f"   📊 Sheets: {summary['total_sheets']}")
            print(f"   📝 Rows: {summary['total_rows']}")
            print(f"   📋 Columns: {summary['total_columns']}")
            print(f"   🔢 Numeric columns: {summary['total_numeric_columns']}")
            print(f"   📄 Text length: {summary['total_text_length']:,} chars")
            
            # Show first sheet content preview
            if result.sheets:
                first_sheet = result.sheets[0]
                preview = first_sheet.text_content[:200] + "..." if len(first_sheet.text_content) > 200 else first_sheet.text_content
                print(f"   📋 Preview: {preview}")
                
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_excel_processor() 