"""
Evaluation Dataset Module
Manages test cases and ground truth data for RAG system evaluation
"""

import json
import os
from typing import List, Dict, Any
import logging
from dataclasses import dataclass
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Container for test case data"""
    query: str
    reference_answer: str
    context: str
    metadata: Dict[str, Any]

class EvaluationDataset:
    """Class for managing evaluation datasets"""
    
    def __init__(self, dataset_path: str = "evaluation/datasets"):
        """
        Initialize evaluation dataset
        
        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = dataset_path
        self.test_cases: List[TestCase] = []
        
        # Create dataset directory if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)
        
        logger.info(f"✅ Evaluation dataset initialized at {dataset_path}")
    
    def load_test_cases(self, dataset_name: str) -> List[TestCase]:
        """
        Load test cases from JSON file
        
        Args:
            dataset_name: Name of dataset file (without extension)
            
        Returns:
            List of TestCase objects
        """
        try:
            file_path = os.path.join(self.dataset_path, f"{dataset_name}.json")
            
            if not os.path.exists(file_path):
                logger.warning(f"Dataset file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            test_cases = []
            for case in data['test_cases']:
                test_case = TestCase(
                    query=case['query'],
                    reference_answer=case['reference_answer'],
                    context=case.get('context', ''),
                    metadata=case.get('metadata', {})
                )
                test_cases.append(test_case)
            
            self.test_cases = test_cases
            logger.info(f"✅ Loaded {len(test_cases)} test cases from {dataset_name}")
            
            return test_cases
        except Exception as e:
            logger.error(f"Failed to load test cases: {e}")
            return []
    
    def save_test_cases(self, dataset_name: str, test_cases: List[TestCase]) -> bool:
        """
        Save test cases to JSON file
        
        Args:
            dataset_name: Name of dataset file (without extension)
            test_cases: List of TestCase objects to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            file_path = os.path.join(self.dataset_path, f"{dataset_name}.json")
            
            data = {
                'dataset_name': dataset_name,
                'test_cases': [
                    {
                        'query': case.query,
                        'reference_answer': case.reference_answer,
                        'context': case.context,
                        'metadata': case.metadata
                    }
                    for case in test_cases
                ]
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Saved {len(test_cases)} test cases to {dataset_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save test cases: {e}")
            return False
    
    def create_financial_test_cases(self) -> List[TestCase]:
        """
        Create sample financial test cases
        
        Returns:
            List of TestCase objects
        """
        test_cases = [
            TestCase(
                query="Şirketin kayıtlı sermayesi nedir?",
                reference_answer="Şirketin kayıtlı sermayesi 133.096 TL'dir.",
                context="Finansal tablolarda kayıtlı sermaye 133.096 TL olarak görünmektedir.",
                metadata={
                    'category': 'financial',
                    'metric': 'registered_capital',
                    'difficulty': 'easy'
                }
            ),
            TestCase(
                query="Son dönem net kar ne kadar?",
                reference_answer="Son dönem net kar 15.2 milyon TL'dir.",
                context="Gelir tablosunda son dönem net kar 15.2 milyon TL olarak raporlanmıştır.",
                metadata={
                    'category': 'financial',
                    'metric': 'net_profit',
                    'difficulty': 'medium'
                }
            ),
            TestCase(
                query="Toplam varlıkların dağılımı nasıl?",
                reference_answer="Toplam varlıkların %60'ı dönen varlıklar, %40'ı duran varlıklardan oluşmaktadır.",
                context="Bilanço analizinde dönen varlıklar 60 milyon TL, duran varlıklar 40 milyon TL olarak görünmektedir.",
                metadata={
                    'category': 'financial',
                    'metric': 'asset_distribution',
                    'difficulty': 'hard'
                }
            )
        ]
        
        return test_cases
    
    def export_to_excel(self, dataset_name: str, output_path: str = None) -> bool:
        """
        Export test cases to Excel file
        
        Args:
            dataset_name: Name of dataset to export
            output_path: Path to save Excel file (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not output_path:
                output_path = os.path.join(self.dataset_path, f"{dataset_name}.xlsx")
            
            # Convert test cases to DataFrame
            data = []
            for case in self.test_cases:
                data.append({
                    'Query': case.query,
                    'Reference Answer': case.reference_answer,
                    'Context': case.context,
                    'Category': case.metadata.get('category', ''),
                    'Metric': case.metadata.get('metric', ''),
                    'Difficulty': case.metadata.get('difficulty', '')
                })
            
            df = pd.DataFrame(data)
            
            # Save to Excel
            df.to_excel(output_path, index=False)
            
            logger.info(f"✅ Exported test cases to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export test cases: {e}")
            return False
    
    def import_from_excel(self, excel_path: str) -> List[TestCase]:
        """
        Import test cases from Excel file
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            List of TestCase objects
        """
        try:
            # Read Excel file
            df = pd.read_excel(excel_path)
            
            # Convert to test cases
            test_cases = []
            for _, row in df.iterrows():
                test_case = TestCase(
                    query=row['Query'],
                    reference_answer=row['Reference Answer'],
                    context=row['Context'],
                    metadata={
                        'category': row['Category'],
                        'metric': row['Metric'],
                        'difficulty': row['Difficulty']
                    }
                )
                test_cases.append(test_case)
            
            self.test_cases = test_cases
            logger.info(f"✅ Imported {len(test_cases)} test cases from {excel_path}")
            
            return test_cases
        except Exception as e:
            logger.error(f"Failed to import test cases: {e}")
            return [] 