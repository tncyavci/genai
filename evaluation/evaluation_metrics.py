"""
Evaluation Metrics Module
Contains various metrics for evaluating RAG system performance
"""

import numpy as np
from typing import List, Dict, Tuple
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Class for calculating various evaluation metrics"""
    
    def __init__(self):
        """Initialize evaluation metrics"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize ROUGE scorer
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            logger.info("✅ Evaluation metrics initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize evaluation metrics: {e}")
            raise
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and hypothesis texts
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            
        Returns:
            Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            
            return {
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            }
        except Exception as e:
            logger.error(f"ROUGE score calculation failed: {e}")
            return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> float:
        """
        Calculate BLEU score between reference and hypothesis texts
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            
        Returns:
            BLEU score
        """
        try:
            # Tokenize texts
            reference_tokens = nltk.word_tokenize(reference.lower())
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            
            # Calculate BLEU score
            score = sentence_bleu([reference_tokens], hypothesis_tokens)
            
            return score
        except Exception as e:
            logger.error(f"BLEU score calculation failed: {e}")
            return 0.0
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """
        Calculate semantic similarity between reference and hypothesis texts
        using cosine similarity of sentence embeddings
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            
        Returns:
            Semantic similarity score (0-1)
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            
            # Load model
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # Encode texts
            reference_embedding = model.encode(reference, convert_to_tensor=True)
            hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)
            
            # Calculate cosine similarity
            similarity = util.pytorch_cos_sim(reference_embedding, hypothesis_embedding)
            
            return float(similarity[0][0])
        except Exception as e:
            logger.error(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    def calculate_financial_accuracy(self, 
                                  reference: str, 
                                  hypothesis: str,
                                  key_metrics: List[str] = None) -> Dict[str, float]:
        """
        Calculate accuracy of financial metrics in generated text
        
        Args:
            reference: Reference text containing correct financial metrics
            hypothesis: Generated text to evaluate
            key_metrics: List of financial metrics to check (e.g., ['revenue', 'profit'])
            
        Returns:
            Dictionary containing accuracy scores for each metric
        """
        try:
            if key_metrics is None:
                key_metrics = [
                    'revenue', 'profit', 'assets', 'liabilities',
                    'equity', 'cash', 'debt', 'income'
                ]
            
            accuracy_scores = {}
            
            for metric in key_metrics:
                # Extract numbers associated with metric from both texts
                ref_numbers = self._extract_metric_numbers(reference, metric)
                hyp_numbers = self._extract_metric_numbers(hypothesis, metric)
                
                if not ref_numbers or not hyp_numbers:
                    accuracy_scores[metric] = 0.0
                    continue
                
                # Calculate accuracy
                correct = sum(1 for r, h in zip(ref_numbers, hyp_numbers) if abs(r - h) < 0.01)
                accuracy_scores[metric] = correct / len(ref_numbers)
            
            return accuracy_scores
        except Exception as e:
            logger.error(f"Financial accuracy calculation failed: {e}")
            return {metric: 0.0 for metric in key_metrics}
    
    def _extract_metric_numbers(self, text: str, metric: str) -> List[float]:
        """
        Extract numbers associated with a specific metric from text
        
        Args:
            text: Text to extract numbers from
            metric: Metric name to look for
            
        Returns:
            List of numbers associated with the metric
        """
        try:
            import re
            
            # Find all numbers near the metric
            pattern = f"{metric}.*?(\d+(?:,\d+)*(?:\.\d+)?)"
            matches = re.finditer(pattern, text.lower())
            
            numbers = []
            for match in matches:
                # Convert string number to float
                num_str = match.group(1).replace(',', '')
                try:
                    numbers.append(float(num_str))
                except ValueError:
                    continue
            
            return numbers
        except Exception as e:
            logger.error(f"Number extraction failed: {e}")
            return []
    
    def evaluate_response(self,
                         reference: str,
                         hypothesis: str,
                         key_metrics: List[str] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of generated response
        
        Args:
            reference: Reference text
            hypothesis: Generated text
            key_metrics: List of financial metrics to check
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        try:
            # Calculate all metrics
            rouge_scores = self.calculate_rouge_scores(reference, hypothesis)
            bleu_score = self.calculate_bleu_score(reference, hypothesis)
            semantic_similarity = self.calculate_semantic_similarity(reference, hypothesis)
            financial_accuracy = self.calculate_financial_accuracy(reference, hypothesis, key_metrics)
            
            # Combine all metrics
            evaluation_results = {
                'rouge_scores': rouge_scores,
                'bleu_score': bleu_score,
                'semantic_similarity': semantic_similarity,
                'financial_accuracy': financial_accuracy,
                'overall_score': np.mean([
                    rouge_scores['rougeL'],
                    bleu_score,
                    semantic_similarity,
                    np.mean(list(financial_accuracy.values()))
                ])
            }
            
            return evaluation_results
        except Exception as e:
            logger.error(f"Response evaluation failed: {e}")
            return {
                'rouge_scores': {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0},
                'bleu_score': 0.0,
                'semantic_similarity': 0.0,
                'financial_accuracy': {},
                'overall_score': 0.0
            } 