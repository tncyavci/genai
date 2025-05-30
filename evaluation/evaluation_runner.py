"""
Evaluation Runner Module
Executes test cases and generates evaluation reports
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime
import pandas as pd
from .evaluation_metrics import EvaluationMetrics
from .evaluation_dataset import EvaluationDataset, TestCase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationRunner:
    """Class for running evaluations and generating reports"""
    
    def __init__(self, 
                 output_dir: str = "evaluation/results",
                 metrics: EvaluationMetrics = None):
        """
        Initialize evaluation runner
        
        Args:
            output_dir: Directory to save evaluation results
            metrics: EvaluationMetrics instance (optional)
        """
        self.output_dir = output_dir
        self.metrics = metrics or EvaluationMetrics()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"✅ Evaluation runner initialized with output directory: {output_dir}")
    
    def run_evaluation(self,
                      test_cases: List[TestCase],
                      model_responses: Dict[str, str],
                      evaluation_name: str = None) -> Dict[str, Any]:
        """
        Run evaluation on test cases
        
        Args:
            test_cases: List of test cases to evaluate
            model_responses: Dictionary mapping query to model response
            evaluation_name: Name for this evaluation run (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            if not evaluation_name:
                evaluation_name = f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            results = {
                'evaluation_name': evaluation_name,
                'timestamp': datetime.now().isoformat(),
                'total_cases': len(test_cases),
                'case_results': [],
                'summary': {}
            }
            
            # Evaluate each test case
            for case in test_cases:
                if case.query not in model_responses:
                    logger.warning(f"No model response found for query: {case.query}")
                    continue
                
                # Get model response
                model_response = model_responses[case.query]
                
                # Calculate metrics
                case_result = self.metrics.evaluate_response(
                    reference=case.reference_answer,
                    hypothesis=model_response,
                    key_metrics=case.metadata.get('metrics', None)
                )
                
                # Add case metadata
                case_result['query'] = case.query
                case_result['reference_answer'] = case.reference_answer
                case_result['model_response'] = model_response
                case_result['metadata'] = case.metadata
                
                results['case_results'].append(case_result)
            
            # Calculate summary statistics
            if results['case_results']:
                summary = {
                    'average_rouge1': sum(r['rouge_scores']['rouge1'] for r in results['case_results']) / len(results['case_results']),
                    'average_rouge2': sum(r['rouge_scores']['rouge2'] for r in results['case_results']) / len(results['case_results']),
                    'average_rougeL': sum(r['rouge_scores']['rougeL'] for r in results['case_results']) / len(results['case_results']),
                    'average_bleu': sum(r['bleu_score'] for r in results['case_results']) / len(results['case_results']),
                    'average_semantic_similarity': sum(r['semantic_similarity'] for r in results['case_results']) / len(results['case_results']),
                    'average_overall_score': sum(r['overall_score'] for r in results['case_results']) / len(results['case_results'])
                }
                
                # Calculate financial accuracy summary
                financial_metrics = {}
                for case_result in results['case_results']:
                    for metric, score in case_result['financial_accuracy'].items():
                        if metric not in financial_metrics:
                            financial_metrics[metric] = []
                        financial_metrics[metric].append(score)
                
                for metric, scores in financial_metrics.items():
                    summary[f'average_{metric}_accuracy'] = sum(scores) / len(scores)
                
                results['summary'] = summary
            
            # Save results
            self._save_results(results, evaluation_name)
            
            logger.info(f"✅ Completed evaluation: {evaluation_name}")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                'evaluation_name': evaluation_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _save_results(self, results: Dict[str, Any], evaluation_name: str):
        """
        Save evaluation results to files
        
        Args:
            results: Evaluation results dictionary
            evaluation_name: Name of evaluation run
        """
        try:
            # Save detailed results as JSON
            json_path = os.path.join(self.output_dir, f"{evaluation_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            # Save summary as Excel
            excel_path = os.path.join(self.output_dir, f"{evaluation_name}_summary.xlsx")
            
            # Create summary DataFrame
            summary_data = []
            for case_result in results['case_results']:
                row = {
                    'Query': case_result['query'],
                    'Reference Answer': case_result['reference_answer'],
                    'Model Response': case_result['model_response'],
                    'ROUGE-1': case_result['rouge_scores']['rouge1'],
                    'ROUGE-2': case_result['rouge_scores']['rouge2'],
                    'ROUGE-L': case_result['rouge_scores']['rougeL'],
                    'BLEU': case_result['bleu_score'],
                    'Semantic Similarity': case_result['semantic_similarity'],
                    'Overall Score': case_result['overall_score']
                }
                
                # Add financial accuracy metrics
                for metric, score in case_result['financial_accuracy'].items():
                    row[f'{metric} Accuracy'] = score
                
                summary_data.append(row)
            
            df = pd.DataFrame(summary_data)
            
            # Add summary statistics
            summary_df = pd.DataFrame([results['summary']])
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(excel_path) as writer:
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
            
            logger.info(f"✅ Saved evaluation results to {json_path} and {excel_path}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")
    
    def generate_report(self, evaluation_name: str) -> str:
        """
        Generate HTML report from evaluation results
        
        Args:
            evaluation_name: Name of evaluation run
            
        Returns:
            Path to generated HTML report
        """
        try:
            # Load results
            json_path = os.path.join(self.output_dir, f"{evaluation_name}.json")
            with open(json_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # Generate HTML report
            html_path = os.path.join(self.output_dir, f"{evaluation_name}_report.html")
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Evaluation Report - {evaluation_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f5f5f5; }}
                    .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                    .case-result {{ margin: 20px 0; padding: 10px; border: 1px solid #ddd; }}
                </style>
            </head>
            <body>
                <h1>Evaluation Report</h1>
                <p>Evaluation Name: {evaluation_name}</p>
                <p>Timestamp: {results['timestamp']}</p>
                <p>Total Test Cases: {results['total_cases']}</p>
                
                <h2>Summary Statistics</h2>
                <div class="summary">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Score</th>
                        </tr>
            """
            
            # Add summary statistics
            for metric, score in results['summary'].items():
                html_content += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()}</td>
                            <td>{score:.4f}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <h2>Detailed Results</h2>
            """
            
            # Add detailed results
            for case_result in results['case_results']:
                html_content += f"""
                <div class="case-result">
                    <h3>Query: {case_result['query']}</h3>
                    <p><strong>Reference Answer:</strong> {case_result['reference_answer']}</p>
                    <p><strong>Model Response:</strong> {case_result['model_response']}</p>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Score</th>
                        </tr>
                """
                
                # Add ROUGE scores
                for rouge_type, score in case_result['rouge_scores'].items():
                    html_content += f"""
                        <tr>
                            <td>{rouge_type.upper()}</td>
                            <td>{score:.4f}</td>
                        </tr>
                    """
                
                # Add other metrics
                html_content += f"""
                        <tr>
                            <td>BLEU</td>
                            <td>{case_result['bleu_score']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Semantic Similarity</td>
                            <td>{case_result['semantic_similarity']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Overall Score</td>
                            <td>{case_result['overall_score']:.4f}</td>
                        </tr>
                """
                
                # Add financial accuracy metrics
                for metric, score in case_result['financial_accuracy'].items():
                    html_content += f"""
                        <tr>
                            <td>{metric.replace('_', ' ').title()} Accuracy</td>
                            <td>{score:.4f}</td>
                        </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save HTML report
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"✅ Generated evaluation report: {html_path}")
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to generate evaluation report: {e}")
            return None

def main():
    """Main function for testing evaluation runner"""
    # Initialize components
    dataset = EvaluationDataset()
    runner = EvaluationRunner()
    
    # Create sample test cases
    test_cases = dataset.create_financial_test_cases()
    
    # Sample model responses
    model_responses = {
        "Şirketin kayıtlı sermayesi nedir?": "Şirketin kayıtlı sermayesi 133.096 TL'dir.",
        "Son dönem net kar ne kadar?": "Son dönem net kar 15.2 milyon TL olarak raporlanmıştır.",
        "Toplam varlıkların dağılımı nasıl?": "Toplam varlıkların %60'ı dönen varlıklar, %40'ı duran varlıklardan oluşmaktadır."
    }
    
    # Run evaluation
    results = runner.run_evaluation(
        test_cases=test_cases,
        model_responses=model_responses,
        evaluation_name="sample_evaluation"
    )
    
    # Generate report
    report_path = runner.generate_report("sample_evaluation")
    
    if report_path:
        print(f"✅ Evaluation report generated: {report_path}")

if __name__ == "__main__":
    main() 