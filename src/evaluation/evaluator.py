from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
import json
from pathlib import Path

class ModelEvaluator:
    def __init__(self, output_dir: str):
        """
        Initialize the model evaluator.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_classification(self, y_true: List[str], y_pred: List[str],
                              labels: List[str]) -> Dict[str, Any]:
        """
        Evaluate document classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: List of label names
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average='weighted'
        )
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred, labels=labels, output_dict=True
        )
        
        # Compile results
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': report
        }
        
        # Save results
        self._save_results('classification_metrics.json', results)
        
        return results
    
    def evaluate_extraction(self, true_fields: List[Dict], 
                          pred_fields: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate information extraction performance.
        
        Args:
            true_fields: List of dictionaries containing true field values
            pred_fields: List of dictionaries containing predicted field values
            
        Returns:
            Dictionary containing evaluation metrics
        """
        field_metrics = {}
        
        # Get all field names
        all_fields = set()
        for fields in true_fields + pred_fields:
            all_fields.update(fields.keys())
        
        # Calculate metrics for each field
        for field in all_fields:
            correct = 0
            total_true = 0
            total_pred = 0
            
            for true, pred in zip(true_fields, pred_fields):
                if field in true:
                    total_true += 1
                    if field in pred and true[field] == pred[field]:
                        correct += 1
                
                if field in pred:
                    total_pred += 1
            
            # Calculate precision, recall, F1
            precision = correct / total_pred if total_pred > 0 else 0
            recall = correct / total_true if total_true > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) \
                if precision + recall > 0 else 0
            
            field_metrics[field] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': total_true
            }
        
        # Calculate macro averages
        macro_metrics = {
            metric: np.mean([
                field['metric']
                for field in field_metrics.values()
            ])
            for metric in ['precision', 'recall', 'f1_score']
        }
        
        results = {
            'field_metrics': field_metrics,
            'macro_avg': macro_metrics
        }
        
        # Save results
        self._save_results('extraction_metrics.json', results)
        
        return results
    
    def _save_results(self, filename: str, results: Dict):
        """Save evaluation results to JSON file."""
        output_file = self.output_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2) 