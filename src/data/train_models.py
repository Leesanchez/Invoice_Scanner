import sys
from pathlib import Path
import logging
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from classification.document_classifier import (
    SVMDocumentClassifier,
    RandomForestDocumentClassifier,
    GradientBoostingDocumentClassifier,
    LSTMDocumentClassifier,
    TransformerDocumentClassifier
)
from preprocessing.preprocessor import DocumentPreprocessor

class ModelTrainer:
    def __init__(self, data_dir: Path):
        """
        Initialize the model trainer.
        
        Args:
            data_dir: Path to the data directory containing train/test splits
        """
        self.data_dir = Path(data_dir)
        self.setup_logging()
        self.load_split_info()
        self.preprocessor = DocumentPreprocessor()
        
    def setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.data_dir / 'evaluation' / 'training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_split_info(self):
        """Load the dataset split information."""
        split_info_path = self.data_dir / 'evaluation' / 'metrics' / 'split_info.json'
        with open(split_info_path) as f:
            self.split_info = json.load(f)
        self.categories = ['Invoices', 'PurchaseOrders', 'Shipping orders', 'Other']
        
    def load_data(self, split: str) -> Dict[str, List]:
        """Load and preprocess data for a given split (train/test)."""
        texts = []
        labels = []
        categories = []
        
        for category_idx, category in enumerate(self.categories):
            category_dir = self.data_dir / 'processed' / split / category
            if not category_dir.exists():
                self.logger.warning(f"Directory {category_dir} does not exist")
                continue
            
            for file_path in category_dir.glob('*.pdf'):
                try:
                    # Process PDF and extract text
                    result = self.preprocessor.process_pdf(file_path)
                    text = ' '.join([page['text'] for page in result['text_content']])
                    
                    if text.strip():  # Only add if we got some text
                        texts.append(text)
                        labels.append(category_idx)
                        categories.append(category)
                        self.logger.info(f"Processed {file_path.name}")
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
        
        if not texts:
            raise ValueError(f"No valid documents found in {split} split")
            
        self.logger.info(f"Loaded {len(texts)} documents from {split} split")
        return {
            'texts': texts,
            'labels': labels,
            'categories': categories
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name: str):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.categories)))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories,
                   yticklabels=self.categories)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plot_path = self.data_dir / 'evaluation' / 'plots' / f'confusion_matrix_{model_name.lower()}.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
    
    def train_and_evaluate(self):
        """Train and evaluate all models"""
        # Load training and test data
        train_data = self.load_data('train')
        test_data = self.load_data('test')
        
        # Initialize models with improved parameters
        models = {
            'SVM': SVMDocumentClassifier(),
            'Random Forest': RandomForestDocumentClassifier(),
            'Gradient Boosting': GradientBoostingDocumentClassifier(),
            'LSTM': LSTMDocumentClassifier(max_length=512, hidden_dim=256, num_layers=2),
            'Transformer': TransformerDocumentClassifier(model_name='distilbert-base-uncased')
        }
        
        # Create models directory if it doesn't exist
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Create results directory for transformer model
        results_dir = Path('results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            try:
                self.logger.info(f"Training {name} model...")
                
                # Train model
                if name in ['LSTM', 'Transformer']:
                    # Deep learning models need different training parameters
                    model.train(
                        train_data['texts'],
                        train_data['labels'],
                        batch_size=8,
                        epochs=3,  # Reduced epochs for faster training
                        validation_split=0.2
                    )
                else:
                    model.train(train_data['texts'], train_data['labels'])
                
                # Save model
                try:
                    model.save(str(models_dir))
                    self.logger.info(f"Saved {name} model successfully")
                except Exception as e:
                    self.logger.error(f"Error saving {name} model: {str(e)}")
                
                # Evaluate model
                self.logger.info(f"Evaluating {name} model...")
                predictions = []
                for text in test_data['texts']:
                    try:
                        pred = model.predict(text)
                        if isinstance(pred, dict):
                            pred_label = max(pred.items(), key=lambda x: x[1])[0]
                            predictions.append(self.categories.index(pred_label))
                        else:
                            predictions.append(pred)
                    except Exception as e:
                        self.logger.error(f"Error predicting with {name} model: {str(e)}")
                        continue
                
                # Calculate metrics
                try:
                    accuracy = accuracy_score(test_data['labels'], predictions)
                    per_category_f1 = f1_score(test_data['labels'], predictions, average=None)
                    
                    # Store results
                    results[name] = {
                        'accuracy': float(accuracy),
                        'per_category_f1': {
                            cat: float(score) 
                            for cat, score in zip(self.categories, per_category_f1)
                        }
                    }
                    
                    # Log results
                    self.logger.info(f"\n{name} Performance Summary:")
                    self.logger.info(f"Accuracy: {accuracy:.4f}")
                    self.logger.info("Per-category F1-scores:")
                    for cat, score in zip(self.categories, per_category_f1):
                        self.logger.info(f"  {cat}: {score:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error calculating metrics for {name} model: {str(e)}")
            
            except Exception as e:
                self.logger.error(f"Error processing {name} model: {str(e)}")
                continue
        
        # Save overall results
        try:
            results_path = self.data_dir / 'evaluation' / 'metrics' / 'model_performance.json'
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Saved performance results to {results_path}")
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
        
        return results  # Make sure to return the results

def main():
    """Main function to run model training and evaluation."""
    data_dir = Path("docs")
    
    trainer = ModelTrainer(data_dir)
    results = trainer.train_and_evaluate()
    
    # Print summary
    print("\nModel Performance Summary:")
    for model_name, report in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print("Per-category F1-scores:")
        for category in trainer.categories:
            print(f"  {category}: {report['per_category_f1'][category]:.4f}")

if __name__ == "__main__":
    main() 