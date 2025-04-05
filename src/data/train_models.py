import sys
from pathlib import Path
import logging
import json
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from classification.document_classifier import (
    SVMDocumentClassifier,
    RandomForestDocumentClassifier,
    LogisticRegressionDocumentClassifier,
    DecisionTreeDocumentClassifier,
    XGBoostDocumentClassifier,
    LightGBMDocumentClassifier,
    CatBoostDocumentClassifier
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
                logging.FileHandler(self.data_dir / 'training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_split_info(self):
        """Load the dataset split information."""
        split_info_path = self.data_dir / 'split_info.json'
        with open(split_info_path) as f:
            self.split_info = json.load(f)
        self.categories = list(self.split_info['categories'].keys())
        
    def load_and_preprocess_data(self, split: str) -> Tuple[List[str], List[str]]:
        """
        Load and preprocess documents from a specific split.
        
        Args:
            split: Either 'train' or 'test'
            
        Returns:
            Tuple of (texts, labels)
        """
        texts = []
        labels = []
        
        for category in self.categories:
            split_dir = self.data_dir / split / category
            self.logger.info(f"Processing {category} documents from {split} set")
            
            for file_name in self.split_info['categories'][category][split]:
                try:
                    file_path = split_dir / file_name
                    if not file_path.exists():
                        self.logger.warning(f"File not found: {file_path}")
                        continue
                        
                    # Process document
                    result = self.preprocessor.process_pdf(file_path)
                    text = ' '.join([page['text'] for page in result['text_content']])
                    
                    if text.strip():
                        texts.append(text)
                        labels.append(category)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {file_path}: {str(e)}")
        
        return texts, labels
    
    def train_and_evaluate(self):
        """Train and evaluate all models."""
        # Load data
        self.logger.info("Loading training data...")
        train_texts, train_labels = self.load_and_preprocess_data('train')
        
        self.logger.info("Loading test data...")
        test_texts, test_labels = self.load_and_preprocess_data('test')
        
        # Initialize models
        models = {
            'SVM': SVMDocumentClassifier(self.categories),
            'Random Forest': RandomForestDocumentClassifier(self.categories),
            'Logistic Regression': LogisticRegressionDocumentClassifier(self.categories),
            'Decision Tree': DecisionTreeDocumentClassifier(self.categories),
            'XGBoost': XGBoostDocumentClassifier(self.categories),
            'LightGBM': LightGBMDocumentClassifier(self.categories),
            'CatBoost': CatBoostDocumentClassifier(self.categories)
        }
        
        # Create models directory
        models_dir = self.data_dir / 'evaluation' / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Train and evaluate each model
        results = {}
        for name, model in models.items():
            self.logger.info(f"\nTraining {name}...")
            try:
                # Train model
                model.train(train_texts, train_labels)
                
                # Save model
                model.save(str(models_dir))
                self.logger.info(f"Saved {name} model")
                
                # Evaluate on test set
                predictions = []
                for text in test_texts:
                    pred = model.predict(text)
                    pred_category = max(pred.items(), key=lambda x: x[1])[0]
                    predictions.append(pred_category)
                
                # Calculate metrics
                report = classification_report(
                    test_labels,
                    predictions,
                    target_names=self.categories,
                    output_dict=True
                )
                
                results[name] = report
                self.logger.info(f"\n{name} Results:")
                self.logger.info(f"Accuracy: {report['accuracy']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
        
        # Save evaluation results
        results_path = self.data_dir / 'evaluation' / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\nResults saved to {results_path}")
        return results

def main():
    """Main function to run model training and evaluation."""
    data_dir = Path("/Users/sm_aswin21/Desktop/Anthony_Work/Invoice_Scanner/data_test")
    
    trainer = ModelTrainer(data_dir)
    results = trainer.train_and_evaluate()
    
    # Print summary
    print("\nModel Performance Summary:")
    for model_name, report in results.items():
        print(f"\n{model_name}:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print("Per-category F1-scores:")
        for category in trainer.categories:
            print(f"  {category}: {report[category]['f1-score']:.4f}")

if __name__ == "__main__":
    main() 