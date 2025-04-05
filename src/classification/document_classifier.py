import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import numpy as np
from abc import ABC, abstractmethod
import pickle
import os
import pandas as pd
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class BaseDocumentClassifier(ABC):
    """Abstract base class for document classifiers"""
    
    def __init__(self, model_name: str, categories: List[str]):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the classifier
            categories: List of document categories
        """
        self.model_name = model_name
        self.categories = categories
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            min_df=2
        )
    
    @abstractmethod
    def train(self, texts: List[str], labels: List[int]) -> None:
        """
        Train the classifier.
        
        Args:
            texts: List of document texts
            labels: List of corresponding labels (indices of self.categories)
        """
        pass
    
    def predict(self, text: str) -> Dict[str, float]:
        """
        Classify a document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of category probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Get probabilities
        probas = self.model.predict_proba(X)[0]
        
        # Map to categories
        return {cat: float(prob) for cat, prob in zip(self.categories, probas)}
    
    def save(self, base_dir: str) -> None:
        """
        Save the trained model and vectorizer.
        
        Args:
            base_dir: Directory to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Create directory if it doesn't exist
        save_dir = Path(base_dir) / 'models'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and vectorizer
        joblib.dump(self.model, save_dir / f"{self.model_name}_model.pkl")
        joblib.dump(self.vectorizer, save_dir / f"{self.model_name}_vectorizer.pkl")
    
    def load(self, base_dir: str) -> None:
        """
        Load the trained model and vectorizer.
        
        Args:
            base_dir: Directory where the model is saved
        """
        save_dir = Path(base_dir) / 'models'
        
        # Load model and vectorizer
        self.model = joblib.load(save_dir / f"{self.model_name}_model.pkl")
        self.vectorizer = joblib.load(save_dir / f"{self.model_name}_vectorizer.pkl")

class SVMDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("svm", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train SVM classifier
        self.model = SVC(kernel='linear', probability=True)
        self.model.fit(X, labels)

class RandomForestDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("random_forest", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train Random Forest classifier
        self.model = RandomForestClassifier(n_estimators=100)
        self.model.fit(X, labels)

class LogisticRegressionDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("logistic_regression", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train Logistic Regression classifier
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

class DecisionTreeDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("decision_tree", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train Decision Tree classifier
        self.model = DecisionTreeClassifier()
        self.model.fit(X, labels)

class XGBoostDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("xgboost", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train XGBoost classifier
        self.model = xgb.XGBClassifier()
        self.model.fit(X, labels)

class LightGBMDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("lightgbm", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train LightGBM classifier
        self.model = lgb.LGBMClassifier()
        self.model.fit(X, labels)

class CatBoostDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("catboost", categories)
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Train CatBoost classifier
        self.model = CatBoostClassifier(verbose=0)
        self.model.fit(X, labels)

class LSTMDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, categories: List[str]):
        super().__init__("lstm", categories)
        try:
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
            
            self.tokenizer = Tokenizer(num_words=10000)
            self.max_sequence_length = 500
            self.tensorflow_available = True
        except ImportError:
            print("Warning: TensorFlow not available. LSTM classifier will not work.")
            self.tensorflow_available = False
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        if not self.tensorflow_available:
            raise ImportError("TensorFlow is not available. Please install it to use the LSTM classifier.")
            
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
        import tensorflow as tf
        import numpy as np
        
        try:
            # Fit tokenizer
            self.tokenizer.fit_on_texts(texts)
            
            # Convert texts to sequences
            sequences = self.tokenizer.texts_to_sequences(texts)
            
            # Pad sequences
            X = pad_sequences(sequences, maxlen=self.max_sequence_length)
            
            # Convert labels to one-hot
            y = tf.keras.utils.to_categorical(labels, num_classes=len(self.categories))
            
            # Create LSTM model
            self.model = Sequential()
            self.model.add(Embedding(input_dim=len(self.tokenizer.word_index) + 1,
                                   output_dim=128,
                                   input_length=self.max_sequence_length))
            self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(Dense(len(self.categories), activation='softmax'))
            
            # Compile model
            self.model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
            
            # Train model
            self.model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1)
        except Exception as e:
            print(f"Error training LSTM model: {str(e)}")
            raise
    
    def predict(self, text: str) -> Dict[str, float]:
        if not self.tensorflow_available:
            raise ImportError("TensorFlow is not available. Please install it to use the LSTM classifier.")
            
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        try:
            # Convert text to sequence
            sequence = self.tokenizer.texts_to_sequences([text])
            
            # Pad sequence
            X = pad_sequences(sequence, maxlen=self.max_sequence_length)
            
            # Get probabilities
            probas = self.model.predict(X)[0]
            
            # Map to categories
            return {cat: float(prob) for cat, prob in zip(self.categories, probas)}
        except Exception as e:
            print(f"Error predicting with LSTM model: {str(e)}")
            raise

def get_classifier(model_type: str, categories: List[str]) -> BaseDocumentClassifier:
    """
    Get a document classifier instance based on the model type.
    
    Args:
        model_type: Type of classifier to create
        categories: List of document categories
        
    Returns:
        Instance of the specified classifier
    """
    classifiers = {
        'svm': SVMDocumentClassifier,
        'random_forest': RandomForestDocumentClassifier,
        'logistic_regression': LogisticRegressionDocumentClassifier,
        'decision_tree': DecisionTreeDocumentClassifier,
        'xgboost': XGBoostDocumentClassifier,
        'lightgbm': LightGBMDocumentClassifier,
        'catboost': CatBoostDocumentClassifier,
        'lstm': LSTMDocumentClassifier
    }
    
    if model_type not in classifiers:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return classifiers[model_type](categories)

class DocumentClassifier:
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        """
        Initialize the document classifier.
        
        Args:
            model_name (str): Name of the pretrained transformer model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=4  # invoice, contract, receipt, email
        )
        self.labels = ['invoice', 'contract', 'receipt', 'email']
        
    def classify_text(self, text: str) -> Dict[str, float]:
        """
        Classify the document based on its text content.
        
        Args:
            text: Document text content
            
        Returns:
            Dictionary with classification probabilities for each category
        """
        # Prepare input
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
        
        # Convert to dictionary
        result = {
            label: float(prob)
            for label, prob in zip(self.labels, probabilities[0])
        }
        
        return result
    
    def is_invoice(self, text: str, threshold: float = 0.7) -> bool:
        """
        Check if the document is an invoice.
        
        Args:
            text: Document text content
            threshold: Confidence threshold for classification
            
        Returns:
            Boolean indicating if the document is an invoice
        """
        classifications = self.classify_text(text)
        return classifications['invoice'] > threshold
    
    def train(self, texts: List[str], labels: List[int], 
             epochs: int = 3, batch_size: int = 8):
        """
        Fine-tune the model on custom data.
        
        Args:
            texts: List of document texts
            labels: List of corresponding labels (0: invoice, 1: contract, etc.)
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        from torch.utils.data import Dataset, DataLoader
        from transformers import AdamW
        
        class DocumentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer):
                self.encodings = tokenizer(texts, truncation=True, padding=True)
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) 
                       for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
        
        # Prepare dataset
        dataset = DocumentDataset(texts, labels, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Prepare optimizer
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            for batch in loader:
                optimizer.zero_grad()
                
                inputs = {k: v for k, v in batch.items() 
                         if k in ['input_ids', 'attention_mask', 'labels']}
                outputs = self.model(**inputs)
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        self.model.eval() 