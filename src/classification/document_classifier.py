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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import logging

class BaseDocumentClassifier(ABC):
    """Abstract base class for document classifiers"""
    
    def __init__(self, model_name: str):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the classifier
        """
        self.model_name = model_name
        self.categories = ['Invoices', 'PurchaseOrders', 'Shipping orders', 'Other']
        self.model = None
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced to avoid overfitting
            ngram_range=(1, 2),  # Reduced to bigrams to avoid sparsity
            min_df=1,  # Allow terms that appear only once
            max_df=1.0,  # Don't filter out any terms based on document frequency
            stop_words='english',
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        self.is_fitted = False
        self.class_weights = None
        self.vectorizer_fitted = False
    
    def _calculate_class_weights(self, labels: List[int]) -> Dict[int, float]:
        """Calculate class weights to handle imbalance."""
        from collections import Counter
        label_counts = Counter(labels)
        total_samples = len(labels)
        weights = {
            label: total_samples / (len(label_counts) * count)
            for label, count in label_counts.items()
        }
        return weights
    
    def _extract_features(self, texts: List[str], is_training: bool = False) -> np.ndarray:
        """Enhanced feature extraction with additional text features."""
        # Basic TF-IDF features
        if is_training:
            tfidf_features = self.vectorizer.fit_transform(texts)
            self.vectorizer_fitted = True
        else:
            if not self.vectorizer_fitted:
                raise ValueError("Vectorizer must be fit on training data first")
            tfidf_features = self.vectorizer.transform(texts)
        
        # Additional features
        additional_features = []
        for text in texts:
            # Document length features
            words = text.split()
            features = [
                len(words),  # Total words
                len(set(words)),  # Unique words
                len(text),  # Total characters
                len(text.split('\n')),  # Number of lines
                sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,  # Uppercase ratio
                sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0,  # Digit ratio
            ]
            additional_features.append(features)
        
        # Combine features
        additional_features = np.array(additional_features)
        return np.hstack([tfidf_features.toarray(), additional_features])
    
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
        
        # Extract features
        features = self._extract_features([text])
        
        # Get probabilities
        probas = self.model.predict_proba(features)[0]
        
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
        self.vectorizer_fitted = True  # Set the flag after loading
        self.is_fitted = True

class SVMDocumentClassifier(BaseDocumentClassifier):
    """SVM-based document classifier"""
    
    def __init__(self):
        super().__init__("SVM")
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced'  # Enable automatic class balancing
        )
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Calculate class weights
        self.class_weights = self._calculate_class_weights(labels)
        
        # Extract features
        X = self._extract_features(texts, is_training=True)
        
        # Train model
        self.model.fit(X, labels)
        self.is_fitted = True

class RandomForestDocumentClassifier(BaseDocumentClassifier):
    """Random Forest-based document classifier"""
    
    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=200,  # Increased from 100
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',  # Enable automatic class balancing
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Calculate class weights
        self.class_weights = self._calculate_class_weights(labels)
        
        # Extract features
        X = self._extract_features(texts, is_training=True)
        
        # Train model
        self.model.fit(X, labels)
        self.is_fitted = True

class GradientBoostingDocumentClassifier(BaseDocumentClassifier):
    """Gradient Boosting-based document classifier"""
    
    def __init__(self):
        super().__init__("Gradient Boosting")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.8,
            random_state=42
        )
    
    def train(self, texts: List[str], labels: List[int]) -> None:
        # Calculate class weights
        self.class_weights = self._calculate_class_weights(labels)
        
        # Extract features
        X = self._extract_features(texts, is_training=True)
        
        # Train model
        self.model.fit(X, labels)
        self.is_fitted = True

class LSTMDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, max_length=512, hidden_dim=256, num_layers=2):
        super().__init__("lstm")
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize LSTM model with correct input size
        self.model = nn.Sequential(
            nn.LSTM(
                input_size=5000,  # TF-IDF vector size
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2,
                bidirectional=True  # Add bidirectional LSTM
            ),
            nn.Linear(hidden_dim * 2, len(self.categories))  # Multiply by 2 for bidirectional
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def _extract_features(self, texts: List[str], is_training: bool = False) -> np.ndarray:
        """Enhanced feature extraction with fixed size output."""
        # Basic TF-IDF features
        if is_training:
            tfidf_features = self.vectorizer.fit_transform(texts)
            self.vectorizer_fitted = True
        else:
            if not self.vectorizer_fitted:
                raise ValueError("Vectorizer must be fit on training data first")
            tfidf_features = self.vectorizer.transform(texts)
        
        # Ensure the output has exactly 5000 features
        if tfidf_features.shape[1] > 5000:
            tfidf_features = tfidf_features[:, :5000]
        elif tfidf_features.shape[1] < 5000:
            # Pad with zeros if needed
            padding = np.zeros((tfidf_features.shape[0], 5000 - tfidf_features.shape[1]))
            tfidf_features = np.hstack([tfidf_features.toarray(), padding])
        else:
            tfidf_features = tfidf_features.toarray()
            
        return tfidf_features
    
    def train(self, texts: List[str], labels: List[int], batch_size=8, epochs=10, validation_split=0.2) -> None:
        # Extract features
        X = self._extract_features(texts, is_training=True)
        y = np.array(labels)  # Use labels directly since they're already indices
        
        # Split into train and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).unsqueeze(1).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).unsqueeze(1).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        
        # Create DataLoaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                lstm_out, _ = self.model[0](batch_X)  # Get LSTM output
                output = self.model[1](lstm_out[:, -1, :])  # Use last hidden state
                loss = self.criterion(output, batch_y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    lstm_out, _ = self.model[0](batch_X)
                    output = self.model[1](lstm_out[:, -1, :])
                    loss = self.criterion(output, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            # Print epoch statistics
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}')
            print(f'Val Accuracy: {100*correct/total:.2f}%')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_model_state = self.model.state_dict()
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        self.is_fitted = True
    
    def predict(self, text: str) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        # Extract features
        features = self._extract_features([text])
        
        # Convert to tensor
        X = torch.FloatTensor(features).unsqueeze(1).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            lstm_out, _ = self.model[0](X)
            output = self.model[1](lstm_out[:, -1, :])
            probabilities = torch.softmax(output, dim=1)
        
        # Map to categories
        return {cat: float(prob) for cat, prob in zip(self.categories, probabilities[0])}

class TransformerDocumentClassifier(BaseDocumentClassifier):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(self.categories)
        ).to(self.device)
    
    def train(self, texts: List[str], labels: List[int], batch_size=8, epochs=3, validation_split=0.2) -> None:
        # Use labels directly since they're already indices
        y = np.array(labels)
        
        # Split into train and validation sets
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, y, test_size=validation_split, random_state=42
        )
        
        # Tokenize texts
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Create datasets
        class DocumentDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
                
            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
                
            def __len__(self):
                return len(self.labels)
        
        train_dataset = DocumentDataset(train_encodings, train_labels)
        val_dataset = DocumentDataset(val_encodings, val_labels)
        
        # Create output directory if it doesn't exist
        output_dir = Path('./results')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
            
            # Train model
        trainer.train()
        self.is_fitted = True
    
    def predict(self, text: str) -> Dict[str, float]:
        if not self.is_fitted:
            raise ValueError("Model not trained")
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
            # Map to categories
        return {cat: float(prob) for cat, prob in zip(self.categories, probabilities[0])}

def get_classifier(model_type: str) -> BaseDocumentClassifier:
    """
    Get a document classifier instance based on the model type.
    
    Args:
        model_type: Type of classifier to create
        
    Returns:
        Instance of the specified classifier
    """
    classifiers = {
        'svm': SVMDocumentClassifier,
        'random_forest': RandomForestDocumentClassifier,
        'gradient_boosting': GradientBoostingDocumentClassifier,
        'lstm': LSTMDocumentClassifier,
        'transformer': TransformerDocumentClassifier
    }
    
    if model_type not in classifiers:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return classifiers[model_type]()

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