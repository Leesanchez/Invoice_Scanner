import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union
import numpy as np

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