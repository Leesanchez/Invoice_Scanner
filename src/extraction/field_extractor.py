import re
import spacy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import dateparser
import logging
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from PIL import Image

@dataclass
class ExtractedField:
    value: str
    confidence: float
    method: str
    bounding_box: Optional[Dict] = None

class FieldExtractor:
    def __init__(self):
        """Initialize the field extractor with improved components."""
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        self.nlp = spacy.load("en_core_web_sm")
        
        # Enhanced regex patterns
        self.patterns = {
            'invoice_number': [
                r'(?i)(?:invoice|inv)[\s#:]*([A-Z0-9][-A-Z0-9]*)',
                r'(?i)invoice\s*number\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)',
                r'(?i)order\s*number\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)',
                r'(?i)ref\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)'
            ],
            'date': [
                r'(?i)(?:date|invoice\s*date)\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)(?:date|invoice\s*date)\s*[:.]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})'
            ],
            'due_date': [
                r'(?i)(?:due\s*date|payment\s*due)\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)(?:due\s*date|payment\s*due)\s*[:.]?\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
                r'(?i)net\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'total_amount': [
                r'(?i)(?:total|amount|balance\s*due)\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                r'(?i)grand\s*total\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                r'(?i)amount\s*due\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                r'(?i)total\s*payable\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)'
            ]
        }
        
        # Improved position heuristics
        self.load_position_heuristics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_position_heuristics(self):
        """Load improved position heuristics for field locations."""
        self.position_heuristics = {
            'invoice_number': {
                'top': 0.1,
                'right': 0.3,
                'weight': 0.8
            },
            'date': {
                'top': 0.1,
                'right': 0.9,
                'weight': 0.8
            },
            'due_date': {
                'top': 0.2,
                'right': 0.9,
                'weight': 0.7
            },
            'total_amount': {
                'bottom': 0.9,
                'right': 0.9,
                'weight': 0.9
            },
            'issuer_name': {
                'top': 0.1,
                'left': 0.1,
                'weight': 0.7
            },
            'recipient_name': {
                'top': 0.3,
                'left': 0.1,
                'weight': 0.7
            }
        }
    
    def validate_date(self, date_str: str) -> bool:
        """Validate date format."""
        try:
            parsed_date = dateparser.parse(date_str)
            return parsed_date is not None
        except:
            return False
    
    def validate_amount(self, amount_str: str) -> bool:
        """Validate amount format."""
        try:
            # Remove currency symbols and commas
            clean_amount = re.sub(r'[^\d.]', '', amount_str)
            float(clean_amount)
            return True
        except:
            return False
    
    def extract_with_regex(self, text: str, field_type: str) -> Optional[ExtractedField]:
        """Extract field using enhanced regex patterns."""
        patterns = self.patterns.get(field_type, [])
        best_match = None
        best_confidence = 0
        
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group(1).strip()
                
                # Validate based on field type
                if field_type in ['date', 'due_date']:
                    if not self.validate_date(value):
                        continue
                elif field_type == 'total_amount':
                    if not self.validate_amount(value):
                        continue
                
                # Calculate confidence based on pattern match
                confidence = 0.8
                if 'invoice' in pattern.lower():
                    confidence += 0.1
                if 'number' in pattern.lower():
                    confidence += 0.1
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = ExtractedField(
                        value=value,
                        confidence=confidence,
                        method='regex'
                    )
        
        return best_match
    
    def extract_with_ner(self, text: str) -> Dict[str, ExtractedField]:
        """Extract entities using enhanced spaCy NER."""
        doc = self.nlp(text)
        entities = {}
        
        # Track positions of entities
        positions = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                positions.append((ent.start_char, ent.end_char))
        
        # Sort entities by position
        positions.sort()
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                # Determine if organization is issuer or recipient based on position
                # and context
                context_before = doc[max(0, ent.start - 5):ent.start].text.lower()
                context_after = doc[ent.end:min(ent.end + 5, len(doc))].text.lower()
                
                is_issuer = any(word in context_before 
                              for word in ['from', 'issued', 'by', 'sender'])
                is_recipient = any(word in context_before 
                                 for word in ['to', 'bill to', 'ship to'])
                
                # If context is unclear, use position
                if not is_issuer and not is_recipient:
                    is_issuer = positions.index((ent.start_char, ent.end_char)) == 0
                
                field_type = 'issuer_name' if is_issuer else 'recipient_name'
                
                if field_type not in entities or \
                   len(ent.text) > len(entities[field_type].value):
                    entities[field_type] = ExtractedField(
                        value=ent.text,
                        confidence=0.7 + (0.2 if is_issuer or is_recipient else 0),
                        method='ner'
                    )
        
        return entities
    
    def extract_with_position(
        self,
        text: str,
        boxes: List[Dict[str, float]],
        field_type: str
    ) -> Optional[ExtractedField]:
        """Extract field using improved positional heuristics."""
        heuristic = self.position_heuristics.get(field_type)
        if not heuristic or not boxes:
            return None
        
        # Normalize coordinates
        max_x = max(box['x'] + box['width'] for box in boxes)
        max_y = max(box['y'] + box['height'] for box in boxes)
        
        best_match = None
        best_score = 0
        
        for i, box in enumerate(boxes):
            # Normalize box coordinates
            x_norm = box['x'] / max_x
            y_norm = box['y'] / max_y
            
            # Calculate position score
            score = 1.0
            for pos, target in heuristic.items():
                if pos == 'top':
                    score *= 1 - abs(y_norm - target)
                elif pos == 'bottom':
                    score *= 1 - abs((y_norm + box['height']/max_y) - target)
                elif pos == 'left':
                    score *= 1 - abs(x_norm - target)
                elif pos == 'right':
                    score *= 1 - abs((x_norm + box['width']/max_x) - target)
            
            # Apply weight
            score *= heuristic.get('weight', 1.0)
            
            if score > best_score:
                best_score = score
                best_match = ExtractedField(
                    value=text[i:i+len(box.get('text', ''))],
                    confidence=score,
                    method='position',
                    bounding_box=box
                )
        
        return best_match if best_score > 0.5 else None
    
    def extract_fields(
        self,
        text: str,
        boxes: List[Dict[str, float]],
        document_type: str,
        image: Optional[Image.Image] = None
    ) -> Dict[str, ExtractedField]:
        """
        Extract fields using multiple methods.
        
        Args:
            text: Document text
            boxes: List of bounding boxes from OCR
            document_type: Type of document (e.g., 'Invoices')
            image: Optional PIL Image (not used)
            
        Returns:
            Dictionary of extracted fields
        """
        if document_type != 'Invoices':
            return {}
        
        self.logger.info("Extracting fields from invoice...")
        fields = {}
        
        # 1. Extract using regex for all fields
        for field_type in self.patterns.keys():
            result = self.extract_with_regex(text, field_type)
            if result:
                fields[field_type] = result
                self.logger.info(f"Found {field_type} using regex: {result.value}")
        
        # 2. Extract using NER for missing fields
        ner_results = self.extract_with_ner(text)
        for field_type, result in ner_results.items():
            if field_type not in fields:
                fields[field_type] = result
                self.logger.info(f"Found {field_type} using NER: {result.value}")
        
        # 3. Try position-based extraction for any remaining fields
        for field_type in self.position_heuristics.keys():
            if field_type not in fields:
                result = self.extract_with_position(text, boxes, field_type)
                if result:
                    fields[field_type] = result
                    self.logger.info(f"Found {field_type} using position: {result.value}")
        
        return fields

def main():
    """Test the field extractor with a sample invoice."""
    # Initialize extractor
    extractor = FieldExtractor()
    
    # Load a sample invoice
    sample_text = """
    INVOICE
    Invoice #: INV-2023-001
    Date: 03/15/2023
    Due Date: 04/15/2023
    
    From:
    ABC Company Ltd.
    123 Business St.
    
    To:
    XYZ Corporation
    456 Client Ave.
    
    Total Amount: $1,234.56
    """
    
    # Sample boxes (normally from OCR)
    sample_boxes = [
        {'x': 100, 'y': 50, 'width': 200, 'height': 30, 'text': 'INV-2023-001'},
        {'x': 500, 'y': 50, 'width': 100, 'height': 30, 'text': '03/15/2023'},
        # ... more boxes ...
    ]
    
    # Extract fields
    results = extractor.extract_fields(sample_text, sample_boxes, 'Invoices')
    
    # Print results
    print("\nExtracted Fields:")
    for field, info in results.items():
        print(f"{field}: {info.value} (confidence: {info.confidence:.2f})")

if __name__ == "__main__":
    main() 