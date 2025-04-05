import re
import spacy
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import dateparser
import logging
from pathlib import Path
import json
from datetime import datetime

@dataclass
class ExtractedField:
    value: str
    confidence: float
    method: str
    bounding_box: Optional[Dict] = None

class FieldExtractor:
    def __init__(self):
        """Initialize the field extractor with necessary components."""
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logging.warning("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize regex patterns
        self.patterns = {
                'invoice_number': [
                r'(?i)invoice\s*#?\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)',
                r'(?i)invoice\s*number\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)',
                r'(?i)inv\s*#?\s*[:.]?\s*([A-Z0-9][-A-Z0-9]*)'
            ],
            'date': [
                r'(?i)date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)invoice\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'due_date': [
                r'(?i)due\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(?i)payment\s*due\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
                ],
                'total_amount': [
                r'(?i)total\s*amount\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                    r'(?i)total\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                r'(?i)amount\s*due\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)'
            ]
        }
        
        # Load field position heuristics
        self.load_position_heuristics()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_position_heuristics(self):
        """Load position heuristics for field locations."""
        self.position_heuristics = {
            'invoice_number': {'top': 0.1, 'right': 0.3},  # Usually top-right
            'date': {'top': 0.1, 'right': 0.9},  # Usually top-right
            'due_date': {'top': 0.2, 'right': 0.9},  # Usually below date
            'total_amount': {'bottom': 0.9, 'right': 0.9},  # Usually bottom-right
            'issuer_name': {'top': 0.1, 'left': 0.1},  # Usually top-left
            'recipient_name': {'top': 0.3, 'left': 0.1}  # Usually middle-left
        }
    
    def extract_with_regex(self, text: str, field_type: str) -> Optional[ExtractedField]:
        """Extract field using regex patterns."""
        patterns = self.patterns.get(field_type, [])
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                value = match.group(1).strip()
                # Basic validation
                if field_type == 'total_amount':
                    try:
                        float(value.replace(',', ''))
                    except ValueError:
                        continue
                return ExtractedField(
                    value=value,
                    confidence=0.8,  # Base confidence for regex matches
                    method='regex'
                )
        return None
    
    def extract_with_ner(self, text: str) -> Dict[str, ExtractedField]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                # Determine if organization is issuer or recipient based on position
                # and context
                context_before = doc[max(0, ent.start - 5):ent.start].text.lower()
                is_issuer = any(word in context_before 
                              for word in ['from', 'issued', 'by', 'sender'])
                
                field_type = 'issuer_name' if is_issuer else 'recipient_name'
                
                if field_type not in entities or \
                   len(ent.text) > len(entities[field_type].value):
                    entities[field_type] = ExtractedField(
                        value=ent.text,
                        confidence=ent._.confidence 
                            if hasattr(ent._, 'confidence') else 0.7,
                        method='ner'
                    )
        
        return entities
    
    def extract_with_position(
        self,
        text: str,
        boxes: List[Dict[str, float]],
        field_type: str
    ) -> Optional[ExtractedField]:
        """Extract field using positional heuristics."""
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
        document_type: str
    ) -> Dict[str, ExtractedField]:
        """
        Extract all fields from the document using multiple methods.
        
        Args:
            text: Document text
            boxes: List of bounding boxes from OCR
            document_type: Type of document (e.g., 'Invoices')
            
        Returns:
            Dictionary of extracted fields
        """
        if document_type != 'Invoices':
            return {}
        
        self.logger.info("Extracting fields from invoice...")
        fields = {}
        
        # 1. Extract using regex
        for field_type in self.patterns.keys():
            result = self.extract_with_regex(text, field_type)
            if result:
                fields[field_type] = result
                self.logger.info(f"Found {field_type} using regex: {result.value}")
        
        # 2. Extract using NER
        ner_results = self.extract_with_ner(text)
        fields.update(ner_results)
        for field_type, result in ner_results.items():
            self.logger.info(f"Found {field_type} using NER: {result.value}")
        
        # 3. Extract using position for missing fields
        for field_type in self.position_heuristics.keys():
            if field_type not in fields:
                result = self.extract_with_position(text, boxes, field_type)
                if result:
                    fields[field_type] = result
                    self.logger.info(
                        f"Found {field_type} using position: {result.value}"
                    )
        
        # Log extraction summary
        self.logger.info("\nExtraction Summary:")
        for field_type, result in fields.items():
            self.logger.info(
                f"{field_type}: {result.value} "
                f"(confidence: {result.confidence:.2f}, method: {result.method})"
            )
        
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