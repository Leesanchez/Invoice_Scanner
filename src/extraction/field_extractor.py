import re
import spacy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import dateparser

@dataclass
class ExtractedField:
    value: str
    confidence: float
    method: str
    position: Optional[Dict[str, int]] = None

class FieldExtractor:
    def __init__(self):
        """Initialize the field extractor with NLP models and patterns."""
        # Load spaCy model
        self.nlp = spacy.load("en_core_web_sm")
        
        # Compile regex patterns for different document types
        self.patterns = {
            'invoice': {
                'invoice_number': [
                    r'(?i)order\s*id\s*:\s*(\d+)',
                    r'(?i)invoice\s*#?\s*(\w+[-/]?\w+)',
                    r'(?i)inv\s*#?\s*(\w+[-/]?\w+)'
                ],
                'customer_id': [
                    r'(?i)customer\s*id\s*:\s*(\w+)',
                    r'(?i)customer\s*number\s*:\s*(\w+)'
                ],
                'order_date': [
                    r'(?i)order\s*date\s*:\s*(\d{4}-\d{2}-\d{2})',
                    r'(?i)date\s*:\s*(\d{4}-\d{2}-\d{2})'
                ],
                'customer_name': [
                    r'(?i)contact\s*name\s*:\s*([^\n]+)',
                    r'(?i)customer\s*name\s*:\s*([^\n]+)'
                ],
                'total_amount': [
                    r'(?i)totalprice\s*(\d+\.?\d*)',
                    r'(?i)total\s*amount\s*:\s*(\d+\.?\d*)',
                    r'(?i)total\s*:\s*(\d+\.?\d*)'
                ],
                'shipping_address': [
                    r'(?i)address\s*:\s*([^\n]+)[\n\s]+city\s*:\s*([^\n]+)[\n\s]+postal\s*code\s*:\s*([^\n]+)[\n\s]+country\s*:\s*([^\n]+)',
                ],
                'contact_info': [
                    r'(?i)phone\s*:\s*([^\n]+)[\n\s]+fax\s*:\s*([^\n]+)'
                ]
            },
            'purchase_order': {
                'po_number': [
                    r'(?i)p\.?o\.?\s*#?\s*(\w+[-/]?\w+)',
                    r'(?i)purchase\s+order\s*#?\s*(\w+[-/]?\w+)'
                ],
                'order_date': [
                    r'(?i)order\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    r'(?i)date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
                ],
                'delivery_date': [
                    r'(?i)delivery\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    r'(?i)expected\s*delivery\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
                ],
                'total_amount': [
                    r'(?i)total\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                    r'(?i)amount\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)'
                ]
            },
            'shipping_order': {
                'tracking_number': [
                    r'(?i)tracking\s*#?\s*(\w+)',
                    r'(?i)shipment\s*#?\s*(\w+)'
                ],
                'ship_date': [
                    r'(?i)ship\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                    r'(?i)shipping\s*date\s*[:.]?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
                ],
                'delivery_address': [
                    r'(?i)deliver\s*to\s*:?\s*(.*?)(?=\n|$)',
                    r'(?i)ship\s*to\s*:?\s*(.*?)(?=\n|$)'
                ]
            },
            'monthly': {
                'month': [
                    r'(?i)month\s*[:.]?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)',
                    r'(?i)period\s*[:.]?\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)'
                ],
                'year': [
                    r'(?i)year\s*[:.]?\s*(\d{4})',
                    r'(?i)20(\d{2})'
                ],
                'total': [
                    r'(?i)monthly\s*total\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)',
                    r'(?i)total\s*[:.]?\s*[\$£€]?\s*([\d,]+\.?\d*)'
                ]
            }
        }
        
        # Compile all patterns
        for doc_type in self.patterns:
            for field in self.patterns[doc_type]:
                self.patterns[doc_type][field] = [
                    re.compile(p) for p in self.patterns[doc_type][field]
                ]
    
    def extract_fields(self, text: str, boxes: List[Dict[str, int]], 
                      doc_type: str) -> Dict[str, ExtractedField]:
        """
        Extract fields using multiple methods.
        
        Args:
            text: OCR text
            boxes: List of bounding boxes for text blocks
            doc_type: Type of document (invoice, purchase_order, etc.)
            
        Returns:
            Dictionary of extracted fields with confidence scores
        """
        results = {}
        
        # Map folder names to internal doc types
        doc_type_map = {
            'Invoices': 'invoice',
            'PurchaseOrders': 'purchase_order',
            'Shipping orders': 'shipping_order',
            'Monthly': 'monthly',
            'MonthlyCategory': 'monthly'
        }
        
        internal_type = doc_type_map.get(doc_type, doc_type)
        
        # Extract using regex
        regex_results = self._extract_with_regex(text, internal_type)
        results.update(regex_results)
        
        # Extract using spaCy for organization names and addresses
        spacy_results = self._extract_with_spacy(text, internal_type)
        for field, value in spacy_results.items():
            if field not in results or results[field].confidence < value.confidence:
                results[field] = value
        
        # Use positional heuristics for remaining fields
        positional_results = self._extract_with_position(text, boxes, internal_type)
        for field, value in positional_results.items():
            if field not in results or results[field].confidence < value.confidence:
                results[field] = value
        
        return results
    
    def _extract_with_regex(self, text: str, doc_type: str) -> Dict[str, ExtractedField]:
        """Extract fields using regex patterns."""
        results = {}
        
        if doc_type not in self.patterns:
            return results
        
        for field, patterns in self.patterns[doc_type].items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    if field == 'shipping_address' and len(match.groups()) == 4:
                        # Combine address components
                        value = f"{match.group(1)}, {match.group(2)}, {match.group(3)}, {match.group(4)}"
                    elif field == 'contact_info' and len(match.groups()) == 2:
                        # Combine phone and fax
                        value = f"Phone: {match.group(1)}, Fax: {match.group(2)}"
                    else:
                        value = match.group(1).strip()
                    
                    if 'amount' in field or 'total' in field:
                        value = value.replace(',', '')
                    elif 'date' in field:
                        parsed_date = dateparser.parse(value)
                        if parsed_date:
                            value = parsed_date.strftime('%Y-%m-%d')
                    
                    results[field] = ExtractedField(
                        value=value,
                        confidence=0.9,  # High confidence for regex matches
                        method='regex'
                    )
                    break
        
        return results
    
    def _extract_with_spacy(self, text: str, doc_type: str) -> Dict[str, ExtractedField]:
        """Extract fields using spaCy NER."""
        results = {}
        doc = self.nlp(text)
        
        # Extract organization names for all document types
        orgs = [ent for ent in doc.ents if ent.label_ == 'ORG']
        if len(orgs) >= 2:
            results['from_organization'] = ExtractedField(
                value=orgs[0].text,
                confidence=0.7,
                method='spacy'
            )
            results['to_organization'] = ExtractedField(
                value=orgs[1].text,
                confidence=0.7,
                method='spacy'
            )
        
        # Extract addresses for shipping orders
        if doc_type == 'shipping_order':
            addresses = []
            current_address = []
            
            for token in doc:
                if token.ent_type_ in ['GPE', 'LOC']:
                    current_address.append(token.text)
                elif current_address:
                    if len(current_address) >= 2:  # Consider it an address if it has at least 2 parts
                        addresses.append(' '.join(current_address))
                    current_address = []
            
            if addresses:
                results['addresses'] = ExtractedField(
                    value=addresses,
                    confidence=0.6,
                    method='spacy'
                )
        
        return results
    
    def _extract_with_position(self, text: str, boxes: List[Dict[str, int]], 
                             doc_type: str) -> Dict[str, ExtractedField]:
        """Extract fields using positional heuristics."""
        results = {}
        
        if not boxes:
            return results
        
        # Sort boxes by y-coordinate (top to bottom)
        sorted_boxes = sorted(
            enumerate(boxes),
            key=lambda x: (x[1]['y'], -x[1]['x'])
        )
        
        # Split text into words
        words = text.split()
        
        # Look for amounts in the bottom right for invoices and purchase orders
        if doc_type in ['invoice', 'purchase_order']:
            bottom_boxes = sorted_boxes[-5:]  # Last 5 boxes
            for idx, box in bottom_boxes:
                if idx < len(words):
                    text_part = words[idx]
                    if re.match(r'[\$£€]?\s*[\d,]+\.?\d*', text_part):
                        field_name = 'total_amount' if doc_type == 'purchase_order' else 'amount'
                        results[field_name] = ExtractedField(
                            value=re.sub(r'[^\d.]', '', text_part),
                            confidence=0.5,
                            method='position',
                            position=box
                        )
                        break
        
        return results 