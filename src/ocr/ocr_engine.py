import pytesseract
import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Dict[str, int]

class OCREngine:
    def __init__(self, lang: str = 'eng', psm: int = 11):
        """
        Initialize the OCR engine.
        
        Args:
            lang (str): Language code for Tesseract
            psm (int): Page segmentation mode
        """
        self.config = f'--oem 1 --psm {psm}'
        self.lang = lang

    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from the image with detailed information.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of OCRResult objects containing text, confidence, and position
        """
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=self.config,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Skip empty results
            if int(data['conf'][i]) < 0:
                continue
                
            if not data['text'][i].strip():
                continue
            
            result = OCRResult(
                text=data['text'][i],
                confidence=float(data['conf'][i]),
                bounding_box={
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i]
                }
            )
            results.append(result)
        
        return results

    def get_structured_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract structured data from the image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Dictionary containing structured information
        """
        results = self.extract_text(image)
        
        # Initialize structured data
        structured_data = {
            'invoice_number': None,
            'invoice_date': None,
            'due_date': None,
            'issuer_name': None,
            'recipient_name': None,
            'total_amount': None
        }
        
        # Process each text block
        text_blocks = [result.text.lower() for result in results]
        
        for i, text in enumerate(text_blocks):
            # Look for invoice number
            if any(key in text for key in ['invoice', 'inv', 'invoice no', 'invoice #']):
                if i + 1 < len(results):
                    structured_data['invoice_number'] = results[i + 1].text
            
            # Look for dates
            if any(key in text for key in ['date', 'invoice date']):
                if i + 1 < len(results):
                    structured_data['invoice_date'] = results[i + 1].text
            
            if any(key in text for key in ['due date', 'payment due']):
                if i + 1 < len(results):
                    structured_data['due_date'] = results[i + 1].text
            
            # Look for total amount
            if any(key in text for key in ['total', 'amount', 'balance due']):
                if i + 1 < len(results):
                    structured_data['total_amount'] = results[i + 1].text
        
        return structured_data 