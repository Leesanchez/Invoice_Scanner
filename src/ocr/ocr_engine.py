import pytesseract
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Dict[str, int]
    block_num: int
    line_num: int
    word_num: int

class OCREngine:
    def __init__(self, lang: str = 'eng', min_confidence: float = 60.0):
        """
        Initialize the OCR engine with improved settings.
        
        Args:
            lang (str): Language code for Tesseract
            min_confidence (float): Minimum confidence threshold for OCR results
        """
        self.lang = lang
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        
        # Define PSM modes for different layouts
        self.psm_modes = {
            'single_column': '6',  # Assume uniform block of text
            'multi_column': '4',   # Assume multiple columns
            'sparse': '11'         # Sparse text
        }
    
    def _get_best_psm(self, image: np.ndarray) -> str:
        """Determine the best PSM mode for the image."""
        # Try different PSM modes and select the one with highest average confidence
        best_psm = '11'  # Default to sparse text
        best_avg_conf = 0
        
        for mode_name, psm in self.psm_modes.items():
            config = f'--oem 1 --psm {psm}'
            try:
                data = pytesseract.image_to_data(
                    image,
                    lang=self.lang,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence
                confidences = [float(conf) for conf in data['conf'] if float(conf) > 0]
                if confidences:
                    avg_conf = sum(confidences) / len(confidences)
                    if avg_conf > best_avg_conf:
                        best_avg_conf = avg_conf
                        best_psm = psm
            except Exception as e:
                self.logger.warning(f"Error testing PSM {psm}: {str(e)}")
        
        return best_psm
    
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """
        Extract text from the image with improved accuracy.
        
        Args:
            image: Preprocessed image
            
        Returns:
            List of OCRResult objects containing text, confidence, and position
        """
        # Get best PSM mode
        best_psm = self._get_best_psm(image)
        config = f'--oem 1 --psm {best_psm}'
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            config=config,
            output_type=pytesseract.Output.DICT
        )
        
        results = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            # Skip empty results and low confidence
            if int(data['conf'][i]) < self.min_confidence:
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
                },
                block_num=data['block_num'][i],
                line_num=data['line_num'][i],
                word_num=data['word_num'][i]
            )
            results.append(result)
        
        return results

    def get_structured_data(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract structured data from the image with improved accuracy.
        
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
        
        # Group results by blocks and lines
        blocks = {}
        for result in results:
            if result.block_num not in blocks:
                blocks[result.block_num] = []
            blocks[result.block_num].append(result)
        
        # Process each block
        for block_num, block_results in blocks.items():
            # Sort by line and word number
            block_results.sort(key=lambda x: (x.line_num, x.word_num))
            
            # Combine text in the same line
            current_line = []
            current_line_num = None
            lines = []
            
            for result in block_results:
                if current_line_num is None:
                    current_line_num = result.line_num
                
                if result.line_num != current_line_num:
                    lines.append(' '.join(current_line))
                    current_line = [result.text]
                    current_line_num = result.line_num
                else:
                    current_line.append(result.text)
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Process each line
            for line in lines:
                line_lower = line.lower()
                
                # Look for invoice number
                if any(key in line_lower for key in ['invoice', 'inv', 'invoice no', 'invoice #']):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if any(key in part.lower() for key in ['invoice', 'inv', 'no', '#']):
                            if i + 1 < len(parts):
                                structured_data['invoice_number'] = parts[i + 1]
                            break
                
                # Look for dates
                if any(key in line_lower for key in ['date', 'invoice date']):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() in ['date:', 'date']:
                            if i + 1 < len(parts):
                                structured_data['invoice_date'] = parts[i + 1]
                            break
                
                if any(key in line_lower for key in ['due date', 'payment due']):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() in ['due:', 'due']:
                            if i + 1 < len(parts):
                                structured_data['due_date'] = parts[i + 1]
                            break
                
                # Look for total amount
                if any(key in line_lower for key in ['total', 'amount', 'balance due']):
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() in ['total:', 'amount:', 'due:']:
                            if i + 1 < len(parts):
                                structured_data['total_amount'] = parts[i + 1]
                            break
        
        return structured_data 