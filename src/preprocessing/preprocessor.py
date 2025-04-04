import cv2
import numpy as np
import pdfplumber
from pathlib import Path
from pdf2image import convert_from_path
from typing import Union, List, Tuple, Dict, Optional

class DocumentPreprocessor:
    def __init__(self, dpi: int = 300):
        """
        Initialize the document preprocessor.
        
        Args:
            dpi (int): DPI for PDF to image conversion
        """
        self.dpi = dpi

    def process_pdf(self, pdf_path: Union[str, Path]) -> Dict[str, Union[List[np.ndarray], List[Dict]]]:
        """
        Process a PDF document and extract both images and text content.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing:
                - 'images': List of preprocessed page images
                - 'text_content': List of dictionaries containing extracted text and metadata per page
        """
        # Convert PDF pages to images
        images = self.convert_pdf_to_images(pdf_path)
        preprocessed_images = [self.preprocess_image(img) for img in images]
        
        # Extract text content using pdfplumber
        text_content = self.extract_text_content(pdf_path)
        
        return {
            'images': preprocessed_images,
            'text_content': text_content
        }

    def convert_pdf_to_images(self, pdf_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Convert PDF document to list of images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of images as numpy arrays
        """
        images = convert_from_path(pdf_path, self.dpi)
        return [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) for img in images]

    def extract_text_content(self, pdf_path: Union[str, Path]) -> List[Dict]:
        """
        Extract text content and metadata from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing text content and metadata for each page
        """
        text_content = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text and its properties
                text = page.extract_text()
                
                # Extract tables if present
                tables = page.extract_tables()
                
                # Get text layout information
                words = page.extract_words()
                
                # Extract form fields if present
                form_fields = self._extract_form_fields(page)
                
                page_content = {
                    'text': text,
                    'tables': tables,
                    'words': words,
                    'form_fields': form_fields,
                    'page_number': page.page_number,
                    'width': page.width,
                    'height': page.height
                }
                
                text_content.append(page_content)
        
        return text_content

    def _extract_form_fields(self, page) -> List[Dict]:
        """
        Extract form fields from a PDF page.
        
        Args:
            page: pdfplumber page object
            
        Returns:
            List of dictionaries containing form field information
        """
        form_fields = []
        
        try:
            # Get annotations that might be form fields
            annots = page.annots if hasattr(page, 'annots') else []
            
            for annot in annots:
                if annot.get('subtype') == 'Widget':  # Form field
                    field = {
                        'type': annot.get('field_type'),
                        'name': annot.get('field_name'),
                        'value': annot.get('field_value'),
                        'bbox': annot.get('bbox')
                    }
                    form_fields.append(field)
        except Exception as e:
            print(f"Warning: Could not extract form fields: {str(e)}")
        
        return form_fields

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing steps to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Correct skew if needed
        corrected = self._correct_skew(binary)
        
        return corrected

    def _correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image skew.
        
        Args:
            image: Binary image
            
        Returns:
            Deskewed image
        """
        # Find all contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        angles = []
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small contours
                continue
            
            # Find minimum area rectangle
            rect = cv2.minAreaRect(contour)
            angle = rect[-1]
            
            # Convert angle to (-90, 90) range
            if angle < -45:
                angle = 90 + angle
            angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle for rotation
        median_angle = np.median(angles)
        
        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated 