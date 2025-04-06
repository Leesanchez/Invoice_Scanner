import os
from pathlib import Path
import json
import logging
from typing import Dict, List, Any
from datetime import datetime

from classification.document_classifier import get_classifier
from preprocessing.preprocessor import DocumentPreprocessor
from ocr.ocr_engine import OCREngine
from extraction.field_extractor import FieldExtractor

class DocumentExtractor:
    def __init__(self, model_path: str = 'models'):
        """
        Initialize the document extractor.
        
        Args:
            model_path: Path to the trained classification model
        """
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.model_path = Path(model_path)
        self.classifier = get_classifier('random_forest')  # Use Random Forest model
        
        # Load both model and vectorizer
        try:
            self.classifier.load(str(self.model_path))  # Load model and vectorizer
            self.classifier.vectorizer_fitted = True  # Explicitly set vectorizer as fitted
            self.logger.info("Successfully loaded model and vectorizer")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
        self.preprocessor = DocumentPreprocessor()
        self.ocr_engine = OCREngine()
        self.field_extractor = FieldExtractor()
    
    def classify_document(self, text: str) -> str:
        """
        Classify a document and return its type.
        
        Args:
            text: Document text content
            
        Returns:
            Document type (e.g., 'Invoices', 'PurchaseOrders')
        """
        predictions = self.classifier.predict(text)
        return max(predictions.items(), key=lambda x: x[1])[0]
    
    def extract_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract information from a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing extracted document information
        """
        # Convert document to images
        if file_path.lower().endswith('.pdf'):
            images = self.preprocessor.convert_pdf_to_images(file_path)
        else:
            # Assume it's an image file
            import cv2
            images = [cv2.imread(file_path)]
        
        all_text = []
        all_boxes = []
        
        # Process each page
        for image in images:
            # Preprocess the image
            processed_image = self.preprocessor.preprocess_image(image)
            
            # Extract text
            ocr_results = self.ocr_engine.extract_text(processed_image)
            page_text = ' '.join(result.text for result in ocr_results)
            all_text.append(page_text)
            
            # Collect bounding boxes
            boxes = [result.bounding_box for result in ocr_results]
            all_boxes.extend(boxes)
        
        # Combine text from all pages
        full_text = ' '.join(all_text)
        
        # Classify document
        doc_type = self.classify_document(full_text)
        self.logger.info(f"Document {file_path} classified as: {doc_type}")
        
        # Extract fields
        extracted_fields = self.field_extractor.extract_fields(
            full_text, all_boxes, doc_type
        )
        
        # Convert to structured format
        result = {
            'file_path': file_path,
            'document_type': doc_type,
            'extraction_date': datetime.now().isoformat(),
            'fields': {
                field: {
                    'value': info.value,
                    'confidence': info.confidence,
                    'method': info.method
                }
                for field, info in extracted_fields.items()
            }
        }
        
        return result

def process_directory(input_dir: str, output_file: str):
    """
    Process all documents in a directory and extract information.
    
    Args:
        input_dir: Directory containing documents to process
        output_file: Path to save the results
    """
    extractor = DocumentExtractor()
    results = []
    
    # Process all files in directory
    for file_path in Path(input_dir).glob('**/*'):
        if file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
            try:
                result = extractor.extract_document_info(str(file_path))
                if result:  # Only add if extraction was successful
                    results.append(result)
                    print(f"Processed document: {file_path.name}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nProcessed {len(results)} documents")
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract information from documents')
    parser.add_argument('--input-dir', required=True, help='Directory containing documents to process')
    parser.add_argument('--output-file', required=True, help='Path to save the extracted information')
    
    args = parser.parse_args()
    
    process_directory(args.input_dir, args.output_file) 