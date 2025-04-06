import os
from pathlib import Path
from typing import Dict, Any, List
import json
import logging
from preprocessing.preprocessor import DocumentPreprocessor
from ocr.ocr_engine import OCREngine
import cv2

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def preprocess_documents(input_dir: str, output_dir: str):
    """
    Preprocess all documents in the input directory.
    
    Args:
        input_dir: Directory containing documents to process
        output_dir: Directory to save preprocessed documents
    """
    logger = setup_logging()
    preprocessor = DocumentPreprocessor()
    ocr_engine = OCREngine()
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each category
    categories = ['Invoices', 'PurchaseOrders', 'Shipping orders']
    for category in categories:
        category_dir = Path(input_dir) / category
        if not category_dir.exists():
            logger.warning(f"Category directory not found: {category_dir}")
            continue
        
        logger.info(f"\nProcessing {category}...")
        
        # Create category output directory
        category_output = output_path / category
        category_output.mkdir(exist_ok=True)
        
        # Process all files in category
        for file_path in category_dir.glob('*'):
            if file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
                try:
                    logger.info(f"Processing: {file_path.name}")
                    
                    # Process document
                    result = preprocessor.process_pdf(str(file_path))
                    
                    # Save preprocessed images
                    for i, image in enumerate(result['images']):
                        image_path = category_output / f"{file_path.stem}_page{i+1}.png"
                        cv2.imwrite(str(image_path), image)
                    
                    # Save text content
                    text_path = category_output / f"{file_path.stem}_text.json"
                    with open(text_path, 'w') as f:
                        json.dump(result['text_content'], f, indent=2)
                    
                    logger.info(f"Saved preprocessed files for: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess documents for information extraction')
    parser.add_argument('--input-dir', required=True, help='Directory containing documents to process')
    parser.add_argument('--output-dir', required=True, help='Directory to save preprocessed documents')
    
    args = parser.parse_args()
    
    preprocess_documents(args.input_dir, args.output_dir) 