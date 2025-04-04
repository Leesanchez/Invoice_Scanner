import os
from pathlib import Path
from typing import Dict, Any, List
import json

from preprocessing.preprocessor import DocumentPreprocessor
from ocr.ocr_engine import OCREngine
from extraction.field_extractor import FieldExtractor
from data.dataset_manager import DatasetManager
from evaluation.evaluator import ModelEvaluator

class DocumentProcessor:
    def __init__(self, base_dir: str):
        """
        Initialize the document processing pipeline.
        
        Args:
            base_dir: Base directory for data and outputs
        """
        self.preprocessor = DocumentPreprocessor()
        self.ocr_engine = OCREngine()
        self.field_extractor = FieldExtractor()
        self.dataset_manager = DatasetManager(base_dir)
        self.evaluator = ModelEvaluator(Path(base_dir) / 'evaluation')
        
    def process_document(self, file_path: str, category: str) -> Dict[str, Any]:
        """
        Process a single document through the pipeline.
        
        Args:
            file_path: Path to the document file
            category: Document category (e.g., 'Invoices', 'PurchaseOrders')
            
        Returns:
            Dictionary containing extracted information
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
        
        # Extract fields based on document category
        extracted_fields = self.field_extractor.extract_fields(
            full_text, all_boxes, category
        )
        
        # Convert ExtractedField objects to simple values
        structured_data = {
            field: info.value
            for field, info in extracted_fields.items()
        }
        
        # Add metadata
        structured_data['file_path'] = file_path
        structured_data['category'] = category
        structured_data['extraction_methods'] = {
            field: info.method
            for field, info in extracted_fields.items()
        }
        structured_data['confidence_scores'] = {
            field: info.confidence
            for field, info in extracted_fields.items()
        }
        
        return structured_data

def process_directory(input_dir: str, output_file: str, base_dir: str = 'data'):
    """
    Process all documents in a directory and save results to a JSON file.
    
    Args:
        input_dir: Directory containing documents to process
        output_file: Path to save the results
        base_dir: Base directory for data management
    """
    processor = DocumentProcessor(base_dir)
    results = []
    
    # Process each category
    for category in processor.dataset_manager.categories:
        category_dir = Path(input_dir) / category
        if not category_dir.exists():
            continue
        
        print(f"\nProcessing {category}...")
        
        # Process all files in category
        for file_path in category_dir.glob('*'):
            if file_path.suffix.lower() in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff']:
                try:
                    result = processor.process_document(str(file_path), category)
                    results.append(result)
                    
                    # Save annotation
                    processor.dataset_manager.save_annotation(
                        file_path.name,
                        category,
                        result
                    )
                    
                    print(f"Processed: {file_path.name}")
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Export annotations to CSV
    processor.dataset_manager.export_to_csv(
        Path(output_file).parent / 'annotations.csv'
    )
    
    return results

def evaluate_extraction(true_annotations: str, pred_annotations: str,
                       output_dir: str = 'evaluation'):
    """
    Evaluate extraction performance for all document types.
    
    Args:
        true_annotations: Path to ground truth annotations
        pred_annotations: Path to predicted annotations
        output_dir: Directory to save evaluation results
    """
    evaluator = ModelEvaluator(output_dir)
    
    # Load annotations
    with open(true_annotations) as f:
        true_data = json.load(f)
    with open(pred_annotations) as f:
        pred_data = json.load(f)
    
    # Group documents by category
    true_by_category = {}
    pred_by_category = {}
    
    for doc in true_data:
        category = doc['category']
        if category not in true_by_category:
            true_by_category[category] = []
        true_by_category[category].append(doc)
    
    for doc in pred_data:
        category = doc['category']
        if category not in pred_by_category:
            pred_by_category[category] = []
        pred_by_category[category].append(doc)
    
    # Evaluate each category
    for category in true_by_category:
        print(f"\nEvaluating {category}:")
        true_docs = true_by_category[category]
        pred_docs = pred_by_category.get(category, [])
        
        metrics = evaluator.evaluate_extraction(true_docs, pred_docs)
        
        print("\nField-level metrics:")
        for field, field_metrics in metrics['field_metrics'].items():
            print(f"\n{field}:")
            print(f"  Precision: {field_metrics['precision']:.3f}")
            print(f"  Recall: {field_metrics['recall']:.3f}")
            print(f"  F1 Score: {field_metrics['f1_score']:.3f}")
        
        print("\nMacro averages:")
        for metric, value in metrics['macro_avg'].items():
            print(f"  {metric}: {value:.3f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process documents for information extraction')
    parser.add_argument('input_dir', help='Directory containing documents to process')
    parser.add_argument('output_file', help='Path to save the extracted information')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate against ground truth annotations')
    parser.add_argument('--true-annotations', help='Path to ground truth annotations')
    
    args = parser.parse_args()
    
    # Process documents
    results = process_directory(args.input_dir, args.output_file)
    
    # Evaluate if requested
    if args.evaluate and args.true_annotations:
        evaluate_extraction(args.true_annotations, args.output_file) 