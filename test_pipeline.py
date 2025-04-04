import sys
from pathlib import Path

# Add src directory to Python path
sys.path.append('src')

from preprocessing.preprocessor import DocumentPreprocessor
from ocr.ocr_engine import OCREngine
from extraction.field_extractor import FieldExtractor
import json

def test_pipeline():
    # Initialize components
    preprocessor = DocumentPreprocessor()
    ocr_engine = OCREngine()
    field_extractor = FieldExtractor()
    
    # Process a sample invoice
    pdf_path = Path('data 2/Invoices/invoice_10248.pdf')
    
    print(f"\nProcessing: {pdf_path}")
    
    # Step 1: Preprocess the document
    print("\nStep 1: Preprocessing...")
    processed_doc = preprocessor.process_pdf(pdf_path)
    
    # Step 2: Extract text using OCR for comparison
    print("\nStep 2: OCR Processing...")
    ocr_results = []
    for img in processed_doc['images']:
        ocr_result = ocr_engine.extract_text(img)
        ocr_results.extend(ocr_result)
    
    # Step 3: Extract fields using both native text and OCR results
    print("\nStep 3: Field Extraction...")
    
    # Extract from native text
    native_text = processed_doc['text_content'][0]['text']
    native_boxes = [
        {
            'x': word['x0'],
            'y': word['top'],
            'width': word['width'],
            'height': word['height']
        }
        for word in processed_doc['text_content'][0]['words']
    ]
    
    native_fields = field_extractor.extract_fields(
        native_text,
        native_boxes,
        'Invoices'
    )
    
    # Extract from OCR text
    ocr_text = ' '.join(result.text for result in ocr_results)
    ocr_boxes = [result.bounding_box for result in ocr_results]
    
    ocr_fields = field_extractor.extract_fields(
        ocr_text,
        ocr_boxes,
        'Invoices'
    )
    
    # Combine results
    results = {
        'document_info': {
            'filename': pdf_path.name,
            'page_count': len(processed_doc['images']),
            'tables_found': len(processed_doc['text_content'][0]['tables'])
        },
        'native_text_extraction': {
            'text': native_text,
            'extracted_fields': {
                field: {
                    'value': info.value,
                    'confidence': info.confidence,
                    'method': info.method
                }
                for field, info in native_fields.items()
            }
        },
        'ocr_extraction': {
            'text': ocr_text,
            'extracted_fields': {
                field: {
                    'value': info.value,
                    'confidence': info.confidence,
                    'method': info.method
                }
                for field, info in ocr_fields.items()
            }
        }
    }
    
    # Save results
    with open('pipeline_test_output.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nResults Summary:")
    print(f"Document: {results['document_info']['filename']}")
    print(f"Pages: {results['document_info']['page_count']}")
    print(f"Tables found: {results['document_info']['tables_found']}")
    
    print("\nFields extracted from native text:")
    for field, info in results['native_text_extraction']['extracted_fields'].items():
        print(f"  {field}: {info['value']} (confidence: {info['confidence']:.2f}, method: {info['method']})")
    
    print("\nFields extracted from OCR:")
    for field, info in results['ocr_extraction']['extracted_fields'].items():
        print(f"  {field}: {info['value']} (confidence: {info['confidence']:.2f}, method: {info['method']})")
    
    print("\nDetailed results saved to pipeline_test_output.json")

if __name__ == '__main__':
    test_pipeline() 