import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from typing import Dict, List

# Add src directory to Python path
sys.path.append('src')

from preprocessing.preprocessor import DocumentPreprocessor
from ocr.ocr_engine import OCREngine
from extraction.field_extractor import FieldExtractor

def process_invoice(pdf_path: Path, preprocessor, ocr_engine, field_extractor) -> Dict:
    """Process a single invoice and extract key fields."""
    # Preprocess document
    processed_doc = preprocessor.process_pdf(pdf_path)
    
    # Get native text extraction results
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
    
    # Extract fields
    fields = field_extractor.extract_fields(
        native_text,
        native_boxes,
        'Invoices'
    )
    
    # Map extracted fields to required fields
    result = {
        'filename': pdf_path.name,
        'invoice_number': fields.get('invoice_number', fields.get('order_id')),
        'invoice_date': fields.get('order_date'),
        'due_date': fields.get('due_date'),
        'issuer_name': fields.get('from_organization'),
        'recipient_name': fields.get('customer_name', fields.get('to_organization')),
        'total_amount': fields.get('total_amount', fields.get('amount')),
        'raw_text': native_text
    }
    
    # Convert ExtractedField objects to dict format
    for key, value in result.items():
        if hasattr(value, '__dict__'):
            result[key] = {
                'value': value.value,
                'confidence': value.confidence,
                'method': value.method
            }
    
    return result

def create_html_report(results: List[Dict], output_file: str):
    """Create a simple HTML report of the extracted information."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Invoice Processing Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .invoice {{ 
                border: 1px solid #ddd; 
                margin: 10px 0; 
                padding: 15px;
                border-radius: 5px;
            }}
            .field {{ margin: 5px 0; }}
            .field-name {{ font-weight: bold; }}
            .confidence {{
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.8em;
                margin-left: 5px;
            }}
            .high {{ background-color: #dff0d8; }}
            .medium {{ background-color: #fcf8e3; }}
            .low {{ background-color: #f2dede; }}
            .method {{ font-style: italic; color: #666; }}
        </style>
    </head>
    <body>
        <h1>Invoice Processing Results</h1>
        <p>Processed at: {timestamp}</p>
    """
    
    for result in results:
        html_content += f"""
        <div class="invoice">
            <h2>Invoice: {result['filename']}</h2>
        """
        
        for field in ['invoice_number', 'invoice_date', 'due_date', 'issuer_name', 'recipient_name', 'total_amount']:
            if field in result and isinstance(result[field], dict):
                confidence = result[field]['confidence']
                confidence_class = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.5 else 'low'
                field_name = field.replace('_', ' ').title()
                value = result[field]['value']
                method = result[field]['method']
                confidence_pct = f"{confidence:.0%}"
                
                html_content += f"""
                <div class="field">
                    <span class="field-name">{field_name}:</span> {value}
                    <span class="confidence {confidence_class}">{confidence_pct}</span>
                    <span class="method">({method})</span>
                </div>
                """
        
        html_content += "</div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    # Initialize components
    preprocessor = DocumentPreprocessor()
    ocr_engine = OCREngine()
    field_extractor = FieldExtractor()
    
    # Get list of invoice files
    invoice_dir = Path('data 2/Invoices')
    invoice_files = sorted(list(invoice_dir.glob('invoice_*.pdf')))[:5]  # Process first 5 invoices
    
    print(f"\nProcessing {len(invoice_files)} invoices...")
    
    # Process each invoice
    results = []
    for pdf_path in invoice_files:
        print(f"\nProcessing: {pdf_path.name}")
        result = process_invoice(pdf_path, preprocessor, ocr_engine, field_extractor)
        results.append(result)
    
    # Save detailed results as JSON
    with open('invoice_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create HTML report
    create_html_report(results, 'invoice_report.html')
    
    # Create summary DataFrame
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Invoice Number': r['invoice_number']['value'] if isinstance(r['invoice_number'], dict) else None,
            'Invoice Date': r['invoice_date']['value'] if isinstance(r['invoice_date'], dict) else None,
            'Total Amount': r['total_amount']['value'] if isinstance(r['total_amount'], dict) else None,
            'Recipient': r['recipient_name']['value'] if isinstance(r['recipient_name'], dict) else None
        }
        for r in results
    ])
    
    print("\nSummary of processed invoices:")
    print(df.to_string())
    print("\nDetailed results saved to invoice_results.json")
    print("HTML report generated as invoice_report.html")

if __name__ == '__main__':
    main() 