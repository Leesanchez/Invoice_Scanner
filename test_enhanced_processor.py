import sys
from pathlib import Path
import json
import pandas as pd
from typing import Dict, List

# Add src directory to Python path
sys.path.append('src')

from extraction.enhanced_processor import EnhancedDocumentProcessor

def create_html_report(results: List[Dict], output_file: str):
    """Create an HTML report comparing traditional and enhanced extraction results."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Document Processing Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .document { 
                border: 1px solid #ddd; 
                margin: 10px 0; 
                padding: 15px;
                border-radius: 5px;
            }
            .field { margin: 5px 0; }
            .field-name { font-weight: bold; }
            .confidence {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 3px;
                font-size: 0.8em;
                margin-left: 5px;
            }
            .high { background-color: #dff0d8; }
            .medium { background-color: #fcf8e3; }
            .low { background-color: #f2dede; }
            .method { font-style: italic; color: #666; }
            .comparison {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Enhanced Document Processing Results</h1>
    """
    
    for result in results:
        html += f"""
        <div class="document">
            <h2>Document: {result['filename']}</h2>
            <div class="comparison">
                <div>
                    <h3>Traditional Extraction</h3>
        """
        
        # Add traditional results
        for field_name, field_info in result['traditional_results'].items():
            confidence = field_info['confidence']
            confidence_class = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.5 else 'low'
            
            html += f"""
            <div class="field">
                <span class="field-name">{field_name}:</span> {field_info['value']}
                <span class="confidence {confidence_class}">{confidence:.0%}</span>
                <span class="method">({field_info['method']})</span>
            </div>
            """
        
        html += """
                </div>
                <div>
                    <h3>Enhanced Extraction</h3>
        """
        
        # Add enhanced results
        for field_name, field_info in result['enhanced_results'].items():
            confidence = field_info['confidence']
            confidence_class = 'high' if confidence >= 0.8 else 'medium' if confidence >= 0.5 else 'low'
            
            html += f"""
            <div class="field">
                <span class="field-name">{field_name}:</span> {field_info['value']}
                <span class="confidence {confidence_class}">{confidence:.0%}</span>
                <span class="method">({field_info['method']})</span>
            </div>
            """
        
        html += """
                </div>
            </div>
        </div>
        """
    
    html += """
    </body>
    </html>
    """
    
    with open(output_file, 'w') as f:
        f.write(html)

def main():
    # Initialize processors
    enhanced_processor = EnhancedDocumentProcessor()
    
    # Get list of invoice files
    invoice_dir = Path('data 2/Invoices')
    invoice_files = sorted(list(invoice_dir.glob('invoice_*.pdf')))[:5]  # Process first 5 invoices
    
    print(f"\nProcessing {len(invoice_files)} invoices with enhanced processor...")
    
    # Process each invoice
    results = []
    for pdf_path in invoice_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Process with enhanced processor
        enhanced_results = enhanced_processor.process_document(pdf_path)
        
        result = {
            'filename': pdf_path.name,
            'traditional_results': {
                k: v for k, v in enhanced_results.items()
                if 'layoutlm' not in v['method']
            },
            'enhanced_results': enhanced_results
        }
        
        results.append(result)
    
    # Save detailed results as JSON
    with open('enhanced_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create HTML report
    create_html_report(results, 'enhanced_report.html')
    
    # Create summary DataFrame
    df = pd.DataFrame([
        {
            'Filename': r['filename'],
            'Traditional Confidence': sum(v['confidence'] for v in r['traditional_results'].values()) / len(r['traditional_results']),
            'Enhanced Confidence': sum(v['confidence'] for v in r['enhanced_results'].values()) / len(r['enhanced_results']),
            'Fields Found (Traditional)': len(r['traditional_results']),
            'Fields Found (Enhanced)': len(r['enhanced_results'])
        }
        for r in results
    ])
    
    print("\nSummary of processed invoices:")
    print(df.to_string())
    print("\nDetailed results saved to enhanced_results.json")
    print("HTML report generated as enhanced_report.html")

if __name__ == '__main__':
    main() 