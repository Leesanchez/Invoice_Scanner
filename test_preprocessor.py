import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent / 'src'
sys.path.append(str(src_dir))

from preprocessing.preprocessor import DocumentPreprocessor
import json

def test_single_invoice():
    preprocessor = DocumentPreprocessor()
    pdf_path = Path('data 2/Invoices/invoice_10248.pdf')
    result = preprocessor.process_pdf(pdf_path)

    print('\nProcessing:', pdf_path)
    print('\nText Content from First Page:')
    if result['text_content']:
        print(result['text_content'][0]['text'])
        print('\nTables found:', len(result['text_content'][0]['tables']))
        print('Words found:', len(result['text_content'][0]['words']))
        print('Form fields found:', len(result['text_content'][0]['form_fields']))
        
        # Save the results
        with open('test_output.json', 'w') as f:
            json.dump(result['text_content'], f, indent=2)
        print('\nFull results saved to test_output.json')

if __name__ == '__main__':
    test_single_invoice() 