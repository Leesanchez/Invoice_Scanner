# Invoice Scanner

A Python-based document processing system that automatically extracts structured information from invoices using OCR and natural language processing techniques.

## Features

- Automatic extraction of key invoice fields:
  - Invoice number
  - Invoice date
  - Due date
  - Issuer name
  - Recipient name
  - Total amount
- Support for both text-based and image-based PDFs
- Multiple extraction methods:
  - Native PDF text extraction using pdfplumber
  - OCR using Tesseract
  - NLP-based entity extraction using spaCy
- Confidence scoring for extracted fields
- HTML report generation with visual confidence indicators
- JSON output for programmatic use

## Requirements

- Python 3.8+
- Poppler (for PDF processing)
- Tesseract (for OCR)
- Required Python packages (see requirements.txt)

## Installation

1. Install system dependencies:

```bash
# macOS (using Homebrew)
brew install poppler tesseract

# Ubuntu/Debian
sudo apt-get install poppler-utils tesseract-ocr
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Download spaCy model:

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Process a batch of invoices:

```bash
python test_invoice_batch.py
```

This will:
- Process invoices from the specified directory
- Generate an HTML report with extraction results
- Save detailed results in JSON format
- Display a summary table in the console

## Project Structure

```
.
├── src/
│   ├── preprocessing/      # Document preprocessing
│   ├── ocr/               # OCR processing
│   ├── extraction/        # Field extraction
│   ├── classification/    # Document classification
│   └── evaluation/        # Evaluation tools
├── data/                  # Sample documents
├── tests/                 # Test files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Output Formats

### HTML Report
- Visual representation of extracted fields
- Color-coded confidence levels
- Extraction method indicators

### JSON Output
- Detailed extraction results
- Confidence scores
- Extraction methods used
- Raw text content

## License

MIT License 