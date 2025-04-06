# Invoice Scanner

A comprehensive document processing system that extracts information from various types of documents (invoices, purchase orders, shipping orders, etc.) using OCR, machine learning, and information extraction techniques.

## System Overview

The system consists of several main components:

1. **Document Classification**: Identifies the type of document (invoice, purchase order, etc.)
2. **OCR Processing**: Extracts text from documents
3. **Field Extraction**: Extracts specific fields from documents (invoice numbers, dates, amounts, etc.)
4. **Data Management**: Handles dataset organization and processing
5. **Evaluation**: Measures system performance

## Main Components and Their Functions

### 1. Document Preprocessing (`preprocessing/preprocessor.py`)
- Converts PDFs to images
- Enhances image quality for better OCR
- Handles document skew correction and noise removal

### 2. OCR Engine (`ocr/ocr_engine.py`)
- Uses Tesseract OCR to extract text from images
- Provides text confidence scores and bounding boxes
- Handles different document layouts

### 3. Document Classification (`classification/document_classifier.py`)
- Uses multiple models (SVM, Random Forest, Gradient Boosting, LSTM, Transformer)
- Classifies documents into categories (Invoices, Purchase Orders, etc.)
- Provides confidence scores for classifications

### 4. Field Extraction (`extraction/field_extractor.py` and `extraction/cutie_extractor.py`)
- Extracts specific fields from documents using multiple methods:
  - Regex patterns
  - Named Entity Recognition (NER)
  - Position-based heuristics
  - CUTIE (Custom Unified Text Information Extractor) model

### 5. Dataset Management (`data/dataset_manager.py`)
- Organizes and manages document datasets
- Handles train/test splits
- Manages annotations and metadata

### 6. Evaluation (`evaluation/evaluator.py`)
- Measures system performance
- Generates metrics and reports
- Creates confusion matrices and classification reports

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Invoice_Scanner
```

2. Create and activate a virtual environment:
```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Document Dataset Setup

**Note:** This repository does not include document datasets due to size constraints. To use this system:

1. Create the required directory structure:
   ```
   data/
   ├── annotations/       # For document annotations
   └── prepared/          # For prepared datasets
       ├── train/
       └── test/
   
   docs/
   ├── raw/
   │   ├── Invoices/
   │   ├── PurchaseOrders/
   │   └── Shipping orders/
   ├── processed/
   └── evaluation/
   ```

2. Add your PDF documents to the appropriate directories in `docs/raw/`

3. Create the required empty directories:
   ```bash
   mkdir -p data/annotations data/prepared/train data/prepared/test
   mkdir -p docs/raw/Invoices docs/raw/PurchaseOrders "docs/raw/Shipping orders"
   mkdir -p docs/processed docs/evaluation
   ```

4. Run the dataset analysis to verify your setup:
   ```bash
   python src/analyze_dataset.py
   ```

5. For testing purposes, you can use freely available invoice datasets from:
   - [Sample Invoices Dataset](https://github.com/invoice-x/invoice-corpus)
   - [Public Invoices](https://www.kaggle.com/datasets/anurag629/invoice-dataset)

## How to Run

### 1. Analyze Dataset

First, analyze your dataset to understand its structure:

```bash
python src/analyze_dataset.py
```

This will:
- Show the structure of your data directory
- Display document counts per category (invoices, purchase orders, etc.)
- Provide file type distribution
- Save a detailed JSON report to `docs/evaluation/dataset_analysis.json`

### 2. Preprocess Documents

Preprocess your documents to prepare them for classification and information extraction:

```bash
python src/preprocess_documents.py --input-dir docs/raw --output-dir docs/processed
```

This will:
- Convert PDFs to images
- Apply image enhancement techniques
- Extract text using OCR
- Save processed files as images and JSON text files

### 3. Train Classification Models

Train the document classification models:

```bash
python src/data/train_models.py
```

This will:
- Load the training data
- Train multiple classification models:
  - SVM
  - Random Forest
  - Gradient Boosting
  - LSTM (deep learning)
  - Transformer (BERT-based)
- Evaluate model performance
- Save trained models to the `models/` directory
- Generate performance metrics and confusion matrices

### 4. Extract Invoice Information

Extract information from processed documents:

```bash
python src/extract_invoice_info.py --input-dir docs/processed --output-file docs/evaluation/extracted_info.json
```

This will:
- Process all documents in the input directory
- Classify each document
- For documents classified as invoices:
  - Extract key fields:
    - Invoice number
    - Date
    - Due date
    - Total amount
    - Issuer name
    - Recipient name
- Save results in a structured JSON format

### 5. Evaluate Performance

Evaluate the system's performance:

```bash
python src/evaluation/evaluator.py
```

This will:
- Calculate precision, recall, and F1-score for document classification
- Measure field extraction accuracy
- Generate detailed reports and visualizations
- Save results to `docs/evaluation/`

## Data Flow

The data flows through the system as follows:

1. **Raw Documents** (`docs/raw/`) → Documents organized by category
2. **Preprocessing** → Converts to images and extracts text (`docs/processed/`)
3. **Classification** → Identifies document types
4. **Field Extraction** → Extracts relevant fields from invoices
5. **Results** → Structured data in JSON/CSV format (`docs/evaluation/`)

## Components in Detail

### Document Classification

The system uses multiple classification models:
- **Random Forest**: High accuracy for document type classification
- **Gradient Boosting**: Provides robust classification with confidence scores
- **SVM**: Alternative classifier with good performance
- **LSTM/Transformer**: Deep learning models for complex documents

### Field Extraction

For invoices, the system extracts fields using:
- **Regex Patterns**: For structured fields like invoice numbers and dates
- **Named Entity Recognition**: For company names and addresses
- **Position-based Heuristics**: For fields that appear in predictable locations

### Performance

The system achieves:
- High classification accuracy across document types
- Reliable field extraction for standard invoice formats
- Detailed confidence scores for each extracted field

## Processing Stages

1. **Document Input**
   - System accepts PDF documents
   - Documents are organized by category

2. **Preprocessing**
   - PDFs are converted to images
   - Images are enhanced for better OCR
   - Skew correction and noise removal are applied

3. **OCR Processing**
   - Text is extracted from images
   - Bounding boxes and confidence scores are generated
   - Text is organized by layout

4. **Document Classification**
   - Documents are classified into categories
   - Multiple models are used for robust classification
   - Confidence scores are generated

5. **Field Extraction**
   - For invoices, specific fields are extracted:
     - Invoice number
     - Date
     - Due date
     - Total amount
     - Issuer name
     - Recipient name
   - Multiple extraction methods are used and results are combined

6. **Output Generation**
   - Results are saved in multiple formats:
     - JSON files with detailed extraction results
     - CSV files with structured data
     - Evaluation metrics and reports

## Performance

Based on the evaluation results:
- Random Forest and Gradient Boosting models achieve 100% accuracy
- SVM model achieves ~81.6% accuracy
- The system performs well across all document categories

## Dataset Statistics

- Total documents: 2,537
- Categories:
  - Invoices: 898 documents
  - Purchase Orders: 830 documents
  - Shipping Orders: 809 documents
- Most documents are in PDF format

## Project Structure

```
src/
├── classification/     # Document classification models
├── data/              # Dataset management
├── evaluation/        # Performance evaluation
├── extraction/        # Field extraction
├── ocr/              # OCR processing
├── preprocessing/     # Document preprocessing
├── analyze_dataset.py
├── extract_invoice_info.py
├── extract_to_dataframe.py
├── main.py
└── preprocess_documents.py
```

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Document Dataset

This repository does not include the document dataset due to size constraints. To use this system:

1. Create the following directory structure:
   ```
   docs/raw/Invoices/
   docs/raw/PurchaseOrders/
   docs/raw/Shipping orders/
   ```

2. Add your PDF documents to the appropriate directories

3. Run the dataset analysis to verify your setup:
   ```bash
   python src/analyze_dataset.py
   ``` 