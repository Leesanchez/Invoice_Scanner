import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import shutil

class DatasetManager:
    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the dataset manager.
        
        Args:
            base_dir: Base directory for the dataset
        """
        self.base_dir = Path(base_dir)
        self.categories = [
            'Invoices',
            'Monthly',
            'MonthlyCategory',
            'PurchaseOrders',
            'Shipping orders'
        ]
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        # Create category directories if they don't exist
        for category in self.categories:
            (self.base_dir / category).mkdir(parents=True, exist_ok=True)
            
        # Create directories for annotations and processed data
        (self.base_dir / 'annotations').mkdir(exist_ok=True)
        (self.base_dir / 'processed').mkdir(exist_ok=True)
        (self.base_dir / 'evaluation').mkdir(exist_ok=True)
    
    def add_document(self, file_path: Union[str, Path], category: str):
        """
        Add a document to the dataset.
        
        Args:
            file_path: Path to the document
            category: Document category
        """
        if category not in self.categories:
            raise ValueError(f"Invalid category. Must be one of {self.categories}")
            
        dest_dir = self.base_dir / category
        shutil.copy2(file_path, dest_dir)
    
    def save_annotation(self, filename: str, category: str, 
                       fields: Optional[Dict] = None):
        """
        Save document annotation.
        
        Args:
            filename: Document filename
            category: Document category
            fields: Dictionary of extracted fields (for invoices)
        """
        annotation = {
            'filename': filename,
            'category': category,
            'fields': fields or {}
        }
        
        # Save to JSON file
        annotation_file = self.base_dir / 'annotations' / f"{Path(filename).stem}.json"
        with open(annotation_file, 'w') as f:
            json.dump(annotation, f, indent=2)
    
    def load_annotations(self) -> pd.DataFrame:
        """
        Load all annotations into a DataFrame.
        
        Returns:
            DataFrame containing all annotations
        """
        annotations = []
        for json_file in (self.base_dir / 'annotations').glob('*.json'):
            with open(json_file) as f:
                annotation = json.load(f)
                annotations.append(annotation)
        
        return pd.DataFrame(annotations)
    
    def get_document_path(self, filename: str, category: str) -> Path:
        """
        Get the full path to a document.
        
        Args:
            filename: Document filename
            category: Document category
            
        Returns:
            Path to the document
        """
        return self.base_dir / category / filename
    
    def get_dataset_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get detailed dataset statistics.
        
        Returns:
            Dictionary with document counts and file types per category
        """
        stats = {}
        for category in self.categories:
            path = self.base_dir / category
            if not path.exists():
                continue
                
            files = list(path.glob('*'))
            file_types = {}
            for file in files:
                if file.is_file() and not file.name.startswith('.'):
                    ext = file.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            stats[category] = {
                'total_count': len(files),
                'file_types': file_types
            }
        
        return stats
    
    def export_to_csv(self, output_file: Union[str, Path]):
        """
        Export annotations to CSV file.
        
        Args:
            output_file: Path to save the CSV file
        """
        df = self.load_annotations()
        df.to_csv(output_file, index=False)
    
    def analyze_dataset(self) -> Dict[str, Any]:
        """
        Perform detailed analysis of the dataset.
        
        Returns:
            Dictionary containing dataset analysis results
        """
        stats = self.get_dataset_stats()
        
        # Get total document count
        total_docs = sum(cat_stats['total_count'] for cat_stats in stats.values())
        
        # Get file type distribution across all categories
        all_file_types = {}
        for cat_stats in stats.values():
            for ext, count in cat_stats['file_types'].items():
                all_file_types[ext] = all_file_types.get(ext, 0) + count
        
        analysis = {
            'total_documents': total_docs,
            'categories': stats,
            'file_type_distribution': all_file_types
        }
        
        # Save analysis to JSON
        analysis_file = self.base_dir / 'evaluation' / 'dataset_analysis.json'
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis 