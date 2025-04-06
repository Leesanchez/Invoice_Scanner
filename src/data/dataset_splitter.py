import random
import shutil
from pathlib import Path
import logging
from typing import List, Dict
import json
from datetime import datetime

class DatasetSplitter:
    def __init__(self, source_dir: Path, target_dir: Path):
        """
        Initialize the dataset splitter.
        
        Args:
            source_dir: Source directory containing all documents
            target_dir: Target directory for train/test split
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.categories = ['Invoices', 'PurchaseOrders', 'Shipping orders', 'Other']
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.create_directories()
    
    def setup_logging(self):
        """Configure logging for the dataset splitter."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.target_dir / 'evaluation' / 'dataset_split.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories for train/test split."""
        for split in ['train', 'test']:
            for category in self.categories:
                dir_path = self.target_dir / 'processed' / split / category
                dir_path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
    
    def select_and_split_documents(
        self,
        samples_per_category: int = 50,  # Total samples per category
        test_split: float = 0.2,  # 20% for test
        random_seed: int = 42
    ) -> Dict:
        """
        Select documents and split into train/test sets.
        
        Args:
            samples_per_category: Total number of samples per category (default: 50)
            test_split: Fraction of documents for test set (default: 0.2)
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing split information
        """
        random.seed(random_seed)
        split_info = {
            'timestamp': datetime.now().isoformat(),
            'samples_per_category': samples_per_category,
            'test_split': test_split,
            'random_seed': random_seed,
            'categories': {},
        }
        
        for category in self.categories:
            self.logger.info(f"\nProcessing category: {category}")
            source_category_dir = self.source_dir / 'raw' / category
            
            if not source_category_dir.exists():
                self.logger.warning(f"Category directory not found: {category}")
                continue
            
            # Get all PDF files
            pdf_files = list(source_category_dir.glob('*.pdf'))
            if len(pdf_files) < samples_per_category:
                self.logger.warning(
                    f"Not enough documents in {category}. "
                    f"Found {len(pdf_files)}, needed {samples_per_category}"
                )
                continue
            
            # Randomly select documents
            selected_files = random.sample(pdf_files, samples_per_category)
            
            # Calculate split sizes
            test_size = int(samples_per_category * test_split)  # 10 for 50 samples
            train_size = samples_per_category - test_size  # 40 for 50 samples
            
            # Split into train/test
            train_files = selected_files[:train_size]
            test_files = selected_files[train_size:]
            
            # Copy files and track information
            split_info['categories'][category] = {
                'train': [],
                'test': []
            }
            
            for files, split in [(train_files, 'train'), (test_files, 'test')]:
                target_category_dir = self.target_dir / 'processed' / split / category
                
                for file in files:
                    target_path = target_category_dir / file.name
                    shutil.copy2(file, target_path)
                    split_info['categories'][category][split].append(file.name)
            
            self.logger.info(
                f"Category {category}: "
                f"Train={len(train_files)}, Test={len(test_files)}"
            )
        
        # Save split information
        split_info_path = self.target_dir / 'evaluation' / 'metrics' / 'split_info.json'
        split_info_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        self.logger.info(f"\nSplit information saved to: {split_info_path}")
        return split_info

def main():
    """Main function to run the dataset splitting process."""
    # Define paths
    source_dir = Path("docs")
    target_dir = Path("docs")
    
    # Initialize and run splitter
    splitter = DatasetSplitter(source_dir, target_dir)
    split_info = splitter.select_and_split_documents(
        samples_per_category=50,
        test_split=0.2,
        random_seed=42
    )

if __name__ == "__main__":
    main() 