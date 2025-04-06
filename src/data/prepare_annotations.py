import os
from pathlib import Path
import json
import logging
from typing import Dict, List
import shutil

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_annotations(annotations_dir: Path) -> Dict[str, Dict]:
    """
    Load all annotation files from the directory.
    
    Args:
        annotations_dir: Directory containing annotation JSON files
        
    Returns:
        Dictionary mapping filenames to their annotations
    """
    annotations = {}
    
    for file_path in annotations_dir.glob('*.json'):
        try:
            with open(file_path) as f:
                data = json.load(f)
                
            # Get the original filename from the annotation
            filename = data.get('filename', '')
            if filename:
                annotations[filename] = data
                
        except Exception as e:
            logging.warning(f"Error loading {file_path}: {str(e)}")
    
    return annotations

def split_dataset(
    annotations: Dict[str, Dict],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Dict[str, Dict[str, Dict]]:
    """
    Split annotations into training and test sets.
    
    Args:
        annotations: Dictionary of annotations
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing train and test splits
    """
    import random
    random.seed(seed)
    
    filenames = list(annotations.keys())
    random.shuffle(filenames)
    
    split_idx = int(len(filenames) * train_ratio)
    train_files = filenames[:split_idx]
    test_files = filenames[split_idx:]
    
    return {
        'train': {f: annotations[f] for f in train_files},
        'test': {f: annotations[f] for f in test_files}
    }

def copy_documents(
    annotations: Dict[str, Dict],
    source_dir: Path,
    target_dir: Path
) -> None:
    """
    Copy documents to train/test directories based on annotations.
    
    Args:
        annotations: Dictionary of annotations
        source_dir: Source directory containing original documents
        target_dir: Target directory for the split
    """
    for filename, anno in annotations.items():
        # Get source path from annotation
        source_path = Path(anno.get('fields', {}).get('file_path', ''))
        if not source_path.exists():
            # Try relative to source_dir
            source_path = source_dir / filename
        
        if source_path.exists():
            target_path = target_dir / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        else:
            logging.warning(f"Source document not found: {source_path}")

def prepare_dataset(
    data_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8
) -> None:
    """
    Prepare the dataset for training.
    
    Args:
        data_dir: Base data directory
        output_dir: Output directory for prepared dataset
        train_ratio: Ratio of data to use for training
    """
    logger = setup_logging()
    
    # Load annotations
    annotations_dir = data_dir / 'annotations'
    logger.info(f"Loading annotations from {annotations_dir}")
    annotations = load_annotations(annotations_dir)
    logger.info(f"Loaded {len(annotations)} annotations")
    
    # Split dataset
    splits = split_dataset(annotations, train_ratio)
    logger.info(
        f"Split dataset into {len(splits['train'])} training and "
        f"{len(splits['test'])} test samples"
    )
    
    # Create output directories
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Save split annotations
    for split_name, split_data in splits.items():
        # Save annotations
        annotations_file = output_dir / split_name / 'annotations.json'
        with open(annotations_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Copy documents
        copy_documents(
            split_data,
            data_dir / 'raw',
            output_dir / split_name / 'documents'
        )
    
    logger.info(f"Dataset prepared in {output_dir}")

def main():
    """Main function to prepare the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Base data directory')
    parser.add_argument('--output-dir', type=str, default='data/prepared',
                      help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                      help='Ratio of data to use for training')
    
    args = parser.parse_args()
    
    prepare_dataset(
        Path(args.data_dir),
        Path(args.output_dir),
        args.train_ratio
    )

if __name__ == '__main__':
    main() 