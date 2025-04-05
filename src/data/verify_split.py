from pathlib import Path
import json
import logging
from typing import Dict

def verify_dataset_split(data_dir: Path) -> Dict:
    """
    Verify the dataset split and return statistics.
    
    Args:
        data_dir: Path to the data directory containing train/test splits
        
    Returns:
        Dictionary containing verification statistics
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load split information
    split_info_path = data_dir / 'split_info.json'
    if not split_info_path.exists():
        raise FileNotFoundError(f"Split info file not found: {split_info_path}")
    
    with open(split_info_path) as f:
        split_info = json.load(f)
    
    # Verify files
    verification = {
        'timestamp': split_info['timestamp'],
        'categories': {}
    }
    
    for category in split_info['categories']:
        verification['categories'][category] = {
            'train': {'expected': 0, 'found': 0},
            'test': {'expected': 0, 'found': 0}
        }
        
        for split in ['train', 'test']:
            expected_files = split_info['categories'][category][split]
            split_dir = data_dir / split / category
            
            if not split_dir.exists():
                logger.warning(f"Directory not found: {split_dir}")
                continue
            
            actual_files = list(split_dir.glob('*.pdf'))
            
            verification['categories'][category][split] = {
                'expected': len(expected_files),
                'found': len(actual_files),
                'missing': [f for f in expected_files 
                           if not (split_dir / f).exists()]
            }
    
    # Log results
    logger.info("\nDataset Split Verification:")
    for category, splits in verification['categories'].items():
        logger.info(f"\n{category}:")
        for split, stats in splits.items():
            logger.info(
                f"  {split}: Expected={stats['expected']}, "
                f"Found={stats['found']}"
            )
            if stats['missing']:
                logger.warning(
                    f"  Missing files in {split}: {len(stats['missing'])}"
                )
    
    return verification

if __name__ == "__main__":
    data_dir = Path("/Users/sm_aswin21/Desktop/Anthony_Work/Invoice_Scanner/data_test")
    verify_dataset_split(data_dir) 