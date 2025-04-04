from data.dataset_manager import DatasetManager
from pathlib import Path
import json

def main():
    """Analyze the dataset and print summary statistics."""
    # Initialize dataset manager with the data directory
    data_dir = Path("data 2")  # Use the actual data directory
    dataset_manager = DatasetManager(data_dir)
    
    # Analyze dataset
    analysis = dataset_manager.analyze_dataset()
    
    # Print summary
    print("\nDataset Analysis Summary")
    print("=" * 50)
    print(f"\nTotal Documents: {analysis['total_documents']}")
    
    print("\nDocument Distribution by Category:")
    print("-" * 40)
    for category, stats in analysis['categories'].items():
        print(f"\n{category}:")
        print(f"  Total files: {stats['total_count']}")
        print("  File types:")
        for ext, count in stats['file_types'].items():
            print(f"    {ext}: {count}")
    
    print("\nOverall File Type Distribution:")
    print("-" * 40)
    for ext, count in analysis['file_type_distribution'].items():
        print(f"{ext}: {count}")
    
    # Save detailed analysis to JSON
    output_file = data_dir / "evaluation" / "dataset_analysis.json"
    print(f"\nDetailed analysis saved to: {output_file}")

if __name__ == "__main__":
    main() 