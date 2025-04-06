from data.dataset_manager import DatasetManager
from pathlib import Path
import json
import os

def analyze_directory(directory: Path, category_name: str) -> dict:
    """Analyze contents of a specific directory."""
    stats = {
        'total_count': 0,
        'file_types': {}
    }
    
    if not directory.exists():
        return stats
        
    for file_path in directory.glob('*.*'):
        stats['total_count'] += 1
        ext = file_path.suffix.lower()
        stats['file_types'][ext] = stats['file_types'].get(ext, 0) + 1
        
    return stats

def main():
    """Analyze the dataset and print summary statistics."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    
    # Initialize paths
    data_dir = project_root / "data"
    docs_dir = project_root / "docs"
    
    print("\nDataset Analysis Summary")
    print("=" * 50)
    
    # Analyze data directory structure
    print("\nData Directory Structure:")
    print("-" * 40)
    
    data_paths = {
        'annotations': data_dir / 'annotations',
        'prepared': data_dir / 'prepared',
    }
    
    for name, path in data_paths.items():
        if path.exists():
            print(f"\n{name}:")
            for item in path.glob('*'):
                if item.is_dir():
                    print(f"  └── {item.name}/")
                else:
                    print(f"  └── {item.name} ({item.stat().st_size / 1024:.1f}KB)")
    
    # Analyze docs directory structure
    print("\nDocs Directory Structure:")
    print("-" * 40)
    
    docs_paths = {
        'raw': docs_dir / 'raw',
        'processed': docs_dir / 'processed',
        'evaluation': docs_dir / 'evaluation'
    }
    
    total_documents = 0
    file_type_distribution = {}
    categories_analysis = {}
    
    # Analyze raw documents
    if docs_paths['raw'].exists():
        for category in ['Invoices', 'PurchaseOrders', 'Shipping orders', 'Other']:
            category_dir = docs_paths['raw'] / category
            stats = analyze_directory(category_dir, category)
            
            if stats['total_count'] > 0:
                categories_analysis[category] = stats
                total_documents += stats['total_count']
                
                # Update overall file type distribution
                for ext, count in stats['file_types'].items():
                    file_type_distribution[ext] = file_type_distribution.get(ext, 0) + count
    
    # Print analysis results
    print(f"\nTotal Documents: {total_documents}")
    
    if total_documents > 0:
        print("\nDocument Distribution by Category:")
        print("-" * 40)
        for category, stats in categories_analysis.items():
            print(f"\n{category}:")
            print(f"  Total files: {stats['total_count']}")
            print("  File types:")
            for ext, count in stats['file_types'].items():
                print(f"    {ext}: {count}")
        
        print("\nOverall File Type Distribution:")
        print("-" * 40)
        for ext, count in file_type_distribution.items():
            print(f"{ext}: {count}")
    
    # Save analysis results
    analysis = {
        'total_documents': total_documents,
        'categories': categories_analysis,
        'file_type_distribution': file_type_distribution
    }
    
    output_dir = docs_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset_analysis.json"
    
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed analysis saved to: {output_file}")
    
    if total_documents == 0:
        print("\nNo documents found! Please add your documents to the following directories:")
        raw_dir = docs_paths['raw']
        print(f"- {raw_dir}/Invoices/")
        print(f"- {raw_dir}/PurchaseOrders/")
        print(f"- {raw_dir}/Shipping orders/")
        print(f"- {raw_dir}/Other/")

if __name__ == "__main__":
    main() 