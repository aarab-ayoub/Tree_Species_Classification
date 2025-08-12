import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from simple_loader import get_dataset_files

def main():
    print("Tree Species Classification Project")
    print("=" * 40)

    train_data = get_dataset_files("train")
    test_data = get_dataset_files("test")
    
    print(f"Train species: {len(train_data)}")
    print(f"Test species: {len(test_data)}")
    
    total_train = sum(len(files) for files in train_data.values())
    total_test = sum(len(files) for files in test_data.values())
    
    print(f"Total train files: {total_train}")
    print(f"Total test files: {total_test}")
    
    print("\nNext: Use notebooks/data_visualization.ipynb for exploration")

if __name__ == "__main__":
    main()
