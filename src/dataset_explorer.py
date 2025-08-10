"""
DATASET_EXPLORER.PY - Core Data Handling Module
===============================================
 PURPOSE: Load, analyze, and explore your tree point cloud dataset
🔧 MAIN FUNCTIONS:
   - load_point_cloud(): Load .pts/.xyz/.txt files
   - show_dataset_statistics(): Show train/test split info
   - create_dataset_loader(): Batch load files for ML pipeline
   - SimplePointCloud: Point cloud data structure

 USAGE: Called by main.py and other scripts for data operations
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class SimplePointCloud:
    """A simple point cloud class to mimic some Open3D functionality"""
    
    def __init__(self, points=None):
        self.points = points if points is not None else np.array([])
        self.colors = None
        self.normals = None
    
    def get_center(self):
        """Get the center of the point cloud"""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.mean(self.points, axis=0)
    
    def get_axis_aligned_bounding_box(self):
        """Get axis-aligned bounding box"""
        if len(self.points) == 0:
            return None
        
        min_bound = np.min(self.points, axis=0)
        max_bound = np.max(self.points, axis=0)
        return min_bound, max_bound
    
    def normalize(self):
        """Normalize point cloud to unit cube"""
        if len(self.points) == 0:
            return self
        
        center = self.get_center()
        self.points = self.points - center
        
        max_extent = np.max(np.abs(self.points))
        if max_extent > 0:
            self.points = self.points / max_extent
        
        return self
    
    def downsample_uniform(self, every_k_points):
        """Downsample by taking every k-th point"""
        if len(self.points) > 0:
            self.points = self.points[::every_k_points]
            if self.colors is not None:
                self.colors = self.colors[::every_k_points]
        return self

def load_point_cloud(file_path):
    """
    Load point cloud data from various file formats (.xyz, .pts, .txt)
    Returns a SimplePointCloud object
    """
    try:
        # For all file types, try to load as space-separated values
        data = np.loadtxt(file_path)
        
        if len(data) == 0:
            print(f"  Error: No data loaded from {file_path}")
            return None
        
        # Take first 3 columns as X, Y, Z
        points = data[:, :3]
        
        # Create point cloud object
        pcd = SimplePointCloud(points)
        
        # If there are more than 3 columns, they might be colors
        if data.shape[1] >= 6:
            colors = data[:, 3:6]
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            pcd.colors = colors
        
        return pcd

    except Exception as e:
        print(f"  Error loading file {file_path}: {e}")
        return None

def get_dataset_info(base_path="./"):
    """
    Get comprehensive information about the train/test dataset
    """
    train_path = Path(base_path) / "train"
    test_path = Path(base_path) / "test"
    
    if not train_path.exists() or not test_path.exists():
        print(" Train/test folders not found! Please run organize_dataset.py first.")
        return None, None
    
    train_info = {}
    test_info = {}
    
    # Scan train folder
    for species_folder in train_path.iterdir():
        if species_folder.is_dir():
            species = species_folder.name
            files = [f for f in species_folder.iterdir() if f.is_file()]
            train_info[species] = {
                'files': [f.name for f in files],
                'count': len(files),
                'formats': list(set([f.suffix for f in files]))
            }
    
    # Scan test folder
    for species_folder in test_path.iterdir():
        if species_folder.is_dir():
            species = species_folder.name
            files = [f for f in species_folder.iterdir() if f.is_file()]
            test_info[species] = {
                'files': [f.name for f in files],
                'count': len(files),
                'formats': list(set([f.suffix for f in files]))
            }
    
    return train_info, test_info

def visualize_dataset_samples(base_path="./", samples_per_species=2):
    """
    Visualize sample point clouds from each species in both train and test sets
    """
    train_info, test_info = get_dataset_info(base_path)
    
    if train_info is None:
        return
    
    print(" Dataset Sample Visualization")
    print("=" * 50)
    
    for species in sorted(train_info.keys()):
        print(f"\n--- {species} ---")
        
        train_path = Path(base_path) / "train" / species
        test_path = Path(base_path) / "test" / species
        
        # Show train samples
        train_files = train_info[species]['files'][:samples_per_species]
        print(f" Train samples ({len(train_info[species]['files'])} total):")
        
        for i, filename in enumerate(train_files):
            file_path = train_path / filename
            print(f"  {i+1}. Loading {filename}...")
            
            pcd = load_point_cloud(file_path)
            if pcd is not None:
                print(f"      {len(pcd.points):,} points")
                
                # Downsample for visualization
                if len(pcd.points) > 20000:
                    pcd.downsample_uniform(len(pcd.points) // 20000)
                
                # Quick visualization
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                          c=pcd.points[:, 2], cmap='viridis', s=0.5, alpha=0.6)
                ax.set_title(f"{species} - Train Sample: {filename}")
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                plt.tight_layout()
                plt.show()
        
        # Show test samples
        test_files = test_info[species]['files'][:samples_per_species]
        print(f" Test samples ({len(test_info[species]['files'])} total):")
        
        for i, filename in enumerate(test_files):
            file_path = test_path / filename
            print(f"  {i+1}. Loading {filename}...")
            
            pcd = load_point_cloud(file_path)
            if pcd is not None:
                print(f"      {len(pcd.points):,} points")

def create_dataset_loader(base_path="./"):
    """
    Create a dataset loader function that can iterate through train/test data
    """
    
    def load_species_files(split="train", species=None, max_files=None):
        """
        Load point cloud files for specific species or all species
        
        Args:
            split: "train" or "test"
            species: Specific species name or None for all
            max_files: Maximum number of files to load per species
        
        Returns:
            List of (file_path, species_name, point_cloud) tuples
        """
        results = []
        split_path = Path(base_path) / split
        
        if not split_path.exists():
            print(f" {split} folder not found!")
            return results
        
        species_folders = [species] if species else [f.name for f in split_path.iterdir() if f.is_dir()]
        
        for species_name in species_folders:
            species_path = split_path / species_name
            if not species_path.exists():
                continue
                
            files = [f for f in species_path.iterdir() if f.is_file()]
            if max_files:
                files = files[:max_files]
            
            print(f"Loading {species_name}: {len(files)} files from {split}")
            
            for file_path in files:
                pcd = load_point_cloud(file_path)
                if pcd is not None:
                    results.append((str(file_path), species_name, pcd))
                    print(f"   {file_path.name}: {len(pcd.points):,} points")
                else:
                    print(f"   Failed: {file_path.name}")
        
        return results
    
    return load_species_files

def show_dataset_statistics(base_path="./"):
    """
    Show comprehensive dataset statistics
    """
    train_info, test_info = get_dataset_info(base_path)
    
    if train_info is None:
        return
    
    print(" Dataset Statistics")
    print("=" * 60)
    
    print(f"{'Species':<15} {'Train':<8} {'Test':<8} {'Total':<8} {'Formats':<15}")
    print("-" * 60)
    
    total_train = 0
    total_test = 0
    
    for species in sorted(train_info.keys()):
        train_count = train_info[species]['count']
        test_count = test_info.get(species, {'count': 0})['count']
        total_count = train_count + test_count
        
        # Get unique formats across train and test
        train_formats = set(train_info[species]['formats'])
        test_formats = set(test_info.get(species, {'formats': []})['formats'])
        all_formats = sorted(train_formats | test_formats)
        formats_str = ', '.join(all_formats)
        
        print(f"{species:<15} {train_count:<8} {test_count:<8} {total_count:<8} {formats_str:<15}")
        total_train += train_count
        total_test += test_count
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_train:<8} {total_test:<8} {total_train + total_test:<8}")
    
    if total_train + total_test > 0:
        test_percentage = (total_test / (total_train + total_test)) * 100
        print(f"\n Split ratio: {100-test_percentage:.1f}% train, {test_percentage:.1f}% test")
        print(f"  Total files: {total_train + total_test:,}")
        print(f" Species count: {len(train_info)}")

if __name__ == "__main__":
    print(" Tree Species Dataset Explorer")
    print("=" * 50)
    
    # Show dataset statistics
    show_dataset_statistics()
    
    print(f"\n" + "=" * 50)
    print("Usage Examples:")
    print("=" * 50)
    
    print("""
# Load dataset
loader = create_dataset_loader()

# Load all training data
train_data = loader(split="train", max_files=5)  # limit for demo

# Load specific species test data  
ash_test = loader(split="test", species="Ash")

# Load samples for visualization
visualize_dataset_samples(samples_per_species=1)
""")
    
    # Create and demonstrate the loader
    print(f"\n Demonstration - Loading Sample Data:")
    loader = create_dataset_loader()
    
    # Load a few samples from each species
    sample_data = loader(split="train", max_files=2)
    print(f"\n Loaded {len(sample_data)} sample files successfully!")
    
    print(f"\n Your dataset is ready for machine learning!")
    print(f"   Next steps: Feature extraction (LBP + FPFH) and SVM classification")
