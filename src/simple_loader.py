"""
Simple Point Cloud Loader - Core functionality only
No complex features, just basic loading for ML pipeline
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

def load_point_cloud(file_path):
    """
    Load point cloud from .pts/.xyz/.txt file
    Returns: numpy array of shape (N, 3) with X,Y,Z coordinates
    """
    try:
        data = np.loadtxt(file_path)
        points = data[:, :3]  # Take first 3 columns (X, Y, Z)
        return points
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_point_cloud(points, title="Point Cloud"):
    """
    Simple 3D visualization of point cloud
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color by Z coordinate
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
              c=points[:, 2], cmap='viridis', s=0.5, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def get_dataset_files(split="train"):
    base_path = Path(".")
    split_path = base_path / split
    
    dataset = {}
    for species_folder in split_path.iterdir():
        if species_folder.is_dir():
            species = species_folder.name
            files = [f for f in species_folder.iterdir() if f.is_file()]
            dataset[species] = files
    
    return dataset

def load_samples(species=None, split="train", n_samples=1):
    """
    Load sample files for visualization
    """
    dataset = get_dataset_files(split)
    
    if species:
        if species in dataset:
            files = dataset[species][:n_samples]
            return [(f, load_point_cloud(f)) for f in files]
    else:
        # Load from all species
        samples = []
        for sp, files in dataset.items():
            for f in files[:n_samples]:
                points = load_point_cloud(f)
                if points is not None:
                    samples.append((sp, f.name, points))
        return samples
    
    return []

if __name__ == "__main__":
    get_dataset_files("train") 