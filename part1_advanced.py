"""
PART1_ADVANCED.PY - Visualization & Analysis Tool
=================================================
🎯 PURPOSE: Advanced point cloud visualization and analysis
🔧 MAIN FEATURES:
   - Multi-view 3D visualization (XY, XZ, YZ projections)
   - Point cloud preprocessing (normalize, downsample, outlier removal)
   - Statistical analysis and bounding box calculations
   - Interactive plotting with matplotlib

📊 USAGE: Run directly for data exploration: python part1_advanced.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
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
    
    def get_max_bound(self):
        """Get maximum bound of the point cloud"""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.max(self.points, axis=0)
    
    def get_min_bound(self):
        """Get minimum bound of the point cloud"""
        if len(self.points) == 0:
            return np.array([0, 0, 0])
        return np.min(self.points, axis=0)
    
    def translate(self, translation):
        """Translate the point cloud"""
        if len(self.points) > 0:
            self.points += translation
        return self
    
    def scale(self, scale_factor):
        """Scale the point cloud"""
        if len(self.points) > 0:
            self.points *= scale_factor
        return self
    
    def normalize(self):
        """Normalize point cloud to unit cube"""
        if len(self.points) == 0:
            return self
        
        center = self.get_center()
        self.translate(-center)
        
        max_extent = np.max(np.abs(self.points))
        if max_extent > 0:
            self.scale(1.0 / max_extent)
        
        return self
    
    def downsample_uniform(self, every_k_points):
        """Downsample by taking every k-th point"""
        if len(self.points) > 0:
            self.points = self.points[::every_k_points]
            if self.colors is not None:
                self.colors = self.colors[::every_k_points]
        return self
    
    def remove_statistical_outliers(self, nb_neighbors=20, std_ratio=2.0):
        """Remove statistical outliers using distance-based approach"""
        if len(self.points) < nb_neighbors:
            return self
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=nb_neighbors, algorithm='ball_tree').fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)
        
        # Calculate mean distance for each point (excluding itself)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        
        # Calculate global statistics
        global_mean = np.mean(mean_distances)
        global_std = np.std(mean_distances)
        
        # Filter outliers
        threshold = global_mean + std_ratio * global_std
        inlier_mask = mean_distances < threshold
        
        self.points = self.points[inlier_mask]
        if self.colors is not None:
            self.colors = self.colors[inlier_mask]
        
        print(f"  Removed {np.sum(~inlier_mask)} outliers, kept {np.sum(inlier_mask)} points")
        return self

def load_point_cloud(file_path):
    """
    Load point cloud data from various file formats (.xyz, .pts, .txt)
    Returns a SimplePointCloud object
    """
    print(f"Attempting to load: {file_path}")
    
    try:
        # For all file types, try to load as space-separated values
        print(f"  Loading file with numpy...")
        data = np.loadtxt(file_path)
        
        if len(data) == 0:
            print(f"  Error: No data loaded from {file_path}")
            return None
        
        # Take first 3 columns as X, Y, Z
        points = data[:, :3]
        
        print(f"  Successfully loaded {len(points)} points")
        print(f"  Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        # Create point cloud object
        pcd = SimplePointCloud(points)
        
        # If there are more than 3 columns, they might be colors or normals
        if data.shape[1] >= 6:
            # Assume columns 3-5 are RGB colors
            colors = data[:, 3:6]
            # Normalize colors if they seem to be in 0-255 range
            if np.max(colors) > 1.0:
                colors = colors / 255.0
            pcd.colors = colors
            print(f"  Found color information")
        
        return pcd

    except Exception as e:
        print(f"  Error loading file {file_path}: {e}")
        return None

def visualize_point_cloud_advanced(pcd, title="Point Cloud", point_size=1):
    """
    Advanced visualization with multiple views and options
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Main 3D view
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Use colors if available, otherwise color by Z coordinate
    if pcd.colors is not None:
        colors = pcd.colors
    else:
        colors = pcd.points[:, 2]  # Color by Z coordinate
    
    scatter = ax1.scatter(pcd.points[:, 0], pcd.points[:, 1], pcd.points[:, 2], 
                         c=colors, cmap='viridis', s=point_size, alpha=0.6)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{title} - 3D View')
    
    # XY projection
    ax2 = fig.add_subplot(222)
    ax2.scatter(pcd.points[:, 0], pcd.points[:, 1], 
               c=colors, cmap='viridis', s=point_size, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection (Top View)')
    ax2.set_aspect('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(223)
    ax3.scatter(pcd.points[:, 0], pcd.points[:, 2], 
               c=colors, cmap='viridis', s=point_size, alpha=0.6)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection (Front View)')
    ax3.set_aspect('equal')
    
    # YZ projection
    ax4 = fig.add_subplot(224)
    ax4.scatter(pcd.points[:, 1], pcd.points[:, 2], 
               c=colors, cmap='viridis', s=point_size, alpha=0.6)
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.set_title('YZ Projection (Side View)')
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_point_cloud(pcd, name="Point Cloud"):
    """
    Perform basic analysis on the point cloud
    """
    print(f"=== Analysis of {name} ===")
    print(f"Number of points: {len(pcd.points):,}")
    
    center = pcd.get_center()
    print(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    min_bound, max_bound = pcd.get_axis_aligned_bounding_box()
    extents = max_bound - min_bound
    print(f"Bounding box:")
    print(f"  Min: ({min_bound[0]:.2f}, {min_bound[1]:.2f}, {min_bound[2]:.2f})")
    print(f"  Max: ({max_bound[0]:.2f}, {max_bound[1]:.2f}, {max_bound[2]:.2f})")
    print(f"  Extents: ({extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f})")
    
    # Point density estimation
    volume = np.prod(extents)
    if volume > 0:
        density = len(pcd.points) / volume
        print(f"Point density: {density:.2f} points/unit³")
    
    # Basic statistics
    print(f"Point statistics:")
    print(f"  X: mean={np.mean(pcd.points[:, 0]):.2f}, std={np.std(pcd.points[:, 0]):.2f}")
    print(f"  Y: mean={np.mean(pcd.points[:, 1]):.2f}, std={np.std(pcd.points[:, 1]):.2f}")
    print(f"  Z: mean={np.mean(pcd.points[:, 2]):.2f}, std={np.std(pcd.points[:, 2]):.2f}")
    print()

# --- Example Usage ---
base_path = './'

# List of sample files to test the loader
sample_files = {
    "Ash (.pts)": os.path.join(base_path, "Ash", "123.pts"),
    "Ash (.xyz)": os.path.join(base_path, "Ash", "E1.xyz"),
    "Beech (.pts)": os.path.join(base_path, "Beech", "102.pts"),
    "Beech (.xyz)": os.path.join(base_path, "Beech", "Bu1.xyz"),
    "Douglas Fir (.txt)": os.path.join(base_path, "Douglas Fir", "31_11.txt"),
    "Douglas Fir (.xyz)": os.path.join(base_path, "Douglas Fir", "Tree1_Border.xyz"),
}

def test_advanced_loader():
    """Test the advanced point cloud loader with analysis and visualization"""
    print("=== Testing Advanced Point Cloud Loader ===")
    print()
    
    successful_loads = 0
    total_files = len(sample_files)
    
    for species, file_path in sample_files.items():
        print(f"--- Testing {species} ---")
        if os.path.exists(file_path):
            pcd = load_point_cloud(file_path)
            
            if pcd is not None:
                print(f"✓ Successfully loaded {species}")
                
                # Perform analysis
                analyze_point_cloud(pcd, species)
                
                # Downsample large point clouds for faster visualization
                original_size = len(pcd.points)
                if original_size > 50000:
                    downsample_factor = max(1, original_size // 50000)
                    pcd.downsample_uniform(downsample_factor)
                    print(f"  Downsampled from {original_size:,} to {len(pcd.points):,} points for visualization")
                
                print(f"  Visualizing {species}... (Close the plot window to continue)")
                visualize_point_cloud_advanced(pcd, species, point_size=0.5)
                successful_loads += 1
            else:
                print(f"✗ Failed to load {species}")
        else:
            print(f"✗ File not found: {file_path}")
        print()
    
    print(f"=== Summary ===")
    print(f"Successfully loaded {successful_loads}/{total_files} files")
    if successful_loads == total_files:
        print("🎉 All files loaded successfully! Your advanced loader is working correctly.")
        print("\nYou now have a solid foundation for:")
        print("• Loading tree point cloud data in multiple formats")
        print("• Visualizing 3D point clouds from multiple angles")
        print("• Analyzing point cloud properties")
        print("• Preprocessing (downsampling, outlier removal)")
        print("\nNext steps could include:")
        print("• Implementing 3D descriptors (FPFH)")
        print("• Computing 2D projections for LBP features")
        print("• Setting up classification pipeline with SVM")
    else:
        print("⚠️  Some files failed to load. Check the error messages above.")

if __name__ == "__main__":
    test_advanced_loader()
