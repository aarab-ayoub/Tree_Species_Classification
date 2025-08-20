import numpy as np
import open3d as o3d
from pathlib import Path
import os

def load_point_cloud(file_path):
    file_path_str = str(file_path)
    pcd = o3d.geometry.PointCloud()

    try:

        if file_path_str.endswith(('.pts', '.txt')):
            points = np.loadtxt(file_path_str)
            if points.ndim == 1: 
                points = points.reshape(1, -1)
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # For other standard formats like .xyz, Open3D's reader is fast and efficient.
        else:
            temp_pcd = o3d.io.read_point_cloud(file_path_str)
            if temp_pcd.has_points():
                pcd.points = temp_pcd.points
        if not pcd.has_points():
            return None
            
        return pcd
        
    except Exception as e:
        print(f"CRITICAL ERROR loading {Path(file_path_str).name}: {e}")
        return None


def visualize_point_cloud(pcd, title="Point Cloud Visualization"):
    """
    Opens an interactive 3D viewer for the point cloud.
    Controls:
    - Mouse drag: Rotate
    - Mouse wheel: Zoom
    - Right-click drag: Pan
    """
    if pcd is None:
        print("Cannot visualize an empty point cloud.")
        return
    
    # This single line creates a rich, interactive window.
    o3d.visualization.draw_geometries([pcd], window_name=title)

# --- Feature Extraction Function (Your Next Step) ---

def extract_fpfh_features(pcd, voxel_size=0.1):
    """
    Calculates a single global feature descriptor for a point cloud using FPFH.
    
    Returns: A 1D NumPy array representing the aggregated features of the tree.
    """
    # 1. Downsample the point cloud. FPFH is computationally expensive.
    #    A voxel grid creates a uniform density, which is good for feature descriptors.
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # 2. Estimate normals. FPFH relies on the normals of the points.
    #    The search radius determines which neighbors are used to calculate the normal.
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 3. Compute FPFH features.
    #    The search radius for FPFH should be larger than the normal estimation radius.
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    # 4. Aggregate features. FPFH gives a feature vector for EACH point.
    #    For classifying the whole tree, we need one single vector.
    #    A common approach is to take the mean of all point features.
    #    The .T (transpose) is needed because of the shape Open3D returns.
    aggregated_features = np.mean(pcd_fpfh.data.T, axis=0)
    
    return aggregated_features

# --- Main execution block for testing ---

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parent.parent

    sample_file_path = base_path / "train" / "Ash" / "E1.xyz"
    
    if not sample_file_path.exists():
        print(f"Sample file not found at: {sample_file_path}")
        print("Please update the path to a valid file in your dataset.")
    else:
        print(f"--- Loading sample file: {sample_file_path.name} ---")
        point_cloud = load_point_cloud(sample_file_path)

        if point_cloud:
            # Show the number of points loaded
            num_points = np.asarray(point_cloud.points).shape[0]
            print(f"Successfully loaded {num_points:,} points.")

            # Test the visualization
            print("Opening interactive 3D viewer. Close the window to continue.")
            visualize_point_cloud(point_cloud, title=f"Ash - {sample_file_path.name}")
            
            # Test the feature extraction
            print("\n--- Extracting FPFH features ---")
            # Using a larger voxel size for dense clouds like this might be faster
            fpfh_vector = extract_fpfh_features(point_cloud, voxel_size=0.2) 
            
            if fpfh_vector is not None:
                print(f"Successfully extracted FPFH feature vector.")
                print(f"Feature vector shape: {fpfh_vector.shape}") # Should be (33,)
                print(f"Feature vector (first 5 values): {fpfh_vector[:5]}")
            else:
                print("Failed to extract features.")
        else:
            print("Failed to load point cloud.")