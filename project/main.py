import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import time
from src.point_cloud_processor import load_point_cloud, extract_fpfh_features

def load_dataset_paths(data_path, test_csv_path):
    test_files_df = np.genfromtxt(test_csv_path, delimiter=',', dtype=str, skip_header=1)
    test_filenames = set(test_files_df[:, 0])
    X_train_paths, y_train, X_test_paths, y_test = [], [], [], []

    # Assuming 'data_path' contains the 'train' folder which has all data initially
    train_root = data_path / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"'{train_root}' directory not found. Make sure all data is inside a 'train' folder within 'data'.")

    for species_dir in train_root.iterdir():
        if not species_dir.is_dir():
            continue
        species_name = species_dir.name
        for file_path in species_dir.iterdir():
            if file_path.name in test_filenames:
                X_test_paths.append(file_path)
                y_test.append(species_name)
            else:
                X_train_paths.append(file_path)
                y_train.append(species_name)
    return X_train_paths, y_train, X_test_paths, y_test

def main():
    print("--- Starting 3D Point Cloud Classification Pipeline ---")
    start_time = time.time()
    base_path = Path(".")
    data_path = base_path / "data" # We'll put the train/test folders here
    test_csv_path = data_path / "test.csv"

    # 1. Load file paths and labels based on the provided test.csv
    print("\n[Step 1/5] Loading dataset file paths...")
    # NOTE: Your project structure has train/test folders. This function assumes all data
    # is under a single 'train' folder first, and it will split based on test.csv.
    # Adjust this if your 'train' and 'test' folders are already perfectly split.
    # For now, let's create our own split to be sure.
    X_train_paths, y_train, X_test_paths, y_test = load_dataset_paths(base_path, test_csv_path)
    print(f"Found {len(X_train_paths)} training samples and {len(X_test_paths)} testing samples.")

    # 2. Extract FPFH features for all training and testing data
    print("\n[Step 2/5] Extracting FPFH features... (This may take a while)")

    # Using a list comprehension for a compact loop
    X_train_features = [extract_fpfh_features(load_point_cloud(p)) for p in X_train_paths]
    X_test_features = [extract_fpfh_features(load_point_cloud(p)) for p in X_test_paths]

    # Convert to NumPy arrays for scikit-learn
    X_train = np.array(X_train_features)
    X_test = np.array(X_test_features)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    print("Feature extraction complete.")

    # 3. Scale the features
    # SVMs are sensitive to feature scales, so scaling is a crucial step.
    print("\n[Step 3/5] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler from training

    # 4. Train the RBF SVM Classifier
    print("\n[Step 4/5] Training RBF SVM classifier...")
    svm_classifier = SVC(kernel='rbf', C=10, gamma='auto', random_state=42, probability=True)
    svm_classifier.fit(X_train_scaled, y_train)
    print("Training complete.")

    # 5. Evaluate the model
    print("\n[Step 5/5] Evaluating model on the test set...")
    y_pred = svm_classifier.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    # Get the sorted list of unique class names for the report
    class_names = sorted(list(set(y_train)))
    print(classification_report(y_test, y_pred, target_names=class_names))

    end_time = time.time()
    print(f"--- Pipeline finished in {end_time - start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()