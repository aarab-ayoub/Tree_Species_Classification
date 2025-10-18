import os
import numpy as np

data_dir = "./multi_view_images/train"

classes = os.listdir(data_dir)

for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for file_name in os.listdir(class_path):
        if not file_name.endswith(".npy"):
            continue
        file_path = os.path.join(class_path, file_name)
        try:
            data = np.load(file_path)
            print(f"{class_name}/{file_name} -> shape: {data.shape}")
        except Exception as e:
            print(f"Error loading {class_name}/{file_name}: {e}")
