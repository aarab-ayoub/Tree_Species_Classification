import os

root_dir = "./train"


valid_extensions = {'.txt', '.pts', '.xyz'}

files_with_extra_columns = []

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if any(file.endswith(ext) for ext in valid_extensions):
            file_path = os.path.join(subdir, file)
            try:
                with open(file_path, 'r') as f:
                    for _ in range(5):
                        line = f.readline()
                        if not line:
                            continue
                        parts = line.strip().split()
                        if len(parts) > 3:
                            files_with_extra_columns.append(file_path)
                            break
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")

print("Files with more than 3 columns (likely xyzrgb):")
for path in files_with_extra_columns:
    print(path)

