"""
Scan raw IXI_2D .npy slices for corruption.

Layout: {data_dir}/Train|Val|Test|Atlas/*.npy
Edit data_dir below if needed.
"""

import os

import numpy as np

data_dir = "./data/IXI_2D/"
splits = ["Train", "Val", "Test", "Atlas"]
corrupt_files = []
total_files = 0

print("Starting scan for corrupt .npy files...")

for split in splits:
    split_path = os.path.join(data_dir, split)
    if not os.path.exists(split_path):
        print(f"Skipping {split} - directory not found.")
        continue

    files = [f for f in os.listdir(split_path) if f.endswith(".npy")]
    n_split = len(files)
    total_files += n_split
    print(f"Scanning {split}... ({n_split} .npy files)")

    for f in files:
        full_path = os.path.join(split_path, f)
        try:
            arr = np.load(full_path, allow_pickle=False)
            _ = arr.shape
        except Exception as e:
            print(f"CORRUPT: {full_path} | Error: {e}")
            corrupt_files.append(full_path)

print("-" * 30)
print(f"Total .npy files scanned: {total_files}")
if not corrupt_files:
    print("Success! No corrupt files found.")
else:
    print(f"Found {len(corrupt_files)} corrupt files.")
    with open("ixi_2d_corrupt_list.txt", "w") as log:
        for item in corrupt_files:
            log.write(f"{item}\n")
    print("List saved to 'ixi_2d_corrupt_list.txt'")
