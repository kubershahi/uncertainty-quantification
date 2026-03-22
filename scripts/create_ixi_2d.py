import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def create_2d_dataset(base_path, output_path, atlas_path, slices_per_volume=10):
    splits = ['Train', 'Val', 'Test']
    
    for split in splits:
        print(f"Processing {split} split...")
        split_src = Path(base_path) / split
        split_dst = Path(output_path) / split
        split_dst.mkdir(parents=True, exist_ok=True)
        
        # Get all pkl files
        pkl_files = list(split_src.glob("*.pkl"))
        
        for pkl_path in tqdm(pkl_files):
            # Load 3D data: image shape is (160, 192, 224)
            image, label = pkload(pkl_path)
            
            # Select middle axial slices (Axis 2 is 224)
            # 107 to 117 gives you the core structural anatomy
            mid = image.shape[2] // 2
            start = mid - (slices_per_volume // 2)
            end = mid + (slices_per_volume // 2)
            
            for i in range(start, end):
                slice_2d = image[:, :, i]
                # label_2d = label[:, :, i] # Keep if you need Dice scores later
                
                # Save as .npy to preserve [0, 1] float precision
                file_name = f"{pkl_path.stem}_slice_{i}.npy"
                np.save(split_dst / file_name, slice_2d)

    # Load atlas image and label
    atlas_image, atlas_label = pkload(atlas_path)

    # Select middle axial slices (Axis 2 is 224)
    # 107 to 117 gives you the core structural anatomy
    mid = atlas_image.shape[2] // 2
    start = mid - (slices_per_volume // 2)
    end = mid + (slices_per_volume // 2)

    atlas_dst = Path(output_path) / "Atlas"
    atlas_dst.mkdir(parents=True, exist_ok=True)
    for i in range(start, end):
        slice_2d = atlas_image[:, :, i]
        file_name = f"atlas_slice_{i}.npy"
        np.save(atlas_dst / file_name, slice_2d)
    
if __name__ == "__main__":
    SOURCE_DIR = "./data/raw/IXI/"
    TARGET_DIR = "./data/raw/IXI_2D/"
    ATLAS_PATH = "./data/raw/IXI/atlas.pkl"
    
    create_2d_dataset(SOURCE_DIR, TARGET_DIR, ATLAS_PATH, slices_per_volume=10)
    print("Done! 2D dataset is ready.")