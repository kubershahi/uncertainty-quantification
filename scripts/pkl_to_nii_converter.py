#!/usr/bin/env python3
"""
PKL to NII Converter for IXI Dataset

Converts pickle (.pkl) files containing brain MRI images and segmentations
to NIfTI (.nii.gz) format for use with the unigradicon package.

Each PKL file contains: (image, label) tuple
- image: 3D brain MRI volume (T1-weighted)
- label: 3D subcortical segmentation
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import nibabel as nib
from nibabel.nifti1 import Nifti1Image


def pkload(fname: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a pickle file containing image and label.
    
    Args:
        fname: Path to the .pkl file
        
    Returns:
        Tuple of (image, label) numpy arrays
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)


def create_nifti_image(data: np.ndarray, affine: Optional[np.ndarray] = None) -> Nifti1Image:
    """
    Create a NIfTI image object from numpy array.
    
    Args:
        data: 3D numpy array containing the image/label data
        affine: 4x4 affine transformation matrix. If None, uses identity matrix.
        
    Returns:
        Nifti1Image object
    """
    if affine is None:
        # Use identity matrix as default affine
        affine = np.eye(4)
    
    # Ensure data is float32 for compatibility
    data = data.astype(np.float32)
    
    return Nifti1Image(data, affine)


def convert_pkl_to_nii(pkl_path: str, output_dir: str, 
                       save_image: bool = True, save_label: bool = True,
                       verbose: bool = False) -> Tuple[bool, str]:
    """
    Convert a single PKL file to NII format.
    
    Args:
        pkl_path: Path to the input .pkl file
        output_dir: Directory to save the output .nii.gz files
        save_image: Whether to save the image
        save_label: Whether to save the label
        verbose: Print progress information
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        pkl_path = Path(pkl_path)
        
        if not pkl_path.exists():
            return False, f"File not found: {pkl_path}"
        
        # Load the PKL file
        image, label = pkload(str(pkl_path))
        
        if verbose:
            print(f"Loading: {pkl_path.name}")
            print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
            print(f"  Label shape: {label.shape}, dtype: {label.dtype}")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get base filename without extension
        base_name = pkl_path.stem
        
        # Save image if requested
        if save_image:
            image_nii = create_nifti_image(image)
            image_output = output_path / f"{base_name}_image.nii.gz"
            nib.save(image_nii, str(image_output))
            if verbose:
                print(f"  Saved: {image_output.name}")
        
        # Save label if requested
        if save_label:
            label_nii = create_nifti_image(label)
            label_output = output_path / f"{base_name}_label.nii.gz"
            nib.save(label_nii, str(label_output))
            if verbose:
                print(f"  Saved: {label_output.name}")
        
        return True, f"Successfully converted {pkl_path.name}"
    
    except Exception as e:
        return False, f"Error converting {pkl_path}: {str(e)}"


def convert_directory(input_dir: str, output_dir: str, 
                     save_image: bool = True, save_label: bool = True,
                     verbose: bool = False) -> None:
    """
    Convert all PKL files in a directory to NII format.
    
    Args:
        input_dir: Directory containing .pkl files
        output_dir: Directory to save output .nii.gz files
        save_image: Whether to save the image
        save_label: Whether to save the label
        verbose: Print progress information
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return
    
    # Find all PKL files
    pkl_files = sorted(input_path.glob("*.pkl"))
    
    if not pkl_files:
        print(f"No .pkl files found in {input_dir}")
        return
    
    print(f"Found {len(pkl_files)} PKL file(s) to convert")
    
    successful = 0
    failed = 0
    
    for pkl_file in pkl_files:
        success, message = convert_pkl_to_nii(
            str(pkl_file), 
            output_dir, 
            save_image=save_image,
            save_label=save_label,
            verbose=verbose
        )
        
        if success:
            successful += 1
            print(f"✓ {message}")
        else:
            failed += 1
            print(f"✗ {message}")
    
    print(f"\nConversion complete: {successful} successful, {failed} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert IXI PKL files to NII format"
    )
    
    parser.add_argument(
        "input",
        help="Input PKL file or directory containing PKL files"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./nii_output",
        help="Output directory for NII files (default: ./nii_output)"
    )
    
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Save only image files, not labels"
    )
    
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="Save only label files, not images"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Determine whether to save image and label
    save_image = not args.label_only
    save_label = not args.image_only
    
    if args.image_only and args.label_only:
        print("Error: Cannot specify both --image-only and --label-only")
        sys.exit(1)
    
    if not input_path.exists():
        print(f"Error: Input path not found: {args.input}")
        sys.exit(1)
    
    # Process single file or directory
    if input_path.is_file():
        if input_path.suffix == ".pkl":
            success, message = convert_pkl_to_nii(
                str(input_path),
                args.output,
                save_image=save_image,
                save_label=save_label,
                verbose=args.verbose
            )
            print(message)
            sys.exit(0 if success else 1)
        else:
            print(f"Error: File is not a PKL file: {args.input}")
            sys.exit(1)
    else:
        convert_directory(
            args.input,
            args.output,
            save_image=save_image,
            save_label=save_label,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
