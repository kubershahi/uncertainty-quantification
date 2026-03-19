#!/usr/bin/env python3
"""
Batch PKL to NII converter for the entire IXI dataset structure.

This script provides convenient functions to convert all IXI dataset splits
(Train, Val, Test) at once.
"""

import os
import sys
from pathlib import Path
from pkl_to_nii_converter import convert_directory


def convert_ixi_dataset(base_input_dir: str, base_output_dir: str, 
                        splits: list = None, verbose: bool = False) -> None:
    """
    Convert all PKL files in IXI dataset structure to NII format.
    
    Args:
        base_input_dir: Base directory containing IXI data (with Train/, Val/, Test/ subdirectories)
        base_output_dir: Base output directory where NII files will be saved
        splits: List of dataset splits to convert (default: ['Train', 'Val', 'Test'])
        verbose: Print progress information
    """
    if splits is None:
        splits = ['Train', 'Val', 'Test']
    
    base_input = Path(base_input_dir)
    base_output = Path(base_output_dir)
    
    if not base_input.exists():
        print(f"Error: Base input directory not found: {base_input_dir}")
        sys.exit(1)
    
    print(f"IXI Dataset PKL to NII Conversion")
    print(f"Input directory: {base_input}")
    print(f"Output directory: {base_output}")
    print(f"Splits to convert: {', '.join(splits)}")
    print("-" * 80)
    
    total_successful = 0
    total_failed = 0
    
    for split in splits:
        input_split_dir = base_input / split
        output_split_dir = base_output / split
        
        if not input_split_dir.exists():
            print(f"⚠ Skipping {split}: directory not found at {input_split_dir}")
            continue
        
        print(f"\n Converting {split} split...")
        print(f"   Input: {input_split_dir}")
        print(f"   Output: {output_split_dir}")
        
        # Count PKL files
        pkl_files = list(input_split_dir.glob("*.pkl"))
        print(f"   Found {len(pkl_files)} PKL files")
        
        if pkl_files:
            convert_directory(
                str(input_split_dir),
                str(output_split_dir),
                save_image=True,
                save_label=True,
                verbose=verbose
            )
    
    print("\n" + "=" * 80)
    print("✓ Batch conversion complete!")
    print(f"Output directory: {base_output}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Batch convert IXI dataset splits from PKL to NII format"
    )
    
    parser.add_argument(
        "input_dir",
        help="Base directory containing IXI data (e.g., data/raw/IXI with Train/, Val/, Test/ subdirectories)"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="./data/processed",
        help="Base output directory (default: ./data/processed)"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["Train", "Val", "Test"],
        help="Dataset splits to convert (default: Train Val Test)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    convert_ixi_dataset(
        args.input_dir,
        args.output,
        splits=args.splits,
        verbose=args.verbose
    )
