#!/usr/bin/env python3
"""
Visualization script for medical image registration results.

Visualizes:
1. Registered images (warped moving image)
2. Deformation fields from HDF5 files
3. Before/after registration comparison

Usage:
python visualize_registration.py --atlas path/to/atlas.nii.gz \
                                 --subject path/to/subject.nii.gz \
                                 --registered path/to/registered.nii.gz \
                                 --deformation path/to/deformation.h5 \
                                 --compare-all \
                                 --slice 90 \
                                 -o assets/images/registration_comparison.png
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Tuple, Optional


def find_datasets(group, path=""):
    """Recursively find all datasets in HDF5 group."""
    datasets = {}
    for key in group.keys():
        current_path = f"{path}/{key}" if path else key
        if isinstance(group[key], h5py.Dataset):
            datasets[current_path] = group[key]
        elif isinstance(group[key], h5py.Group):
            datasets.update(find_datasets(group[key], current_path))
    return datasets


def load_nifti(filepath: str) -> np.ndarray:
    """Load NIfTI file and return data as numpy array."""
    img = nib.load(filepath)
    return img.get_fdata()


def visualize_nifti_slice(filepath: str, slice_idx: int, title: str, 
                         output_file: str, figsize: Tuple[int, int] = (8, 8)) -> None:
    """
    Visualize a single 2D slice from a 3D NIfTI image.
    
    Args:
        filepath: Path to NII file
        slice_idx: Z-axis slice index (default: middle slice)
        title: Plot title
        output_file: Save image to file
        figsize: Figure size
    """
    data = load_nifti(filepath)
    
    if slice_idx is None:
        slice_idx = data.shape[2] // 2  # Middle slice
    
    if title is None:
        title = f"Image - Slice {slice_idx}"
    
    plt.figure(figsize=figsize)
    plt.imshow(data[:, :, slice_idx], cmap='gray')
    plt.title(title, fontsize=14)
    plt.colorbar(label='Intensity')
    plt.axis('off')
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    
    plt.show()





def visualize_deformation_field(hdf5_filepath: str, slice_idx: int = None, 
                               output_file: str = None) -> None:
    """
    Visualize deformation field magnitude from HDF5 file.
    
    Args:
        hdf5_filepath: Path to HDF5 deformation file
        slice_idx: Z-axis slice index (default: middle slice)
        output_file: Save image to file
    """
    try:
        with h5py.File(hdf5_filepath, 'r') as f:
            print(f"HDF5 keys: {list(f.keys())}")
            
            # Find all datasets recursively
            all_datasets = find_datasets(f)
            
            # Look for deformation field
            field = None
            field_name = None
            
            # First, try to find TransformGroup/2/TransformParameters (the deformation field)
            for path, dataset in all_datasets.items():
                if 'TransformGroup/2/TransformParameters' in path:
                    transform_params = dataset[:]
                    field_name = path
                    n_params = len(transform_params)
                    
                    print(f"Found TransformParameters: shape={dataset.shape}")
                    
                    # Try to reshape to 3D vector field
                    if n_params % 3 == 0:
                        total_voxels = n_params // 3
                        cube_size = round(total_voxels ** (1/3))
                        if cube_size ** 3 == total_voxels:
                            field = transform_params.reshape((cube_size, cube_size, cube_size, 3))
                            print(f"✓ Reshaped to: {field.shape}")
                            break
            
            if field is None:
                print("Error: Could not find or reshape deformation field")
                return
            
            print(f"Deformation field shape: {field.shape}")
            print(f"Min: {np.min(field)}, Max: {np.max(field)}")
            
            # Determine slice
            if slice_idx is None:
                slice_idx = field.shape[2] // 2
            
            # Calculate displacement magnitude
            if len(field.shape) == 4 and field.shape[3] == 3:
                displacement_mag = np.linalg.norm(field[:, :, slice_idx, :], axis=-1)
            else:
                displacement_mag = np.abs(field[:, :, slice_idx])
            
            print(f"Displacement magnitude - Min: {np.min(displacement_mag)}, Max: {np.max(displacement_mag)}")
            
            # Visualize
            plt.figure(figsize=(10, 8))
            im = plt.imshow(displacement_mag, cmap='hot')
            plt.title(f'Deformation Magnitude - Slice {slice_idx}', fontsize=14)
            cbar = plt.colorbar(im, label='Displacement (voxels)')
            plt.axis('off')
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {output_file}")
            
            plt.show()
    
    except Exception as e:
        print(f"Error reading HDF5 file: {e}")
        import traceback
        traceback.print_exc()


def visualize_deformation_vectors(hdf5_filepath: str, slice_idx: int, 
                                 skip: int = 5, output_file: str = None) -> None:
    """
    Visualize deformation field as vector field.
    
    Args:
        hdf5_filepath: Path to HDF5 deformation file
        slice_idx: Z-axis slice index (default: middle slice)
        skip: Plot every Nth vector for clarity
        output_file: Save image to file
    """
    try:
        with h5py.File(hdf5_filepath, 'r') as f:
            # Find all datasets recursively
            all_datasets = find_datasets(f)
            
            field = None
            
            # Find TransformGroup/2/TransformParameters
            for path, dataset in all_datasets.items():
                if 'TransformGroup/2/TransformParameters' in path:
                    transform_params = dataset[:]
                    n_params = len(transform_params)
                    
                    print(f"Found TransformParameters: shape={dataset.shape}")
                    
                    # Try to reshape to 3D vector field
                    if n_params % 3 == 0:
                        total_voxels = n_params // 3
                        cube_size = round(total_voxels ** (1/3))
                        if cube_size ** 3 == total_voxels:
                            field = transform_params.reshape((cube_size, cube_size, cube_size, 3))
                            print(f"✓ Reshaped to: {field.shape}")
                            break
            
            if field is None:
                print("Error: Could not find deformation field")
                return
            
            if slice_idx is None:
                slice_idx = field.shape[2] // 2
            
            # Extract 2D slice
            if len(field.shape) == 4 and field.shape[3] == 3:
                dx = field[:, :, slice_idx, 0]
                dy = field[:, :, slice_idx, 1]
            else:
                print("Error: Expected 3D displacement vectors")
                return
            
            # Create vector field plot
            plt.figure(figsize=(12, 10))
            
            y, x = np.mgrid[0:dx.shape[0]:skip, 0:dx.shape[1]:skip]
            u = dx[::skip, ::skip]
            v = dy[::skip, ::skip]
            
            # Plot displacement magnitude as background
            displacement_mag = np.sqrt(dx**2 + dy**2)
            plt.imshow(displacement_mag, cmap='gray', alpha=0.6)
            
            # Plot vectors
            plt.quiver(x, y, u, v, displacement_mag[::skip, ::skip], 
                      cmap='jet', scale=50, scale_units='inches')
            
            plt.title(f'Deformation Vector Field - Slice {slice_idx}', fontsize=14)
            plt.colorbar(label='Displacement Magnitude')
            plt.axis('off')
            
            if output_file:
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {output_file}")
            
            plt.show()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def compare_before_after(fixed_file: str, moving_file: str, registered_file: str,
                        slice_idx: int, output_file: str) -> None:
    """
    Compare atlas, original subject, and registered subject side-by-side.
    
    Args:
        fixed_file: Path to atlas/fixed image
        moving_file: Path to original subject/moving image
        registered_file: Path to registered subject image
        slice_idx: Z-axis slice index (default: middle slice)
        output_file: Save image to file
    """
    atlas = load_nifti(fixed_file)
    subject = load_nifti(moving_file)
    registered = load_nifti(registered_file)
    
    if slice_idx is None:
        slice_idx = atlas.shape[2] // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(atlas[:, :, slice_idx], cmap='gray')
    axes[0].set_title('Atlas (Fixed)', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(subject[:, :, slice_idx], cmap='gray')
    axes[1].set_title('Subject Before Registration', fontsize=12)
    axes[1].axis('off')
    
    axes[2].imshow(registered[:, :, slice_idx], cmap='gray')
    axes[2].set_title('Subject After Registration', fontsize=12)
    axes[2].axis('off')
    
    plt.suptitle(f'Registration Comparison - Slice {slice_idx}', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    
    plt.show()


def compare_complete_registration(fixed_file: str, moving_file: str, registered_file: str,
                                 deformation_file: str, slice_idx: int, 
                                 output_file: str) -> None:
    """
    Complete registration comparison with all 4 visualizations.
    
    Shows: Atlas | Moving | Deformation Magnitude | Registered
    
    Args:
        fixed_file: Path to atlas/fixed image
        moving_file: Path to original subject/moving image
        registered_file: Path to registered subject image
        deformation_file: Path to deformation field (HDF5 file)
        slice_idx: Z-axis slice index (default: middle slice)
        output_file: Save image to file
    """
    try:
        # Load NII files
        atlas = load_nifti(fixed_file)
        subject = load_nifti(moving_file)
        registered = load_nifti(registered_file)
        
        if slice_idx is None:
            slice_idx = atlas.shape[2] // 2
        
        # Extract deformation field
        with h5py.File(deformation_file, 'r') as f:
            all_datasets = find_datasets(f)
            field = None
            
            for path, dataset in all_datasets.items():
                if 'TransformGroup/2/TransformParameters' in path:
                    transform_params = dataset[:]
                    n_params = len(transform_params)
                    
                    if n_params % 3 == 0:
                        total_voxels = n_params // 3
                        cube_size = round(total_voxels ** (1/3))
                        if cube_size ** 3 == total_voxels:
                            field = transform_params.reshape((cube_size, cube_size, cube_size, 3))
                            break
            
            if field is None:
                print("Error: Could not extract deformation field")
                return
            
            # Calculate displacement magnitude
            displacement_mag = np.linalg.norm(field[:, :, slice_idx, :], axis=-1)
        
        # Create 4-panel visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Panel 1: Atlas
        im1 = axes[0].imshow(atlas[:, :, slice_idx], cmap='gray')
        axes[0].set_title('Atlas (Fixed)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Panel 2: Moving (Before Registration)
        im2 = axes[1].imshow(subject[:, :, slice_idx], cmap='gray')
        axes[1].set_title('Subject\n(Before Registration)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Panel 3: Deformation Magnitude
        im3 = axes[2].imshow(displacement_mag, cmap='hot')
        axes[2].set_title('Deformation\nMagnitude', fontsize=12, fontweight='bold')
        cbar3 = plt.colorbar(im3, ax=axes[2], label='Displacement (voxels)')
        axes[2].axis('off')
        
        # Panel 4: Registered
        im4 = axes[3].imshow(registered[:, :, slice_idx], cmap='gray')
        axes[3].set_title('Subject\n(After Registration)', fontsize=12, fontweight='bold')
        axes[3].axis('off')
        
        plt.suptitle(f'Complete Registration Visualization - Slice {slice_idx}', 
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"✓ Saved: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def overlay_images(fixed_file: str, moving_file: str, slice_idx: int,
                  output_file: str) -> None:
    """
    Overlay two images with transparency for comparison.
    
    Args:
        fixed_file: Path to fixed/atlas image
        moving_file: Path to moving/subject image
        slice_idx: Z-axis slice index (default: middle slice)
        output_file: Save image to file
    """
    fixed = load_nifti(fixed_file)
    moving = load_nifti(moving_file)
    
    if slice_idx is None:
        slice_idx = fixed.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before registration
    axes[0].imshow(fixed[:, :, slice_idx], cmap='gray', alpha=0.7, label='Atlas')
    axes[0].imshow(moving[:, :, slice_idx], cmap='Reds', alpha=0.3, label='Subject')
    axes[0].set_title('Before Registration (Atlas + Subject Overlay)', fontsize=12)
    axes[0].axis('off')
    axes[0].legend(loc='upper right')
    
    # After registration
    axes[1].imshow(fixed[:, :, slice_idx], cmap='gray', alpha=0.7, label='Atlas')
    axes[1].imshow(moving[:, :, slice_idx], cmap='Reds', alpha=0.3, label='Registered Subject')
    axes[1].set_title('After Registration (Atlas + Subject Overlay)', fontsize=12)
    axes[1].axis('off')
    axes[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize medical image registration results"
    )
    
    # Input files
    parser.add_argument('--atlas', help='Path to atlas/fixed image (NII file)')
    parser.add_argument('--subject', help='Path to subject/moving image (NII file)')
    parser.add_argument('--registered', help='Path to registered image (NII file)')
    parser.add_argument('--deformation', help='Path to deformation field (HDF5 file)')
    
    # Visualization modes
    parser.add_argument('--compare', action='store_true',
                       help='Show before/after comparison (requires --atlas, --subject, --registered)')
    parser.add_argument('--compare-all', action='store_true',
                       help='Show all 4: atlas, moving, deformation, registered (requires --atlas, --subject, --registered, --deformation)')
    parser.add_argument('--overlay', action='store_true',
                       help='Show overlay comparison (requires --atlas, --subject)')
    parser.add_argument('--deformation-magnitude', action='store_true',
                       help='Visualize deformation field magnitude')
    parser.add_argument('--deformation-vectors', action='store_true',
                       help='Visualize deformation field as vectors')
    
    # Options
    parser.add_argument('--slice', type=int, default=None,
                       help='Z-axis slice to display (default: middle slice)')
    parser.add_argument('-o', '--output', help='Save figure to file')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Default behavior: compare all three images
    if not any([args.compare, args.compare_all, args.overlay, args.deformation_magnitude, 
                args.deformation_vectors]):
        args.compare = True
    
    # Visualization logic
    if args.compare_all:
        if not all([args.atlas, args.subject, args.registered, args.deformation]):
            print("Error: --compare-all requires --atlas, --subject, --registered, and --deformation")
            return
        print("Generating complete registration comparison...")
        compare_complete_registration(args.atlas, args.subject, args.registered, args.deformation,
                                     slice_idx=args.slice, output_file=args.output)
    
    elif args.compare:
        if not all([args.atlas, args.subject, args.registered]):
            print("Error: --compare requires --atlas, --subject, and --registered")
            return
        print("Generating before/after comparison...")
        compare_before_after(args.atlas, args.subject, args.registered, 
                           slice_idx=args.slice, output_file=args.output)
    
    elif args.overlay:
        if not all([args.atlas, args.subject]):
            print("Error: --overlay requires --atlas and --subject")
            return
        print("Generating overlay visualization...")
        overlay_images(args.atlas, args.subject, 
                      slice_idx=args.slice, output_file=args.output)
    
    elif args.deformation_magnitude:
        if not args.deformation:
            print("Error: --deformation-magnitude requires --deformation")
            return
        print("Visualizing deformation magnitude...")
        visualize_deformation_field(args.deformation, 
                                   slice_idx=args.slice, output_file=args.output)
    
    elif args.deformation_vectors:
        if not args.deformation:
            print("Error: --deformation-vectors requires --deformation")
            return
        print("Visualizing deformation vectors...")
        visualize_deformation_vectors(args.deformation, 
                                     slice_idx=args.slice, output_file=args.output)
    
    if args.verbose:
        print("Visualization complete!")


if __name__ == "__main__":
    main()
