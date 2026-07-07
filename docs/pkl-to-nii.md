# PKL to NII Converter for IXI Dataset

This script converts IXI dataset files from pickle (.pkl) format to NIfTI (.nii.gz) format

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

Or install directly:

```bash
pip install nibabel numpy
```

## What the Script Does

Each PKL file in the IXI dataset contains a tuple of:

- **image**: 3D T1-weighted brain MRI volume (normalized to [0,1])
- **label**: 3D subcortical segmentation

The converter reads these PKL files and saves them as separate NII files:

- `{filename}_image.nii.gz` - The brain MRI volume
- `{filename}_label.nii.gz` - The segmentation label

## Usage

### Convert a Single File

```bash
python scripts/pkl_to_nii_converter.py path/to/file.pkl -o output_directory/
```

### Convert All Files in a Directory

```bash
python scripts/pkl_to_nii_converter.py path/to/pkl_directory/ -o output_directory/
```

### Convert Image Only (No Labels)

```bash
python scripts/pkl_to_nii_converter.py path/to/pkl_directory/ -o output_directory/ --image-only
```

### Convert Labels Only (No Images)

```bash
python scripts/pkl_to_nii_converter.py path/to/pkl_directory/ -o output_directory/ --label-only
```

### Verbose Output

For detailed progress information:

```bash
python scripts/pkl_to_nii_converter.py path/to/pkl_directory/ -o output_directory/ -v
```

## Examples

### Convert IXI Test Dataset

```bash
python scripts/pkl_to_nii_converter.py data/raw/IXI/Test/ -o data/processed/Test/ -v
```

### Convert Entire IXI Dataset

```bash
# Convert training data
python scripts/pkl_to_nii_converter.py data/raw/IXI/Train/ -o data/processed/Train/ -v

# Convert validation data
python scripts/pkl_to_nii_converter.py data/raw/IXI/Val/ -o data/processed/Val/ -v

# Convert test data
python scripts/pkl_to_nii_converter.py data/raw/IXI/Test/ -o data/processed/Test/ -v
```

## Command Line Arguments

```
positional arguments:
  input                 Input PKL file or directory containing PKL files

optional arguments:
  -h, --help            show this help message and exit
  -o, --output OUTPUT   Output directory for NII files (default: ./nii_output)
  --image-only          Save only image files, not labels
  --label-only          Save only label files, not images
  -v, --verbose         Print detailed progress information
```

## Output Structure

When converting a directory, the output will have the same structure:

```
Input:  data/raw/IXI/Train/image_001.pkl
Output: data/processed/Train/image_001_image.nii.gz
Output: data/processed/Train/image_001_label.nii.gz
```

## Notes

- All output files are saved as gzipped NIfTI format (.nii.gz)
- Identity affine transformation matrix is used (you can modify the script if you need specific affine transformations)
- Data is automatically converted to float32 for compatibility
- The script creates output directories if they don't exist
