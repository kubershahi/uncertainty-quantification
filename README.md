# Uncertainty Quantification for Medical Image Registration

Utilities for converting IXI dataset pickle files to NIfTI format and visualizing image registration outputs (registered scans and deformation fields).

## Project structure

```text
.
├── assets/
│   └── images/                 # Generated visualization outputs
├── configs/
│   └── transform_to_atlas.hdf5 # Registration transform / deformation field
├── data/
│   ├── raw/                    # Original dataset (e.g., IXI)
│   ├── processed/              # Converted NIfTI outputs
│   └── metadata/               # CSV and dataset metadata
├── docs/
│   └── README_PKL_to_NII.md    # Detailed PKL->NII notes
├── models/
│   └── unigradicon1.0/         # Model weights
├── scripts/
│   ├── pkl_to_nii_converter.py
│   ├── batch_pkl_to_nii.py
│   ├── visualize_registration.py
│   ├── visualize_compare.py
│   └── visualize_deform.py
├── requirements.txt
└── README.md
```

## Quick start

### 1) Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Convert PKL to NIfTI

Single file or directory:

```bash
python scripts/pkl_to_nii_converter.py data/raw/IXI/Test -o data/processed/Test
```

Batch across dataset splits:

```bash
python scripts/batch_pkl_to_nii.py data/raw/IXI -o data/processed
```

### 3) Visualize registration

Before/after comparison:

```bash
python scripts/visualize_registration.py \
  --compare \
  --atlas data/processed/atlas_image.nii.gz \
  --subject data/processed/Test/subject_1_image.nii.gz \
  --registered assets/images/subject_1_image_registered_to_atlas.nii.gz \
  -o assets/images/before_after_registration.png
```

Deformation magnitude:

```bash
python scripts/visualize_registration.py \
  --deformation-magnitude \
  --deformation configs/transform_to_atlas.hdf5 \
  -o assets/images/deformation_magnitude.png
```

## Notes

- Run scripts from repository root for consistent relative paths.
- Large data, model files, and generated outputs are intentionally ignored by `.gitignore`.
- See `docs/README_PKL_to_NII.md` for detailed conversion guidance.

## License

MIT — see `LICENSE`.
