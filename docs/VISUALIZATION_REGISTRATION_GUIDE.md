# Visualization Registration Script Guide

Complete reference for all visualization options in `visualize_registration.py` with exhaustive usage examples.

## Command-Line Options Reference

### Input File Arguments

```
--atlas FILE              Path to template/fixed image (NIfTI .nii.gz)
--subject FILE            Path to patient/moving image (NIfTI .nii.gz)
--registered FILE         Path to registered image (NIfTI .nii.gz)
--deformation FILE        Path to deformation field (HDF5 .hdf5)
```

### Visualization Mode Flags

```
--compare                 Show atlas, subject, and registered side-by-side (3 panels)
--compare-all             Show atlas, subject, deformation, and registered (4 panels)
--overlay                 Overlay atlas and subject with transparency
--deformation-magnitude   Show displacement magnitude heatmap
--deformation-vectors     Show displacement as vector field
```

### Optional Arguments

```
--slice N                 Select specific Z-axis slice (default: middle slice)
-o, --output FILE         Save figure to PNG file
-v, --verbose             Print detailed information during execution
```

---


## Visualization Modes

### 1. **--compare** — Before/After Comparison

Shows three side-by-side images: atlas, original subject, and registered subject.

**Requires:**

- `--atlas` — Template/fixed image
- `--subject` — Original patient image
- `--registered` — Warped patient image

**Examples:**

```bash
# Basic comparison with default middle slice
python scripts/visualize_registration.py --compare \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz
```

```bash
# Compare at specific slice
python scripts/visualize_registration.py --compare \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz \
  --slice 50
```

```bash
# Save figure to file
python scripts/visualize_registration.py --compare \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz \
  --slice 45 \
  -o assets/images/comparison_subject_1.png
```

---

### 2. **--compare-all** — Complete Registration Visualization

Displays four panels: atlas, moving image, deformation magnitude, and registered image in one figure.

**Requires:**

- `--atlas` — Template/fixed image
- `--subject` — Original patient image
- `--registered` — Warped patient image
- `--deformation` — HDF5 deformation field file

**Examples:**

```bash
# Complete visualization with default middle slice
python scripts/visualize_registration.py --compare-all \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5
```

```bash
# Complete visualization at specific slice
python scripts/visualize_registration.py --compare-all \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50
```

```bash
# Save publication-ready figure
python scripts/visualize_registration.py --compare-all \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --registered data/nii/IXI/registered/subject_1/image_to_atlas.nii.gz \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 48 \
  -o assets/images/publication_figure.png
```
---

### 3. **--overlay** — Image Overlay Comparison

Overlays two images with transparency for precise alignment visualization.

**Requires:**

- `--atlas` — Template image
- `--subject` — Original patient image

**Examples:**

```bash
# Basic overlay with default middle slice
python scripts/visualize_registration.py --overlay \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz
```

```bash
# Overlay at specific slice
python scripts/visualize_registration.py --overlay \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --slice 55
```

```bash
# Save overlay comparison
python scripts/visualize_registration.py --overlay \
  --atlas data/nii/IXI/atlas_image.nii.gz \
  --subject data/nii/IXI/Test/subject_1_image.nii.gz \
  --slice 55 \
  -o assets/images/overlay_subject_1.png
```


---

### 4. **--deformation-magnitude** — Deformation Field Magnitude

Visualizes the magnitude of displacement vectors at each voxel.

**Requires:**

- `--deformation` — HDF5 deformation field file

**Examples:**

```bash
# View deformation magnitude at default middle slice
python scripts/visualize_registration.py --deformation-magnitude \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5
```

```bash
# Deformation magnitude at specific slice
python scripts/visualize_registration.py --deformation-magnitude \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50
```

```bash
# Save deformation magnitude heatmap
python scripts/visualize_registration.py --deformation-magnitude \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50 \
  -o assets/images/deformation_magnitude_slice_50.png
```

```bash
# Verbose output to see deformation field statistics
python scripts/visualize_registration.py --deformation-magnitude \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50 \
  -v
```

---

### 5. **--deformation-vectors** — Deformation Vector Field

Displays deformation as 2D vectors (arrows) showing direction and magnitude of displacement.

**Requires:**

- `--deformation` — HDF5 deformation field file

**Examples:**

```bash
# View deformation vectors at default middle slice
python scripts/visualize_registration.py --deformation-vectors \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5
```

```bash
# Deformation vectors at specific slice
python scripts/visualize_registration.py --deformation-vectors \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50
```

```bash
# Save vector field visualization
python scripts/visualize_registration.py --deformation-vectors \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50 \
  -o assets/images/deformation_vectors_slice_50.png
```

```bash
# Verbose output for vector field information
python scripts/visualize_registration.py --deformation-vectors \
  --deformation data/nii/IXI/registered/subject_1/transform_to_atlas.hdf5 \
  --slice 50 \
  -v
```

---

