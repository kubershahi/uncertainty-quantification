# UniGrad Synth experiment — data, error-map U-Net, and runs

End-to-end notes for **synthetic registration** and error-map U-Net training. This repo has two
synth-related tracks:

| Track | Dim | Phase I script | Output |
| --- | ---: | --- | --- |
| **IXI 2D** (legacy) | 2D | `create_synth_data.py` (superseded for HCP) | `*_triplet.npz` |
| **HCP 3D** (current) | 3D | `create_synth_data.py` | `datasets/hcp_synth/<split>/*.npz` |

For 3D IO on IXI see `docs/unigrad-io-experiment.md`. For HCP download/QC see `docs/hcp-dataset.md`.
For registration vocabulary see `docs/registration-concepts.md`.

Artifact roots:

| Role | Path |
| --- | --- |
| IXI 2D slices | `data/IXI_2D/` (or `datasets/IXI_2D/`) |
| IXI synth triplets | `data/IXI_2D_synth_trip/` |
| IXI UniGrad fivers | `data/IXI_2D_unigrad_synth_fiver/` |
| HCP T1w (real) | `datasets/hcp/` |
| **HCP 3D synth** | `datasets/hcp_synth/{Train,Val,Test}/` |
| Data scripts | `experiments/unigrad-synth/` |
| Train / eval (2D) | `experiments/regression/unigrad-synth/` |
| Run outputs | `assets/runs/2d/unigrad-synth/error_unet_run{N}/` |
| QC figures | `assets/images/unigrad-synth/`, `assets/images/synth/` |

---

## Pipeline (IXI 2D — legacy)

1. **Phase I — Triplets** — TorchIO affine + elastic → `*_triplet.npz` (`image`, `warped`, `phi`, …).
2. **Phase II — Fivers** — `create_unigrad_synth_data.py`: UniGradICON → `phi_pred`, `error_map`.
3. **Phase III — Train** — `train_unigrad_synth_unet.py`: 4-channel `UNet2D` → scalar `error_map`.
4. **Eval / QC** — `eval_unigrad_synth_unet.py`, `visualize_synth_data.py`, `visualize_unigrad_data.py`.

Model inputs: normalized **image**, **warped**, **`phi_pred / phi_scale`**.  
Target: **`error_map`** (pixels). Loss on **`valid_mask`** where applicable.

---

## HCP 3D synthetic data (`create_synth_data.py`)

Phase I for the **HCP branch**: one synthetic registration pair per subject, 3D volumes in native
LAS voxel grid.

### Input / output layout

**Input** (per subject):

```text
datasets/hcp/<subject_id>/T1w/
  T1w_acpc_dc_restore_brain.nii.gz
  brainmask_fs.nii.gz
```

**Output:**

```text
datasets/hcp_synth/{Train,Val,Test}/<subject_id>_<suffix>.npz
```

Suffixes: `none`, `rig`, `aff`, `ela`, `aela` (see deformation classes below).

### NPZ schema

| Key | Shape / type | Meaning |
| --- | --- | --- |
| `source` | `(X, Y, Z)` float32 | Fixed image; **masked z-score** (μ, σ from source brain mask) |
| `moving` | `(X, Y, Z)` float32 | Warped image; same μ, σ as `source` |
| `u` | `(3, X, Y, Z)` float32 | Ground-truth displacement; **voxel units** (`u_unit="vox"`) |
| `mask` | `(X, Y, Z)` bool | `brainmask_fs` |
| `source_affine` | `(4, 4)` float32 | NIfTI voxel → world (mm) |
| `source_spacing` | `(3,)` float32 | Voxel size in mm |
| `u_unit` | str | `"vox"` |
| `deformation_class` | str | See below |
| `subject_id` | str | HCP ID |
| `qc_passed` | bool | QC on raw warp (see below) |

**Naming:** we store **`u`** (displacement), not φ. Position map would be φ = identity + u.

### Split and deformation balance

- **Split:** deterministic 70 / 15 / 15 by subject hash (`Train` / `Val` / `Test`). One sample per
  subject (no multi-warp reuse).
- **Deformation mix** (same ratios in every split): 5% none, 20% rigid, 25% affine, 25%
  elastic, 25% affine+elastic (`affine_elastic`).

Manifest: `datasets/hcp_synth/split_manifest.json`.

### Deformation classes and file nomenclature

Each HCP subject gets **one** synthetic sample. The warp type is stored in NPZ as `deformation_class`
and encoded in the filename suffix: `<subject_id>_<suffix>.npz`.

There are **five** classes (not four): four are actual warps plus an identity baseline.

| `deformation_class` | Suffix | Example filename | TorchIO | Description |
| --- | --- | --- | --- | --- |
| `none` | `none` | `100206_none.npz` | (identity) | No warp; `moving ≈ source`, `u ≈ 0`. Baseline (~5%). |
| `rigid` | `rig` | `100206_rig.npz` | `RandomAffine` | Rotation + translation only (`scales=1`). (~20%). |
| `affine` | `aff` | `100206_aff.npz` | `RandomAffine` | Full affine: scale, shear, rotate, translate. (~25%). |
| `elastic` | `ela` | `100206_ela.npz` | `RandomElasticDeformation` | B-spline elastic / non-rigid only. (~25%). |
| `affine_elastic` | `aela` | `100206_aela.npz` | Affine → elastic | Global affine **then** local elastic (combined). (~25%). |

**Naming rationale**

- **`elastic`** — matches TorchIO `RandomElasticDeformation` (clearer than `non_rigid`).
- **`affine_elastic`** — reads as “affine then elastic” (clearer than `affine_rigid_plus_non_rigid`).
- **`rigid`** — rotation + translation without scale (was `rigid_like` in early versions).
- Short **suffixes** keep filenames readable; full class name is always in `deformation_class`.

**Registration taxonomy mapping**

```text
none          →  identity (not a deformation; QC / baseline)
rigid         →  rigid motion
affine        →  affine motion
elastic       →  non-rigid / deformable (elastic only)
affine_elastic →  composite (global + local), common in real registration
```

**Legacy names** (regenerate data if NPZ still uses these):

| Legacy `deformation_class` | Legacy suffix | Current |
| --- | --- | --- |
| `rigid_like` | `_rig` | `rigid` / `_rig` |
| `non_rigid` | `_nr` | `elastic` / `_ela` |
| `affine_rigid_plus_non_rigid` | `_ar` | `affine_elastic` / `_aela` |

Constants in `experiments/unigrad-synth/create_synth_data.py`: `DEFORMATION_RATIOS`,
`DEFORMATION_SUFFIX`.

**QC visualization** (`visualize_synth_data.py`, default `--selection random`): rows are one example
each of **`rigid`**, **`affine`**, **`elastic`** (not `none` or `affine_elastic`).

### TorchIO: how deformation is created

Full technical walkthrough: **`docs/registration-concepts.md` § 3D synthetic deformation and
displacement extraction** (implicit backward warp, identity-grid trick, z-score vs φ).

Summary:

1. TorchIO/SimpleITK samples transform parameters (mm) and applies **implicit backward warping**
   — no full-resolution deformation field is materialized in memory.
2. Dense **`u`** is recovered via the **identity-grid trick** in `create_synth_data.py`.
3. Intensities are **masked z-scored after** warp; **`u` is unchanged** (see below and registration-concepts).

See `docs/registration-concepts.md` for φ vs u and backward warping.

### Intensity normalization vs displacement (important)

**Z-scoring `source` and `moving` after computing `u` does not change `u`.**

| Quantity | What it is | Affected by intensity z-score? |
| --- | --- | --- |
| `u` (stored GT) | Geometric map: which **voxel index** each output voxel samples from | **No** — from coordinate grid, not intensities |
| `source`, `moving` | Scalar **intensity** per voxel | **Yes** — linear `(I − μ) / σ` for training stability |
| `phi_pred` (downstream) | Registration network output: displacement / position map in **voxel index space** | **No** (in principle) — geometry, not intensity scaling |

#### Phase I: why z-score does not alter `u`

`u` is extracted from warping an **identity coordinate grid**, not from MRI intensities. Masked
z-score `(I − μ) / σ` is applied only to scalar intensity arrays **after** `u` is computed. It
rescales signal values per voxel; it does not move indices on the grid.

#### Downstream registration: why z-score does not alter `phi_pred`

When Phase II runs UniGradICON (or similar) on the saved NPZ pair:

- Inputs are z-scored **`source`** and **`moving`** on the **same voxel grid** as stored **`u`**.
- Registration estimates **where** structures align (geometry). Displacement φ (or u) lives in
  **index space**, not intensity space.
- Masked z-score with **shared μ, σ** (from the fixed/source brain mask) is an **affine intensity
  transform** applied consistently to both images. Similarity metrics used by deformable registration
  (e.g. LNCC) are invariant to affine intensity rescaling; anatomical alignment — hence φ — is
  unchanged in principle.
- We apply the **same** μ, σ to `source` and `moving` so the pair remains on one intensity scale
  for the network. Do **not** z-score `source` and `moving` with independent statistics.

**Caveat:** we warp raw T1, then z-score (`moving_z = (warp(source) − μ) / σ`). This differs
slightly from `warp(z-score(source))` near interpolation boundaries, but does not change the stored
GT `u`. Downstream registration optimizes φ on the normalized pair it receives; that φ targets the
same voxel-grid correspondence as `u`.

**Do not** rescale or recompute `u` / φ after z-score.

Pipeline order in `create_synth_data.py`:

1. Warp **raw** T1 with TorchIO (+ affine); extract **`u`** from grid.
2. QC on **raw** intensities (e.g. moving mean vs source in mask).
3. Apply **masked z-score** to `source` and `moving` using μ, σ from **source** mask only; save
   with unchanged **`u`**.

### Transform defaults (physical units)

See table in § Deformation classes and file nomenclature for class ↔ suffix mapping. Parameter
defaults per class:

| Class | TorchIO | Key parameters |
| --- | --- | --- |
| `none` | (identity) | — |
| `rigid` | `RandomAffine` | `scales=1`, `degrees=±6°`, `translation=±4` **mm** |
| `affine` | `RandomAffine` | `scales=0.97–1.03`, `degrees=±8°`, `translation=±4` **mm** |
| `elastic` | `RandomElasticDeformation` | 7 control points, `max_displacement=±6` **mm** |
| `affine_elastic` | Affine then elastic | Both of the above |

### QC

| Check | Default | Notes |
| --- | --- | --- |
| `MAX_U_INTERIOR_VOX` | 25 | Max ‖u‖ in mask ∩ interior margin |
| `MAX_U_GLOBAL_VOX` | 60 | Max ‖u‖ full volume |
| `MIN_MOVING_MEAN_RATIO` | 0.05 | On **raw** intensities before z-score |
| `INTERIOR_MARGIN` | 10 voxels | Border excluded from interior QC |
| `MAX_TRANSFORM_ATTEMPTS` | 20 | Resample transform if QC fails |

Failed samples are still saved with `qc_passed=False`; list in `qc_flagged_paths.txt`.

### Orientation (storage vs display)

- **Stored volumes** stay in native **LAS** voxel array order (from NIfTI); no reorientation in
  `create_synth_data.py`.
- **`u`** is in the same voxel index space as `source` / `moving`.
- Radiological axial display (`rot90`) is **visualization only** — see
  `visualize_hcp_data.py` and `docs/hcp-dataset.md`.

### Example commands

```bash
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --workers 8
python experiments/unigrad-synth/create_synth_data.py --input-path datasets/hcp --output-path datasets/hcp_synth --max-subjects 100 --workers 8
```

### Planned next steps (HCP branch)

- [ ] Phase II: UniGradICON on HCP synth pairs → `u_pred`, `error_map`
- [ ] Phase III: 3D error-map U-Net (analogous to IO track)
- [ ] `visualize_hcp_synth_data.py` for NPZ QC

---

## Phase I — IXI 2D triplets (legacy)

<!-- TODO: document affine/elastic defaults for old 2D path if still referenced -->

### Example commands

<!-- TODO -->

---

## Phase II — UniGrad fivers (`create_unigrad_synth_data.py`)

### NPZ schema

<!-- TODO: list keys, units (pixels), phi scaling from ICON -->

### Example commands

<!-- TODO -->

---

## Training (`train_unigrad_synth_unet.py`)

### Loss and metrics

<!-- TODO: MSE default, smooth_weight, early stop, metrics.csv columns -->

### Model architecture

<!-- TODO: UNet2D, base channels, 4 → 1 -->

### Example commands

<!-- TODO -->

---

## Evaluation (`eval_unigrad_synth_unet.py`)

### Default outputs

<!-- TODO: training_curves.png, test_metrics.json, QC PNGs -->

### Example commands

<!-- TODO -->

---

## Error-map U-Net runs

<!-- TODO: comparison table run1, run2, … -->

### run1 — `error_unet_run1`

<!-- TODO -->

---

## Related files

| File | Role |
| --- | --- |
| `experiments/unigrad-synth/create_synth_data.py` | Phase I — HCP 3D synth NPZ |
| `experiments/unigrad-synth/create_unigrad_synth_data.py` | Phase II fivers (IXI 2D) |
| `experiments/unigrad-synth/modify_synth_data.py` | Triplet post-processing (IXI 2D) |
| `experiments/unigrad-synth/visualize_synth_data.py` | Triplet QC |
| `experiments/unigrad-synth/visualize_unigrad_data.py` | Fiver QC |
| `experiments/unigrad-synth/visualize_hcp_data.py` | HCP T1w QC |
| `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` | Train 2D U-Net |
| `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py` | Eval + figures |
| `scripts/download_hcp.sh` | HCP S3 download |
| `docs/registration-concepts.md` | Registration vocabulary |
| `docs/hcp-dataset.md` | HCP layout and download |
