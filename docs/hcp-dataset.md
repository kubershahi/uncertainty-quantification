# HCP Dataset Details

Notes for **Human Connectome Project (HCP) Young Adult (S1200)** structural T1w data used in this
repo. See `docs/registration-concepts.md` for registration vocabulary.

---

## Role in this project

HCP is the **real-anatomy** source for synthetic registration pairs: minimally preprocessed T1w
volumes in **native subject space**, with known ground-truth displacement fields from TorchIO.

**Why HCP (vs datasets like IXI):**

- ~0.7 mm isotropic T1, large healthy cohort, consistent acquisition protocol
- **T1w** minimally preprocessed in **native subject space** (not MNI) — registration remains a
  meaningful inter-subject problem
- Skull-stripped brain volumes and FreeSurfer masks ship with the release — useful for masking,
  QC, and visualization
- Public AWS mirror compatible with batch download on NRP

**Why T1w only (for now):**

- Primary structural modality for deformable registration and segmentation QC
- T2, diffusion, fMRI reserved for future multimodal work

---

## Access and download

**Source:** [HCP Young Adult](https://www.humanconnectome.org/study/hcp-young-adult) — S1200 release.

**AWS bucket:**

```text
s3://hcp-openaccess/HCP_1200/
```

**Credentials:** ConnectomeDB HCP Open Access terms + project-specific AWS keys (not anonymous
`--no-sign-request`). On NRP, credentials live on the PVC under `~/.aws/` (see
`deploy/nautilus/scripts/env.sh`).

**Download script:** `scripts/download_hcp.sh`

| Env / flag | Purpose |
| --- | --- |
| `HCP_OUTDIR` | Output root (default `datasets/hcp`) |
| `SUBJECT_LIST_FILE` | Subset list (one 6-digit ID per line); skips S3 listing |
| `REFRESH_SUBJECT_LIST=1` | Re-list all subjects → `datasets/hcp/.subjects.txt` |
| `PARALLEL_JOBS` | Parallel subject downloads (default 4) |
| `AWS_REGION` | Default `us-east-1` |

Example (full cohort, parallel):

```bash
bash scripts/download_hcp.sh
```

Smoke test (10 subjects, bundled IDs):

```bash
SUBJECT_LIST_FILE=deploy/nautilus/scripts/hcp_subjects_test10.txt PARALLEL_JOBS=4 bash scripts/download_hcp.sh
```

**On NRP:** data persists on PVC `unc-files` at
`/files/repo/uncertainty-quantification/datasets/hcp/` (same path after `env.sh`).

---

## HCP release layout (what we skip)

```text
HCP_1200/<subject_id>/
├── unprocessed/     # near-raw scanner output
├── T1w/             # native-space minimal preprocessing  ← we use this
└── MNINonLinear/    # + nonlinear MNI registration        ← not selected
```

| Folder | Content | Used? |
| --- | --- | --- |
| `unprocessed/` | Scanner-native | No |
| **`T1w/`** | AC-PC, distortion/bias corrected, skull-stripped T1 | **Yes** |
| `MNINonLinear/` | T1w + nonlinear template warp | No — removes inter-subject variability we want for registration |

### Minimal preprocessing in `T1w/` (HCP pipeline)

- Gradient distortion correction
- Readout distortion correction
- Bias field correction
- AC-PC alignment (rigid only — **not** nonlinear registration)
- Skull stripping

---

## What we store locally

Per subject, under `datasets/hcp/<subject_id>/T1w/` — **original HCP filenames** (no rename):

| File | Role in project |
| --- | --- |
| `T1w_acpc_dc_restore_brain.nii.gz` | Brain-extracted T1 — primary anatomy / **fixed** or **moving** image |
| `aparc+aseg.nii.gz` | FreeSurfer labels — QC, ROI evaluation, visualization |
| `brainmask_fs.nii.gz` | Binary brain mask — restrict metrics / loss to brain voxels |

Example:

```text
datasets/hcp/
├── .subjects.txt          # cached S3 subject list (full download)
├── download.log
└── 100206/
    └── T1w/
        ├── T1w_acpc_dc_restore_brain.nii.gz
        ├── aparc+aseg.nii.gz
        └── brainmask_fs.nii.gz
```

**Scale:** full download ≈ 1200 subjects × 3 files. The current full synth cohort uses **1113**
subjects that have complete T1w + brainmask on disk (see split table below).

---

## Volume geometry

HCP T1w brains are **~0.7 mm isotropic** in **native subject space** (LAS). Exact `(X, Y, Z)`
varies slightly by subject; a typical skull-stripped volume is on the order of
**~260 × 311 × 260** voxels.

Synth NPZs keep that native grid: `source`, `moving`, and `u_gt` all share the same `(X, Y, Z)`.
Displacement units in the NPZ are **voxels** (TorchIO warps are parameterized in mm via the NIfTI
affine, then converted to voxel `u_gt`).

---

## Full cohort split counts (seed 42)

From `datasets/synth-data/torchio/hcp/split_manifest.json`:

| Split | Subjects / NPZs | Share |
| --- | ---: | ---: |
| **Train** | **857** | ~77% |
| **Val** | **109** | ~10% |
| **Test** | **147** | ~13% |
| **Total** | **1113** | 100% |

Policy target is **75 / 10 / 15**; realized counts use hash assignment + largest-remainder class
quotas. Per-split deformation mix is recorded in `split_manifest.json` /
`split_class_u_stats.csv`.

---

## Raw HCP QC visualization

**Script:** `experiments/synth-data-gen/torchio/visualize_hcp_data.py`

Layout: **columns = subjects**, **rows = modality** (T1 top, mask middle; optional segmentation row).

**Orientation:** volumes are stored as **LAS** (`aff2axcodes` → `L/A/S`). Training and synth data
keep this native voxel order. `visualize_hcp_data.py` applies `np.rot90` for **radiological** axial
display only (image-left = R, image-right = L); see figure subtitle.

```bash
python experiments/synth-data-gen/torchio/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --save-path assets/images/synth-data/torchio/hcp/hcp_random3.png --no-show
python experiments/synth-data-gen/torchio/visualize_hcp_data.py --data-dir datasets/hcp --num-samples 3 --show-segmentation --save-path assets/images/synth-data/torchio/hcp/hcp_random3_seg.png --no-show
```

---

## HCP synthetic data generation

Build **known-displacement registration samples** from HCP T1w: fixed `source`, warped `moving`,
and ground-truth displacement `u_gt` (voxel units) on the **source/fixed** lattice.
Scripts live under `experiments/synth-data-gen/torchio/`.

| Script | Role |
| --- | --- |
| `create_synth_data.py` | Generate NPZ samples from HCP T1w |
| `visualize_synth_data.py` | QC figures from NPZ output |

**Dataset paths:**

| Artifact | Path |
| --- | --- |
| Raw HCP | `datasets/hcp/` |
| Full synth cohort | `datasets/synth-data/torchio/hcp/{Train,Val,Test}/` |
| Dry-run synth | `datasets/synth-data/torchio/hcp_dryrun/` (flat) |
| Subset run (e.g. 100 subjects) | `datasets/synth-data/torchio/hcp_100/` |
| QC figures | `assets/images/synth-data/torchio/hcp/` |

---

### `create_synth_data.py` — generation logic

**Input:** `datasets/hcp/<subject_id>/T1w/T1w_acpc_dc_restore_brain.nii.gz` +
`brainmask_fs.nii.gz`

**Output:**

- **Full run:** `datasets/synth-data/torchio/hcp/{Train,Val,Test}/<subject_id>_<suffix>.npz`
- **Dry run:** flat folder, `<subject_id>_<class>[_NN].npz`

**Each NPZ contains:**

| Key | Description |
| --- | --- |
| `source` | Fixed image (masked z-score, float32) `(X, Y, Z)` |
| `moving` | Deformed image (same normalization) `(X, Y, Z)` |
| `u_gt` | Registration displacement `(3, X, Y, Z)` in **voxels** on the **source/fixed** lattice: `moving(x + u_gt(x)) ≈ source(x)` |
| `source_mask`, `moving_mask`, `identity_grid_mask` | Masks for viz / Phase II; `identity_grid_mask` marks valid `u_gt` voxels |
| `deformation_class` | One of `none`, `rigid`, `affine`, `elastic`, `affine_elastic` |
| `subject_id` | HCP subject ID |

TorchIO transforms run in physical space (mm) using the NIfTI affine. The identity-grid trick
first yields a temporary backward field `u_back` (`moving(x) ≈ source(x + u_back(x))`); SimpleITK
`InvertDisplacementField` (voxel units) converts it to stored **`u_gt`** on the shared source grid.

#### Deformation classes and filenames

One warp per subject in full run. Class is encoded in the filename suffix:

| Class | Suffix | Target share (full run) |
| --- | --- | --- |
| `none` | `none` | 5% |
| `rigid` | `rig` | 20% |
| `affine` | `aff` | 25% |
| `elastic` | `ela` | 25% |
| `affine_elastic` | `aela` | 25% |

Single parameter envelope per class (no low/mid/high tiers). Rigid / affine / affine+elastic sit
between previous mid and high settings; pure elastic uses a stronger envelope (~12 mm max
displacement) for clearer deformation.

#### Full run: splits and class assignment

- **Split policy:** deterministic **75 / 10 / 15** Train / Val / Test by subject hash (`--seed`,
  default 42). Realized counts for the current full cohort: **857 / 109 / 147** (1113 total).
  Before generation, the script prints split sizes and per-split class counts.
- **Class assignment:** within each split, subjects are shuffled with a split-specific seed; class
  quotas follow the target ratios above (largest-remainder rounding).

Use `--max-subjects N` for a subset (e.g. 100 subjects → ~75 / 10 / 15 with the same ratios).

#### Dry run

`--dry-run [N]` (default N=5) writes **5 classes × N samples** to a **flat** output folder.
Distinct subjects are used within and across classes when enough HCP subjects are available.
Filenames use the full class name (e.g. `100206_rigid.npz`, `100206_affine_elastic_02.npz`).

#### ‖u_gt‖ cleanup (both modes)

Applied in order after inversion:

1. Zero invalid voxels using `identity_grid_mask` (in-bounds for `x + u_gt(x)`, intersected with warp validity)
2. Zero a **12-voxel border** on each face
3. Clip ‖u_gt‖ at **p99.9** (percentile over **nonzero** voxels only)

#### Stats written at generation time

| Mode | CSV | Content |
| --- | --- | --- |
| Dry run | `dryrun_class_u_stats.csv` | Per class: per-sample min/Q1/mean/Q3/max, then **mean over samples** |
| Full run | `split_class_u_stats.csv` | Per split × class: same aggregation over **all** samples in that bucket |

Also writes `split_manifest.json` (full run) with split counts and deformation mix.

#### Commands

```bash
# Dry run (25 samples)
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp_dryrun --dry-run 5 --workers 16

# Full cohort
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp --workers 16

# Subset (100 subjects)
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp_100 --max-subjects 100 --workers 16
```

---

### `visualize_synth_data.py` — QC figures

**Input:** NPZ folder from `create_synth_data.py` (flat dry-run or `{Train,Val,Test}/` full cohort).

**Figure layout:** columns = source (fixed), source + `u_gt` vectors, warped (moving), ‖u_gt‖
(+ reserved colorbar); optional checkerboard (`--checkerboard`). Axial slices use **radiological**
display (`rot90`). Row layout: `--run-view orthogonal` (axial / coronal / sagittal) or `montage`
(three axial slices).

**Titles:**

- Main: `HCP Synthetic Data Plot ({No/Rigid/Affine/Elastic/Affine+Elastic} Transformation)`
- Subtitle: `Subject <id> - Radiological-style display - Orthogonal/Montage View`
- Footer: per-sample ‖u_gt‖ stats (min, Q1, mean, Q3, max) for the plotted sample

Class grouping for full-run selection uses the **filename suffix only** (no volume load).

#### Dry run — `--selection per_class`

One random sample per deformation class (5 classes). Writes:

- One PNG per class under `--save-dir`
- `chosen_sample_u_stats.csv` — ‖u_gt‖ stats for the **plotted samples only**

```bash
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp_dryrun --selection per_class --save-dir assets/images/synth-data/torchio/hcp/dryrun_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard
```

#### Full cohort — `--selection random` (default)

For each split (`Train`, `Val`, `Test`):

1. Group NPZs by class from filename suffix
2. Pick **one random sample per class** (`--seed`, split-specific offset)
3. Plot → `{save_dir}/{split}/{class}.png`

**Total: up to 15 figures** (5 classes × 3 splits). Does **not** recompute cohort-wide stats
(those live in `split_class_u_stats.csv` from creation).

```bash
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp --selection random --save-dir assets/images/synth-data/torchio/hcp/fullrun_random_orthogonal --no-show --run-view orthogonal --u-contours --checkerboard
```

#### Full cohort — `--selection min_median_max`

`none` is excluded (‖u_gt‖ ≈ 0 everywhere). For each split:

1. Score every non-`none` sample by `--u-metric` (`mean` or `max` of ‖u_gt‖ over the volume)
2. Select **min**, **median**, and **max** across that split
3. Plot → `{save_dir}/{split}/{min|median|max}.png`
4. Subtitle adds rank label, e.g. `Minimum of u mean sample`

**Total: 9 figures** (3 ranks × 3 splits).

On first run, writes `{save_dir}/min_median_max_selection.csv` with columns
`split`, `rank`, `u_metric`, `subject_id`, `file`, `deformation_class`, `u_score`.
Reruns with the same `--u-metric` reuse cached picks and skip re-scoring (delete the CSV to force
recompute).

```bash
python experiments/synth-data-gen/torchio/visualize_synth_data.py --data-dir datasets/synth-data/torchio/hcp --selection min_median_max --u-metric mean --save-dir assets/images/synth-data/torchio/hcp/fullrun_mmm --no-show --run-view orthogonal --u-contours --checkerboard
```

---

## Planned downstream

Error-map generation (UniGradICON) and error-map U-Net training build on this synth cohort. See
`docs/unigrad-synth-experiment.md` and `docs/registration-concepts.md`.

---

## Related files

| File | Role |
| --- | --- |
| `scripts/download_hcp.sh` | S3 download to `datasets/hcp/` |
| `deploy/nautilus/scripts/hcp_subjects_test10.txt` | 10-subject smoke list |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | HCP synth NPZ generation |
| `experiments/synth-data-gen/torchio/visualize_synth_data.py` | NPZ QC figures |
| `experiments/synth-data-gen/torchio/visualize_hcp_data.py` | Raw HCP T1/mask QC figure |
| `docs/registration-concepts.md` | Registration / displacement / error-map concepts |
