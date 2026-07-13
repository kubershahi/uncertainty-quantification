# HCP dataset — role, download, layout, and QC

Notes for **Human Connectome Project (HCP) Young Adult (S1200)** structural T1w data used in this
repo. HCP is the **real-data** track alongside **IXI 2D synth** (`experiments/synth-data-gen/torchio/`) and
**IXI 3D IO** (`experiments/error-map-gen/unigrad-io/`). See `docs/registration-concepts.md` for registration
vocabulary and `docs/unigrad-synth-experiment.md` for the IXI synth pipeline.

---

## Role in this project

| Track | Data | Space | Ground truth |
| --- | --- | --- | --- |
| **Synth (IXI 2D)** | TorchIO triplets → UniGrad fivers | 2D slices, pixels | Known `phi_true` |
| **IO (IXI 3D)** | UniGrad ICON instance optimization | 3D volumes, voxels | `phi_predio` vs `phi_pred` |
| **HCP (this doc)** | Minimally preprocessed T1w NIfTI | **Native** 3D, LAS voxels | Synth `u` in `hcp_synth/` |

**Why HCP (vs IXI for real data):**

- ~0.7 mm isotropic T1, large healthy cohort, consistent acquisition protocol
- **T1w** minimally preprocessed in **native subject space** (not MNI) — registration remains a meaningful problem
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
| `T1w_acpc_dc_restore_brain.nii.gz` | Brain-extracted T1 — primary anatomy / future **fixed** or **moving** image |
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

**Scale:** full download ≈ 1200 subjects × 3 files (~1113+ objects on disk depending on completeness).

---

## QC visualization

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

## HCP 3D synthetic data (Phase I)

**Script:** `experiments/synth-data-gen/torchio/create_synth_data.py`  
**Output:** `datasets/synth-data/torchio/hcp/{Train,Val,Test}/<subject_id>_<suffix>.npz`

One warp per subject; balanced deformation classes (`none`, `rigid`, `affine`, `elastic`,
`affine_elastic`); filenames `<subject_id>_<suffix>.npz` with suffixes `none`, `rig`, `aff`, `ela`,
`aela`. Masked z-score intensities; ground-truth `u` in voxels. Class table:
`docs/unigrad-synth-experiment.md` § *Deformation classes and file nomenclature*; pipeline detail:
`docs/registration-concepts.md` § *3D synthetic deformation and displacement extraction*;
experiment config: `docs/unigrad-synth-experiment.md`.

```bash
python experiments/synth-data-gen/torchio/create_synth_data.py --input-path datasets/hcp --output-path datasets/synth-data/torchio/hcp --workers 8
```

---

## Planned downstream

- [x] Synthetic deformations on HCP T1 (TorchIO 3D) — `create_synth_data.py`
- [ ] UniGradICON on HCP synth pairs → predicted displacement, error map
- [ ] 3D error-map U-Net training
- [ ] `visualize_hcp_synth_data.py` for NPZ QC

---

## Related files

| File | Role |
| --- | --- |
| `scripts/download_hcp.sh` | S3 download to `datasets/hcp/` |
| `deploy/nautilus/scripts/hcp_subjects_test10.txt` | 10-subject smoke list |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | HCP 3D synth NPZ generation |
| `docs/registration-concepts.md` | Registration / displacement / error-map concepts |
| `experiments/synth-data-gen/torchio/visualize_hcp_data.py` | Random-sample QC figure |
| `docs/unigrad-synth-experiment.md` | HCP synth pipeline detail |
| `docs/unigrad-io-experiment.md` | IXI 3D IO + error-map U-Net track |
