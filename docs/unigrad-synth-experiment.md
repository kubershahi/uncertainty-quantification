# UniGrad Synth experiment — HCP pipeline overview

End-to-end plan for synthetic data and error-map generation, and error-map U-Net training
on HCP anatomy. For registration vocabulary see `docs/registration-concepts.md`. For HCP download,
layout, and Phase I script detail see `docs/hcp-dataset.md`.

---

## Three phases

| Phase | Goal | Scripts | Output |
| --- | --- | --- | --- |
| **1) Synthetic data generation** | Known GT `u_gt` on **source** lattice from TorchIO + invert | `experiments/synth-data-gen/torchio/` | `datasets/synth-data/torchio/hcp/` |
| **2) Error-map generation** | `u_pred` from `net(moving, source)` → ‖u_gt − u_pred‖ | `experiments/error-map-gen/unigrad-synth/` | `datasets/error-map/unigrad-synth/hcp/` |
| **3) Error-map U-Net regression** | Learn to predict error map from registration inputs | `experiments/regression/unigrad-synth/` | `assets/runs/regression/unigrad-synth/` (planned) |

**Artifact roots:**

| Role | Path |
| --- | --- |
| HCP T1w (real) | `datasets/hcp/` |
| HCP synth NPZ | `datasets/synth-data/torchio/hcp/{Train,Val,Test}/` |
| Error-map NPZ | `datasets/error-map/unigrad-synth/hcp/` |
| QC figures (Phase I) | `assets/images/synth-data/torchio/hcp/` |
| QC figures (Phase II) | `assets/images/error-map/unigrad-synth/hcp/` |
| U-Net runs (planned) | `assets/runs/regression/unigrad-synth/` |

**Shared displacement convention (Phases 1–2):**

```text
moving(x + u(x)) ≈ source(x)     # u_gt and u_pred on the source/fixed lattice
```

---

## Phase 1 — Synthetic data generation

TorchIO applies controlled deformations to HCP T1w volumes. Identity-grid warping yields a temporary
backward field; SimpleITK field inversion stores ground-truth **`u_gt`** in voxel units on the
**source/fixed** lattice. One sample per subject in full run; five deformation classes
(`none`, `rigid`, `affine`, `elastic`, `affine_elastic`); deterministic Train / Val / Test splits
and balanced class mix.

For NPZ schema, ‖u_gt‖ cleanup, split policy, visualization modes (`per_class`, `random`,
`min_median_max`), and commands, see **`docs/hcp-dataset.md` § HCP synthetic data generation
(Phase I)**.

---

## Phase 2 — Error-map generation (medical image registration model)

**Goal:** For each synth pair from Phase 1, run **UniGradICON** to estimate `u_pred` on the same
source lattice as `u_gt`. The **error map** `‖u_gt − u_pred‖` is the per-voxel registration error
that Phase 3 learns to predict.

**Workflow:**

1. Load `source`, `moving`, `u_gt`, `identity_grid_mask` from HCP synth NPZ.
2. Register with UniGradICON: **`net(moving, source)`** → `u_pred` in the same voxel index space as `u_gt`.
3. Cleanup `u_pred` (OOB via mask → 12-voxel border → p99.9 clip).
4. Write `u_error_map = ‖u_gt − u_pred‖` and `error_map_mask` under
   `datasets/error-map/unigrad-synth/hcp/{Train,Val,Test}/`.

**Why synth first:** GT `u_gt` is known, so registration error is supervised without manual labels.
This isolates model failure modes (affine vs elastic vs composite warps) before real held-out
anatomy.

**Status:** implemented for HCP volumes (`create_unigrad_synth_data.py`, `visualize_unigrad_data.py`).

**Scripts:**

- `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` — batch error-map generation
- `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` — QC figures (optional cosine similarity)

---

## Phase 3 — Error-map U-Net training on HCP synth

**Goal:** Train a U-Net to **predict the error map** from registration inputs available at inference
time: fixed image, moving image, and predicted displacement from the registration model. Trained on
Phase 2 outputs where the target error map is known from Phase 1 GT.

**Planned workflow:**

1. **Inputs:** channels derived from `source`, `moving`, and `u_pred` (normalized consistently with
   Phase 2).
2. **Target:** `u_error_map` from Phase 2 (voxel units), masked by `error_map_mask`.
3. **Train / val / test:** same split convention as Phase 1 (`Train` / `Val` / `Test` folders).
4. **Eval:** held-out metrics (e.g. MAE on error map, calibration plots) and QC overlays on sample
   volumes.

The U-Net acts as an **uncertainty proxy**: it learns where the registration model is likely to be
wrong as a function of image appearance and predicted deformation, without re-running the heavy
registration network at inference.

**Status:** planned for HCP 3D (existing train/eval scripts target legacy 2D).

**Scripts (target):**

- `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py`
- `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py`

---

## Intensity normalization and displacement (cross-phase note)

Masked z-score on `source` and `moving` does **not** change stored `u_gt` or predicted `u_pred` in
voxel index space — intensities are rescaled; geometry is not. Phase 2 registration and Phase 3
training both use the same normalized pair on the fixed voxel grid. See
`docs/registration-concepts.md` for φ vs `u` and registration-map conventions.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | Phase 1 — synth NPZ generation (`u_gt`) |
| `experiments/synth-data-gen/torchio/visualize_synth_data.py` | Phase 1 — NPZ QC figures |
| `experiments/synth-data-gen/torchio/visualize_hcp_data.py` | Raw HCP T1w QC |
| `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` | Phase 2 — error-map generation |
| `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` | Phase 2 — error-map QC |
| `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` | Phase 3 — U-Net training (HCP adapt planned) |
| `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py` | Phase 3 — eval and figures |
| `docs/hcp-dataset.md` | HCP download, layout, Phase 1 detail |
| `docs/registration-concepts.md` | Registration / displacement / error-map concepts |
