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
| **3) Error-map U-Net regression** | Learn to predict error map from registration inputs | `experiments/regression/unigrad-synth/` | `assets/runs/regression/unigrad-synth/hcp/` |

**Artifact roots:**

| Role | Path |
| --- | --- |
| HCP T1w (real) | `datasets/hcp/` |
| HCP synth NPZ | `datasets/synth-data/torchio/hcp/{Train,Val,Test}/` |
| Error-map NPZ | `datasets/error-map/unigrad-synth/hcp/` |
| QC figures (Phase I) | `assets/images/synth-data/torchio/hcp/` |
| QC figures (Phase II) | `assets/images/error-map/unigrad-synth/hcp/` |
| U-Net runs | `assets/runs/regression/unigrad-synth/hcp/` |

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
`min_median_max`), and commands, see **`docs/hcp-dataset.md` § HCP synthetic data generation**.

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

**Status:** implemented (`create_unigrad_synth_data.py`, `visualize_unigrad_data.py`).

---

## Phase 3 — Error-map U-Net training on HCP synth

**Goal:** Train a **3D U-Net** to predict the error map from registration inputs available at
inference: `source`, `moving`, and `u_pred` (5 channels). Target is Phase 2 `u_error_map`; loss and
metrics use `source_mask`.

**Workflow:**

1. **Inputs:** `[source, moving, u_pred_x, u_pred_y, u_pred_z]` with `u_pred / u_scale`.
2. **Target:** `‖u_gt − u_pred‖` (or stored `u_error_map`).
3. **Splits:** same `Train` / `Val` / `Test` folders as Phase 1.
4. **Eval:** masked MAE / RMSE / Pearson r; QC mid-slice overlays.

The U-Net is an **uncertainty proxy**: it learns where the registration model is likely wrong from
image appearance and predicted deformation, without re-running UniGradICON at inference.

**Results (example):** [`assets/runs/regression/unigrad-synth/hcp/error_unet_run1/`](../assets/runs/regression/unigrad-synth/hcp/error_unet_run1/) — Test (147 vols, `source_mask`): MAE **0.338**, RMSE **0.485**, Pearson r **0.874**. See that folder for curves and orthogonal QC.

**Status:** implemented (`train_unigrad_synth_unet.py`, `eval_unigrad_synth_unet.py`). Example run `error_unet_run1` used
`base_channels=16`, val starting at epoch 3 then every 3 epochs (80-epoch budget, early stop). Eval writes orthogonal QC
(`test_random_orthogonal/`, `test_mmm_orthogonal/`) then `test_metrics.json`; use `--mode figures|metrics|both`.
See `docs/registration-concepts.md` § Phase III for CNN / U-Net background.

---

## Intensity normalization and displacement (cross-phase note)

Masked z-score on `source` and `moving` does **not** change stored `u_gt` or predicted `u_pred` in
voxel index space — intensities are rescaled; geometry is not. See
`docs/registration-concepts.md` for φ vs `u`.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | Phase 1 — synth NPZ (`u_gt`) |
| `experiments/synth-data-gen/torchio/visualize_synth_data.py` | Phase 1 — NPZ QC |
| `experiments/synth-data-gen/torchio/visualize_hcp_data.py` | Raw HCP T1w QC |
| `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` | Phase 2 — error maps |
| `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` | Phase 2 — QC |
| `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` | Phase 3 — U-Net train |
| `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py` | Phase 3 — eval |
| `docs/hcp-dataset.md` | HCP download, layout, Phase 1 detail |
| `docs/registration-concepts.md` | Registration / U-Net concepts |
