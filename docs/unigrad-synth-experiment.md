# UniGrad Synth experiment — HCP pipeline overview

End-to-end plan for synthetic data and error-map generation, and error-map U-Net training
on HCP anatomy. For registration vocabulary see `docs/registration-concepts.md`. For HCP download,
layout, and Phase I script detail see `docs/hcp-dataset.md`.

---

## Three phases

| Phase | Goal | Scripts (current / planned) | Output |
| --- | --- | --- | --- |
| **1) Synthetic data generation** | Known GT displacement `u` from TorchIO warps | `experiments/synth-data-gen/torchio/` | `datasets/synth-data/torchio/hcp/` |
| **2) Error-map generation** | `u_pred` from UniGradICON → error map vs GT | `experiments/error-map-gen/unigrad-synth/` | `datasets/error-map/unigrad-synth/hcp/` (planned) |
| **3) Error-map U-Net regression** | Learn to predict error map from registration inputs | `experiments/regression/unigrad-synth/` | `assets/runs/regression/unigrad-synth/` (planned) |

**Artifact roots:**

| Role | Path |
| --- | --- |
| HCP T1w (real) | `datasets/hcp/` |
| HCP synth NPZ | `datasets/synth-data/torchio/hcp/{Train,Val,Test}/` |
| Error-map NPZ (planned) | `datasets/error-map/unigrad-synth/hcp/` |
| QC figures | `assets/images/synth-data/torchio/hcp/` |
| U-Net runs (planned) | `assets/runs/regression/unigrad-synth/` |

---

## Phase 1 — Synthetic data generation

TorchIO applies controlled deformations to HCP T1w volumes and writes registration triplets with
ground-truth displacement `u` in voxel units. One sample per subject in full run; five deformation
classes (`none`, `rigid`, `affine`, `elastic`, `affine_elastic`); deterministic Train / Val / Test
splits and balanced class mix.

For NPZ schema, ‖u‖ cleanup, split policy, visualization modes (`per_class`, `random`,
`min_median_max`), and commands, see **`docs/hcp-dataset.md` § HCP synthetic data generation
(Phase I)**.

---

## Phase 2 — Error-map generation (medical image registration model)

**Goal:** For each synth pair from Phase 1, run **UniGradICON** (or an equivalent pretrained
deformable registration model) to estimate a predicted displacement field `u_pred`. Compare against
stored ground truth `u` to build an **error map** — the per-voxel registration error that Phase 3
will learn to predict.

**Planned workflow:**

1. Load `source`, `moving` from HCP synth NPZ (masked z-score intensities on the native voxel grid).
2. Register moving → fixed with UniGradICON; recover `u_pred` in the same voxel index space as `u`.
3. Compute error map (e.g. magnitude ‖u_pred − u‖ or stacked component error).
4. Write augmented NPZ under `datasets/error-map/unigrad-synth/hcp/{Train,Val,Test}/` with keys such
   as `u_pred`, `u` (GT), `error_map`, and masks for valid voxels.

**Why synth first:** GT `u` is exact, so registration error is known without manual annotation.
This isolates model failure modes (affine vs elastic vs composite warps) before applying the same
pipeline to real HCP-held-out subjects.

**Status:** planned. Existing `create_unigrad_synth_data.py` targets legacy 2D triplets; HCP volume
input and output paths will be wired in next.

**Scripts (target):**

- `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` — batch error-map generation
- `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` — QC figures

---

## Phase 3 — Error-map U-Net training on HCP synth

**Goal:** Train a U-Net to **predict the error map** from registration inputs available at inference
time: fixed image, moving image, and predicted displacement (or position map) from the registration
model. Trained on Phase 2 outputs where the target error map is known from Phase 1 GT.

**Planned workflow:**

1. **Inputs:** channels derived from `source`, `moving`, and `u_pred` (normalized consistently with
   Phase 2).
2. **Target:** `error_map` from Phase 2 (voxel units).
3. **Train / val / test:** same split convention as Phase 1 (`Train` / `Val` / `Test` folders).
4. **Eval:** held-out metrics (e.g. MAE on error map, calibration plots) and QC overlays on sample
   volumes.

The U-Net acts as an **uncertainty proxy**: it learns where the registration model is likely to be
wrong as a function of image appearance and predicted deformation, without re-running the heavy
registration network at inference.

**Status:** planned. Existing `train_unigrad_synth_unet.py` / `eval_unigrad_synth_unet.py` implement
the 2D slice pipeline; volume architecture and HCP data loaders will be added next.

**Scripts (target):**

- `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py`
- `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py`

---

## Intensity normalization and displacement (cross-phase note)

Masked z-score on `source` and `moving` does **not** change stored `u` or predicted displacement in
voxel index space — intensities are rescaled; geometry is not. Phase 2 registration and Phase 3
training both use the same normalized pair on the fixed voxel grid. See
`docs/registration-concepts.md` for φ vs `u` and backward-warp details.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | Phase 1 — synth NPZ generation |
| `experiments/synth-data-gen/torchio/visualize_synth_data.py` | Phase 1 — NPZ QC figures |
| `experiments/synth-data-gen/torchio/visualize_hcp_data.py` | Raw HCP T1w QC |
| `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` | Phase 2 — error-map generation (HCP adapt planned) |
| `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` | Phase 2 — error-map QC |
| `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` | Phase 3 — U-Net training (HCP adapt planned) |
| `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py` | Phase 3 — eval and figures |
| `docs/hcp-dataset.md` | HCP download, layout, Phase 1 detail |
| `docs/registration-concepts.md` | Registration / displacement / error-map concepts |
