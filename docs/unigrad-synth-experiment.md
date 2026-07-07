# UniGrad Synth experiment — data, error-map U-Net, and runs

End-to-end notes for **2D synthetic registration** on IXI slices: TorchIO triplets, UniGradICON
fivers, and training a **2D U-Net** to predict per-pixel error magnitude. For 3D IO on IXI see
`docs/unigrad-io-experiment.md`. For HCP real T1w data see `docs/hcp-dataset.md`.

Artifact roots (this repo):

| Role | Path |
| --- | --- |
| IXI 2D slices | `data/IXI_2D/` (or `datasets/IXI_2D/`) |
| Synth triplets | `data/IXI_2D_synth_trip/` |
| UniGrad fivers | `data/IXI_2D_unigrad_synth_fiver/` |
| HCP T1w (real) | `datasets/hcp/` |
| Data scripts | `experiments/unigrad-synth/` |
| Train / eval | `experiments/regression/unigrad-synth/` |
| Run outputs | `assets/runs/2d/unigrad-synth/error_unet_run{N}/` |
| QC figures | `assets/images/unigrad-synth/`, `assets/images/synth/` |

---

## Pipeline

1. **Phase I — Triplets** — `create_synth_data.py`: TorchIO affine + elastic → `*_triplet.npz`
   (`image`, `warped`, `phi`, `valid_mask`, `qc_passed`).
2. **Phase II — Fivers** — `create_unigrad_synth_data.py`: UniGradICON → `phi_pred`, `error_map`
   in `*_fiver.npz`.
3. **Phase III — Train** — `train_unigrad_synth_unet.py`: 4-channel `UNet2D` → scalar `error_map`.
4. **Eval / QC** — `eval_unigrad_synth_unet.py`, `visualize_synth_data.py`, `visualize_unigrad_data.py`.

Model inputs: normalized **image**, **warped**, **`phi_pred / phi_scale`**.  
Target: **`error_map`** (pixels). Loss on **`valid_mask`** where applicable.

---

## Phase I — Synthetic triplets (`create_synth_data.py`)

### Config / QC

<!-- TODO: document affine/elastic defaults, INTERIOR_MARGIN, MAX_PHI_*, qc_passed behavior -->

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

## HCP branch (real 3D T1w)

Download and QC live under `datasets/hcp/`; visualization in `visualize_hcp_data.py`.  
Registration / fiver / U-Net on HCP volumes: **not started** — link experiments here when defined.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/unigrad-synth/create_synth_data.py` | Phase I triplets |
| `experiments/unigrad-synth/create_unigrad_synth_data.py` | Phase II fivers |
| `experiments/unigrad-synth/modify_synth_data.py` | Triplet post-processing |
| `experiments/unigrad-synth/visualize_synth_data.py` | Triplet QC |
| `experiments/unigrad-synth/visualize_unigrad_data.py` | Fiver QC |
| `experiments/unigrad-synth/visualize_hcp_data.py` | HCP T1w QC |
| `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` | Train 2D U-Net |
| `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py` | Eval + figures |
| `scripts/download_hcp.sh` | HCP S3 download |
| `docs/registration-concepts.md` | Registration vocabulary |
| `docs/hcp-dataset.md` | HCP layout and download |
