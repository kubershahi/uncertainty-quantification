# Uncertainty Quantification for Medical Image Registration

This repository supports **ongoing work** on **uncertainty quantification for medical image registration**. The long-term aim is to relate **uncertainty estimates** to **registration quality** in settings where dense ground truth is usually unknown, so that future UQ methods can be trained, calibrated, and interpreted against meaningful error behavior.

**First step (this codebase and report):** we **study the relationship between registration error and signals we can measure or predict**. Concretely, we use **synthetic 2D slices** with known deformations, run a strong foundation registrator (**UniGradICON**), build a **dense registration error map** from predicted vs.\ true motion, and train a **U-Net** to regress that map from images plus the predicted field. That does not solve full clinical UQ by itself; it isolates how error is structured and whether an observable-driven regressor can track it when supervision is available. That is **essential groundwork** before harder real-world uncertainty models.

Implementation targets **TransMorph-style preprocessed IXI** axial slices. Phase I emits ground-truth triplets; Phase II produces **fiver** archives with `phi_pred` and scalar error; Phase III trains and evaluates the regressor. Code lives in `experiments/`; figures and an example run are under `assets/`. Large tensors (`data/`, `models/`) stay local or on Drive (see below). The PDF’s introduction should state the same bigger-picture framing: see **`reports/CSE293_introduction.tex`** to sync LaTeX with the compiled report.

| Artifact | Location |
| --- | --- |
| **Written report (PDF)** | [`reports/CSE293_Uncertainty_Estimation.pdf`](reports/CSE293_Uncertainty_Estimation.pdf) |
| **Datasets used in the paper** | [Google Drive](https://drive.google.com/drive/folders/1VYUxjbYqrMLb_KWqfJepUZN3i0jNYdU7?usp=sharing) (`IXI_2D*.zip`, `IXI_2D_synth_trip.zip`, `IXI_2D_unigrad_synth_fiver.zip`, etc.) |

### Pipeline in three stages

1. **Phase I (synthetic ground truth):** Fixed/warped images and true displacement `phi` in `*_triplet.npz` (TorchIO-style transforms + QC).
2. **Phase II (foundation registration):** UniGradICON produces `phi_pred`; residual against channel-aligned `phi_true` yields dense **error map** fivers (`*_fiver.npz`, plus `valid_mask`, `qc_passed`, …).
3. **Phase III (supervised regression):** A U-Net predicts the dense error map from registration inputs. **2D synth** (`data/IXI_2D_unigrad_synth_fiver`, 4-channel `UNet2D`): `experiments/regression/unigrad-synth/`. **3D IO** (`datasets/IXI_unigrad_io`, 5-channel `UNet3D`): `experiments/regression/unigrad-io/`.

### 3D U-Net architecture (`UNet3D`)

Implemented in `experiments/regression/unigrad-io/train_unigrad_io_unet.py` as a standard **encoder–decoder U-Net** on 3D volumes (layout `N × C × D × H × W`).

| | |
| --- | --- |
| **Input** | 5 channels: robust-normalized **subject** (`source`), **atlas** (`target`), **φ_pred** (3 channels, divided by `--phi-scale`, default 64 voxel units) |
| **Output** | 1 channel: predicted **error_map** — per-voxel ‖φ_IO − φ_pred‖₂ (voxels), same shape as the label |
| **Default width** | `--base-channels 32` (denoted `b` below) |

**Encoder (contracting path):** four `MaxPool3d(2)` steps after double-conv blocks. Each block is **DoubleConv3d**: `Conv3d 3×3×3 → BatchNorm3d → ReLU → Conv3d 3×3×3 → BatchNorm3d → ReLU` (no bias in convolutions).

| Stage | Channels (out) | Spatial size (per axis, relative to input) |
| --- | --- | --- |
| `down1` | `b` | 1× |
| `down2` | `2b` | ½× |
| `down3` | `4b` | ¼× |
| `down4` | `8b` | ⅛× |
| `bot` (bottleneck) | `16b` | ¹⁄₁₆× |

**Decoder (expanding path):** four upsampling steps with **skip connections** (channel-wise `concat` with the matching encoder feature map, then DoubleConv3d):

- `ConvTranspose3d` kernel 2, stride 2: `16b → 8b`, concat with `c4` → `conv4` (`16b → 8b`)
- same pattern: `8b → 4b` (+ `c3`), `4b → 2b` (+ `c2`), `2b → b` (+ `c1`)
- **Head:** `Conv3d 1×1×1`: `b → 1` (scalar error_map per voxel)

With default `b = 32`, the channel ladder is **32 → 64 → 128 → 256 → 512** in the encoder and the reverse in the decoder. Depth must be divisible by 16 along D, H, and W (four poolings); IXI volumes at ~160×192×224 satisfy this.

**Training:** volume MSE over all voxels; optional `--smooth-weight` adds 3D edge total-variation on the prediction. Optimizer: AdamW; mixed precision on CUDA when enabled.

## Repository layout

```text
.
├── experiments/           # Pipeline: synth → UniGradICON → train / eval / viz
├── deploy/nautilus/       # NRP Kubernetes (PVC, GPU pods, Job)
├── reports/               # PDF + LaTeX intro (CSE293_introduction.tex)
├── assets/                # Report figures; example run under assets/runs/error_unet_run1/
├── data/                  # Local tensors (gitignored; use Drive or regenerate)
├── scripts/               # IXI 2D slicing, PKL/NIfTI helpers
├── docs/                  # See docs/NAUTILUS.md (cluster runbook), GPU_MEMORY_OPTIMIZATIONS.md, unigrad-io-experiment.md
├── models/                # Optional local checkpoints (gitignored)
├── requirements.txt
└── README.md
```

## Quick start (end-to-end)

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pipeline assumes local dataset folders match the defaults below (or download the matching archives from [Drive](https://drive.google.com/drive/folders/1VYUxjbYqrMLb_KWqfJepUZN3i0jNYdU7?usp=sharing) and extract under `./data/`).

### Phase I: Ground truth generation (synthetic triplets)

Phase I uses **2D** slices instead of full 3D volumes. The generator `experiments/synthetic/create_synth_data.py` expects `.npy` images as:

```text
./data/IXI_2D/
  Train/
  Val/
  Test/
  Atlas/
```

Build that layout from TransMorph preprocessed 3D IXI PKL files:

```bash
python scripts/create_ixi_2d.py
```

Notes:

- `scripts/create_ixi_2d.py` slices a mid-axial band (default `slices_per_volume=10`) per volume.
- Paths `SOURCE_DIR`, `TARGET_DIR`, and `ATLAS_PATH` are near the bottom of the script—edit for your machine.
- Default output is `./data/raw/IXI_2D/`; align with `./data/IXI_2D/` or pass `--input-path` to Phase I.

Optional sanity check:

```bash
python scripts/visualize_ixi_2d.py --input-dir ./data/IXI_2D --recursive --no-show --save-path ./assets/images/ixi_2d.png
```

Create synthetic triplets (`I_fixed`, `I_warped`, `phi`) as `*_triplet.npz`:

```bash
python experiments/synthetic/create_synth_data.py --input-path ./data/IXI_2D/ --output-path ./data/IXI_2D_synth_trip/ --workers 64
```

Near-identity resampling (optional):

```bash
python experiments/synthetic/modify_synth_data.py --near-zero-keep-frac 0.10 --near-zero-eps 1e-4
```

QC:

```bash
python experiments/data_checks/check_synth_data.py --data-dir ./data/IXI_2D_synth_trip/
```

### Phase II: UniGradICON error maps (fivers)

```bash
python experiments/synthetic/create_unigrad_synth_data.py --input-path ./data/IXI_2D_synth_trip/ --output-path ./data/IXI_2D_unigrad_synth_fiver/
```

Writes `*_fiver.npz` with `phi_true`, `phi_pred`, `phi_diff`, `error_map`, `valid_mask`, `qc_passed`, etc.

### Phase III: Supervised error-map training (3D IO volumes)

Data from `experiments/unigrad-io/create_unigrad_io_data.py` under `datasets/IXI_unigrad_io/` (`Train|Val|Test/*.npz`). Each file: `source` (subject), `target` (same atlas for all subjects), `phi_pred`, `error_map`.

```bash
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 50 --batch-size 1 --out-dir assets/runs/3d/unigrad-io/error_unet_run1
```

Produces `metrics.csv`, `best_model.pt` (best validation MSE), and `run_config.json`.

### Evaluation + qualitative figures (3D IO)

```bash
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/3d/unigrad-io/error_unet_run1 --eval-dir datasets/IXI_unigrad_io --no-show
```

Typical outputs under `--run-path`:

- `training_curves.png`
- `test_metrics.json`
- `test_error_pred_random.png`
- `test_error_pred_easy_normal_hard.png`

### Phase III: 2D synthetic fiver training

```bash
python experiments/regression/unigrad-synth/train_unigrad_synth_unet.py --data-dir data/IXI_2D_unigrad_synth_fiver --epochs 50 --batch-size 8 --out-dir assets/runs/2d/unigrad-synth/error_unet_run1
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/2d/unigrad-synth/error_unet_run1 --eval-dir data/IXI_2D_unigrad_synth_fiver --no-show
```

Example 2D run artifacts: `assets/runs/2d/unigrad-synth/error_unet_run1/`.

### Optional visualization (no training)

- `python experiments/synthetic/visualize_synth_data.py`
- `python experiments/synthetic/visualize_unigrad_data.py`
- `python experiments/unigrad-io/visualize_unigrad_io_data.py`

## Report and figures

- **PDF:** [`reports/CSE293_Uncertainty_Estimation.pdf`](reports/CSE293_Uncertainty_Estimation.pdf)
- **Figures:** `assets/images/synthetic/`, `assets/images/unigrad-synth/`, `assets/images/unigrad-io/`, `assets/images/synth/`, and `assets/runs/error_unet_run1/` (training curves, Test panels, metrics JSON).

## Notes

- **Units (displacements).** Stored fields are in **pixels** on the 2D slice grid: `phi` in `*_triplet.npz`; `phi_true`, `phi_pred`, `phi_diff`, `error_map` in `*_fiver.npz`; Phase I QC limits in `experiments/synthetic/create_synth_data.py`.
- **Units (internals).** TorchIO elastic uses mm only to sample the B-spline; saved `phi` is still grid-based **pixels**. UniGradICON outputs are **normalized** until `create_unigrad_synth_data.py` rescales to **pixels**. Phase III `--phi-scale` rescales `phi_pred` **inputs** to the U-Net; fiver tensors remain in pixels.
- Run commands from the **repository root** so relative paths resolve.
- Large artifacts are not committed (`data/`, `models/`, etc.); use the [Drive folder](https://drive.google.com/drive/folders/1VYUxjbYqrMLb_KWqfJepUZN3i0jNYdU7?usp=sharing) or regenerate locally.
- `scripts/` and older docs may lag `experiments/` defaults; prefer argparse `--help` when in doubt.

## License

MIT — see `LICENSE`.
