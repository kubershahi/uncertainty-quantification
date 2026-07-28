# Uncertainty Quantification for Medical Image Registration

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository supports **ongoing work** on **uncertainty quantification for medical image registration**. The long-term aim is to relate **uncertainty estimates** to **registration quality** in settings where dense ground truth is usually unknown, so that future UQ methods can be trained, calibrated, and interpreted against meaningful error behavior.

**Primary experiment (this codebase):** we study how **dense registration error** relates to signals available at inference. On **HCP** T1w anatomy we synthesize known deformations (TorchIO), run a foundation registrator (**UniGradICON**), form a voxel-wise **error map** ‖u_gt − u_pred‖, and train a **3D U-Net** to regress that map from images plus the predicted displacement. That does not solve clinical UQ by itself; it isolates error structure when supervision is available — essential groundwork before harder real-world uncertainty models.

**Secondary experiment:** the same error-map regression idea on **IXI** volumes, where UniGradICON **instance optimization (IO)** provides a dense proxy target (‖φ_IO − φ_pred‖) without synthetic ground truth.

Code lives under `experiments/`; concepts and run notes under `docs/`; figures and trained runs under `assets/`. Large tensors (`datasets/`, `models/`) stay local or on cluster storage.

| Artifact | Location |
| --- | --- |
| **Concepts (φ vs u, U-Net, phases)** | [`docs/registration-concepts.md`](docs/registration-concepts.md) |
| **HCP data + Phase I synth** | [`docs/hcp-dataset.md`](docs/hcp-dataset.md) |
| **Primary: UniGrad synth (HCP)** | [`docs/unigrad-synth-experiment.md`](docs/unigrad-synth-experiment.md) |
| **Secondary: UniGrad IO (IXI)** | [`docs/unigrad-io-experiment.md`](docs/unigrad-io-experiment.md) |
| **Example HCP U-Net run** | [`assets/runs/regression/unigrad-synth/hcp/error_unet_run1/`](assets/runs/regression/unigrad-synth/hcp/error_unet_run1/) |
| **Example IXI IO U-Net run** | [`assets/runs/regression/unigrad-io/error_unet_run4/`](assets/runs/regression/unigrad-io/error_unet_run4/) |
| **Written report (PDF)** | [`reports/CSE293_Uncertainty_Estimation.pdf`](reports/CSE293_Uncertainty_Estimation.pdf) |

---

## Repository layout

```text
.
├── experiments/
│   ├── synth-data-gen/torchio/     # Phase I — HCP TorchIO synth (u_gt)
│   ├── error-map-gen/
│   │   ├── unigrad-synth/          # Phase II — UniGradICON error maps (HCP)
│   │   └── unigrad-io/             # Phase II — IO error maps (IXI)
│   └── regression/
│       ├── unigrad-synth/          # Phase III — 3D U-Net (HCP)
│       └── unigrad-io/             # Phase III — 3D U-Net (IXI IO)
├── docs/                           # Dataset, experiment, and concept notes
├── assets/
│   ├── images/                     # QC / sweep figures
│   └── runs/regression/            # Trained runs (metrics, checkpoints, QC)
├── datasets/                       # Local data (gitignored)
├── scripts/                        # HCP download, registration viz helpers
├── deploy/nautilus/                # NRP Kubernetes
├── reports/                        # PDF / LaTeX
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# UniGradICON + TorchIO / SimpleITK as needed for Phase I–II (see docs)
```

Run commands from the **repository root**. Primary (HCP) defaults:

```bash
# Phase III — train (after Phase I–II data exist)
python experiments/regression/unigrad-synth/train_unigrad_synth_unet.py --data-dir datasets/error-map/unigrad-synth/hcp --out-dir assets/runs/regression/unigrad-synth/hcp/error_unet_run1 --wandb --wandb-run-name unigradsynth_unet_run1

# Phase III — eval (figures then Test metrics)
python experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py --run-path assets/runs/regression/unigrad-synth/hcp/error_unet_run1 --eval-dir datasets/error-map/unigrad-synth/hcp --mode both --no-show
```

Data download and Phase I–II commands: [`docs/hcp-dataset.md`](docs/hcp-dataset.md), [`docs/unigrad-synth-experiment.md`](docs/unigrad-synth-experiment.md).

---

## Pipeline overview

Shared three-stage pattern for both experiments:

| Stage | Intent | Typical outputs |
| --- | --- | --- |
| **I — Supervision** | Obtain a dense displacement reference (synthetic GT or IO-refined field) | NPZ with images + displacement / masks |
| **II — Foundation registration** | Run UniGradICON; form a **dense error map** vs the reference | NPZ with `u_pred` / `phi_pred` + error magnitude |
| **III — Error-map regression** | Train a 3D U-Net to predict the error map from **observable** inputs (images + predicted field) | `best_model.pt`, `metrics.csv`, QC figures, `test_metrics.json` |

Displacement convention (HCP synth): `moving(x + u(x)) ≈ source(x)` on the source/fixed lattice. Vocabulary: [`docs/registration-concepts.md`](docs/registration-concepts.md).

---

## Datasets

### Human Connectome Project (HCP) — primary

**HCP Young Adult (S1200)** minimally preprocessed T1w in **native** subject space (~0.7 mm isotropic). Used to build synthetic registration pairs with known `u_gt`. Full download layout, splits, and Phase I schema: **[`docs/hcp-dataset.md`](docs/hcp-dataset.md)**.

| Artifact | Path |
| --- | --- |
| Raw T1w | `datasets/hcp/` |
| Synth NPZ | `datasets/synth-data/torchio/hcp/` |
| Error-map NPZ | `datasets/error-map/unigrad-synth/hcp/` |

### IXI — secondary (IO track)

**IXI** 3D volumes (TransMorph-style preprocessing) for atlas–subject registration with UniGradICON **instance optimization**. Error maps measure how much IO changes the zero-shot field. Details: **[`docs/unigrad-io-experiment.md`](docs/unigrad-io-experiment.md)**.

| Artifact | Path |
| --- | --- |
| IO NPZ | `datasets/error-map/unigrad-io/ixi/` |

---

## Foundation model: UniGradICON

[UniGradICON](https://github.com/uncbiag/uniGradICON) is a foundation network for deformable registration. In this project it supplies the **predicted displacement** used both to define supervised error maps (vs TorchIO GT or vs IO) and as U-Net input channels.

- **HCP synth:** `net(moving, source)` → `u_pred` on the source lattice; target ‖u_gt − u_pred‖.
- **IXI IO:** zero-shot `phi_pred`, then per-pair IO → `phi_predio`; target ‖φ_predio − φ_pred‖ inside the atlas mask.

---

## Experiments

### 1. UniGrad synthetic experiment (HCP) — primary

**Goal.** Predict dense UniGradICON error on HCP anatomy when GT motion is known from TorchIO.

| Item | Choice |
| --- | --- |
| Anatomy | HCP T1w (native space) |
| Registration | UniGradICON zero-shot |
| Error target | ‖u_gt − u_pred‖ (voxels), masked by `source_mask` |
| Regressor | `UNet3D`, 5 in / 1 out, GroupNorm, `base_channels=16` |
| Train objective | masked MAE (default); select by val MAE |

**Stages**

1. **Synth** — TorchIO warp + identity-grid → invert → `u_gt` (`experiments/synth-data-gen/torchio/`).
2. **Error maps** — UniGradICON → `u_pred`, `u_error_map` (`experiments/error-map-gen/unigrad-synth/`).
3. **U-Net** — regress error from `[source, moving, u_pred/u_scale]` (`experiments/regression/unigrad-synth/`).

**U-Net (HCP).** Encoder–decoder with 4× pooling (pad to ÷16 inside `forward`, crop to native size). Inputs: source, moving, three `u_pred` components. Output: scalar error map. Loss/metrics inside `source_mask`.

**Results (example run `error_unet_run1`).** Artifacts: [`assets/runs/regression/unigrad-synth/hcp/error_unet_run1/`](assets/runs/regression/unigrad-synth/hcp/error_unet_run1/).

| Split | Volumes | Masked MAE | Masked RMSE | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Test | 147 | **0.353** | **0.515** | **0.865** |

QC: `training_curves.png`, `test_random_orthogonal/`, `test_mmm_orthogonal/`, `test_metrics.json`.

Full pipeline, commands, and interpretation: **[`docs/unigrad-synth-experiment.md`](docs/unigrad-synth-experiment.md)**.

---

### 2. UniGrad IO experiment (IXI) — secondary

**Goal.** Predict how much **instance optimization** changes the zero-shot field on real IXI pairs (no synthetic GT).

| Item | Choice |
| --- | --- |
| Anatomy | IXI 3D (atlas–subject) |
| Registration | UniGradICON zero-shot + IO |
| Error target | ‖φ_predio − φ_pred‖ (voxels), masked by atlas `valid_mask` |
| Regressor | `UNet3D`, 5 in / 1 out, BatchNorm, `base_channels=32` |

**Stages**

1. **IO data** — zero-shot + IO iterations → `error_map` (`experiments/error-map-gen/unigrad-io/`).
2. **U-Net** — regress error from subject, atlas, `phi_pred/phi_scale` (`experiments/regression/unigrad-io/`).

**Results (example run `error_unet_run4`, L1 + early stop).** Artifacts: [`assets/runs/regression/unigrad-io/error_unet_run4/`](assets/runs/regression/unigrad-io/error_unet_run4/).

| Split | Volumes | Masked MSE | Masked L1 |
| --- | ---: | ---: | ---: |
| Test | 115 | 0.563 | 0.479 |

Run comparison (MSE vs L1, TV ablations) and next steps: **[`docs/unigrad-io-experiment.md`](docs/unigrad-io-experiment.md)**.

---

## Notes

- Displacement / error units for both 3D tracks are **voxels** (index space).
- Prefer `argparse --help` and the linked docs over stale one-off scripts.
- Cluster / PVC setup: `deploy/nautilus/`; GPU tips: [`docs/gpu-memory-optimizations.md`](docs/gpu-memory-optimizations.md).

---

## License

MIT — see [`LICENSE`](LICENSE).
