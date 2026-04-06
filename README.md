# Uncertainty Quantification for Medical Image Registration

This repository supports **ongoing work** on **uncertainty quantification for medical image registration**. The long-term aim is to relate **uncertainty estimates** to **registration quality** in settings where dense ground truth is usually unknown, so that future UQ methods can be trained, calibrated, and interpreted against meaningful error behavior.

**First step (this codebase and report):** we **study the relationship between registration error and signals we can measure or predict**. Concretely, we use **synthetic 2D slices** with known deformations, run a strong foundation registrator (**UniGradICON**), build a **dense registration error map** from predicted vs.\ true motion, and train a **U-Net** to regress that map from images plus the predicted field. That does not solve full clinical UQ by itself; it isolates how error is structured and whether an observable-driven regressor can track it when supervision is available. That is **essential groundwork** before harder real-world uncertainty models.

Implementation targets **TransMorph-style preprocessed IXI** axial slices. Phase I emits ground-truth triplets; Phase II produces **fiver** archives with `phi_pred` and scalar error; Phase III trains and evaluates the regressor. Code lives in `datahub/`; figures and an example run are under `assets/`. Large tensors (`data/`, `models/`) stay local or on Drive (see below). The PDF’s introduction should state the same bigger-picture framing: see **`reports/CSE293_introduction.tex`** to sync LaTeX with the compiled report.

| Artifact | Location |
| --- | --- |
| **Written report (PDF)** | [`reports/CSE293_Uncertainty_Estimation.pdf`](reports/CSE293_Uncertainty_Estimation.pdf); LaTeX intro snippet: [`reports/CSE293_introduction.tex`](reports/CSE293_introduction.tex) |
| **Datasets used in the paper** | [Google Drive](https://drive.google.com/drive/folders/1VYUxjbYqrMLb_KWqfJepUZN3i0jNYdU7?usp=sharing) (`IXI_2D*.zip`, `IXI_2D_synth_trip.zip`, `IXI_2D_unigrad_fiver.zip`, etc.) |

### Pipeline in three stages

1. **Phase I (synthetic ground truth):** Fixed/warped images and true displacement `phi` in `*_triplet.npz` (TorchIO-style transforms + QC).
2. **Phase II (foundation registration):** UniGradICON produces `phi_pred`; residual against channel-aligned `phi_true` yields dense **error map** fivers (`*_fiver.npz`, plus `valid_mask`, `qc_passed`, …).
3. **Phase III (supervised regression):** A 2D U-Net predicts the error map from fixed/warped intensities and `phi_pred`; `train_error_map_unet.py` / `eval_error_map_unet.py` handle training, Test/Atlas metrics, and qualitative panels.

## Repository layout

```text
.
├── datahub/              # Pipeline: synth triplets → UniGradICON fivers → train / eval / viz
├── reports/              # PDF + LaTeX intro snippet (CSE293_introduction.tex) for the write-up
├── assets/               # Figures for the report; example run under assets/runs/error_unet_run1/
├── data/                 # Local data outputs (gitignored except .gitkeep; use Drive or regenerate)
├── scripts/              # IXI 2D slicing, PKL/NIfTI helpers, registration demos
├── docs/                 # Legacy / helper documentation
├── models/               # Optional local weight checkpoints (gitignored)
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

Phase I uses **2D** slices instead of full 3D volumes. The generator `datahub/create_synth_data.py` expects `.npy` images as:

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
python datahub/create_synth_data.py \
  --input-path ./data/IXI_2D/ \
  --output-path ./data/IXI_2D_synth_trip/ \
  --workers 64
```

Near-identity resampling (optional):

```bash
python datahub/modify_synth_data.py --near-zero-keep-frac 0.10 --near-zero-eps 1e-4
```

QC:

```bash
python datahub/data_checks/check_synth_data.py --data-dir ./data/IXI_2D_synth_trip/
```

### Phase II: UniGradICON error maps (fivers)

```bash
python datahub/create_unigrad_data.py \
  --input-path ./data/IXI_2D_synth_trip/ \
  --output-path ./data/IXI_2D_unigrad_fiver/
```

Writes `*_fiver.npz` with `phi_true`, `phi_pred`, `phi_diff`, `error_map`, `valid_mask`, `qc_passed`, etc.

### Phase III: Supervised error-map training

```bash
python datahub/train_error_map_unet.py \
  --data-dir ./data/IXI_2D_unigrad_fiver/ \
  --epochs 50 \
  --batch-size 8 \
  --out-dir ./runs/error_unet_run1
```

Produces `metrics.csv`, `best_model.pt` (best **validation** masked MSE), and `run_config.json`.

### Evaluation + qualitative figures

```bash
python datahub/eval_error_map_unet.py \
  --run-path ./runs/error_unet_run1 \
  --eval-dir ./data/IXI_2D_unigrad_fiver/ \
  --no-show
```

Typical outputs under `--run-path`:

- `training_curves.png`
- `test_metrics.json`
- `test_error_pred_random.png`, `test_error_pred_minmedmax.png`
- `atlas_error_pred_random.png`, `atlas_error_pred_minmedmax.png`

The repo includes an **example** training/eval snapshot under `assets/runs/error_unet_run1/` (for figures in the report without re-running training).

### Optional visualization (no training)

- `python datahub/visualize_synth_data.py`
- `python datahub/visualize_unigrad_data.py`

## Report and figures

- **PDF:** [`reports/CSE293_Uncertainty_Estimation.pdf`](reports/CSE293_Uncertainty_Estimation.pdf)
- **Figures:** `assets/images/synthetic/`, `assets/images/unigrad/`, `assets/images/synth/`, and `assets/runs/error_unet_run1/` (training curves, Test panels, metrics JSON).

## Notes

- **Units (displacements).** Stored fields are in **pixels** on the 2D slice grid: `phi` in `*_triplet.npz`; `phi_true`, `phi_pred`, `phi_diff`, `error_map` in `*_fiver.npz`; Phase I QC limits in `datahub/create_synth_data.py`.
- **Units (internals).** TorchIO elastic uses mm only to sample the B-spline; saved `phi` is still grid-based **pixels**. UniGradICON outputs are **normalized** until `create_unigrad_data.py` rescales to **pixels**. Phase III `--phi-scale` rescales `phi_pred` **inputs** to the U-Net; fiver tensors remain in pixels.
- Run commands from the **repository root** so relative paths resolve.
- Large artifacts are not committed (`data/`, `models/`, etc.); use the [Drive folder](https://drive.google.com/drive/folders/1VYUxjbYqrMLb_KWqfJepUZN3i0jNYdU7?usp=sharing) or regenerate locally.
- Legacy helpers in `scripts/` and `docs/` may not track every `datahub/` default.

## License

MIT — see `LICENSE`.
