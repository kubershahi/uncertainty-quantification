# Uncertainty Quantification for Medical Image Registration

This repository contains utilities and scripts for a three-stage pipeline that learns a supervised \`error_map\` (used as an uncertainty proxy) for image registration.

The current end-to-end workflow is implemented in \`datahub/\`:

1. Phase I: synthetic ground truth registration triplets from preprocessed IXI (2D slicing).
2. Phase II: generate UniGradICON "fivers" and a scalar registration error map.
3. Phase III: train a U-Net to regress the scalar error map from observable inputs (and evaluate qualitative selections).

The \`assets/\` directory stores generated figures (used by your report, including the LaTeX methodology section).

## Repository layout

```text
.
├── datahub/            # Main pipeline: synthetic data -> UniGradICON -> training/eval/visualization
├── assets/            # Figures and LaTeX snippets for the report
├── data/              # Data outputs (ignored from git; large)
├── scripts/           # Auxiliary utilities for IXI data handling and registration visualization
├── docs/              # Documentation for legacy conversion / visualization helpers
├── models/            # Pretrained weights of models (ignored from git; large)
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

The pipeline assumes your local dataset folders match the defaults used by the scripts (edit paths as needed):

### Phase I: Ground truth generation (synthetic triplets)

To mitigate computational and storage limitations, Phase I utilizes a 2D slice representation instead of the original 3D TransMorph preprocessed IXI volumes. The Phase~I synthetic triplet generator in `datahub/create_synth_data.py` expects a 2D dataset of `.npy` images organized as:

```text
./data/IXI_2D/
  Train/
  Val/
  Test/
  Atlas/
```

This 2D dataset is created from the TransMorph preprocessed 3D IXI PKL files using:

```bash
python scripts/create_ixi_2d.py
```

Notes:

- `scripts/create_ixi_2d.py` slices the middle axial region of each 3D volume (default: `slices_per_volume=10`) and saves each slice as a separate `.npy`.
- The script uses hardcoded paths (`SOURCE_DIR`, `TARGET_DIR`, and `ATLAS_PATH`) near the bottom of the file. Update them if your data lives elsewhere.
- By default, it writes to `./data/raw/IXI_2D/`; you can either point `datahub/create_synth_data.py` to that folder, or move/copy into `./data/IXI_2D/` to match the README defaults.

Optional sanity check:

```bash
python scripts/visualize_ixi_2d.py --input-dir ./data/IXI_2D --recursive --no-show --save-path ./assets/images/ixi_2d.png
```

Create synthetic $(I_{fixed}, I_{warped}, \phi_{true})$ triplets (stored as \`\*\_triplet.npz\`) for Train/Val/Test/Atlas:

```bash
python datahub/create_synth_data.py \
  --input-path ./data/IXI_2D/ \
  --output-path ./data/IXI_2D_synth_trip/ \
  --workers 64
```

If you need to control the fraction of near-identity deformation fields, use:

```bash
python datahub/modify_synth_data.py --near-zero-keep-frac 0.10 --near-zero-eps 1e-4
```

QC stats are produced via:

```bash
python datahub/data_checks/check_synth_data.py --data-dir ./data/IXI_2D_synth_trip/
```

### Phase II: UniGradICON error map generation (fivers)

Generate \`\*\_fiver.npz\` files (containing \`phi_true\`, \`phi_pred\`, \`error_map\`, \`valid_mask\`, etc.):

```bash
python datahub/create_unigrad_data.py \
  --input-path ./data/IXI_2D_synth_trip/ \
  --output-path ./data/IXI_2D_unigrad_fiver/
```

### Phase III: Supervised error-map training

Train the 2D error-regression U-Net:

```bash
python datahub/train_error_map_unet.py \
  --data-dir ./data/IXI_2D_unigrad_fiver/ \
  --epochs 50 \
  --batch-size 8 \
  --out-dir ./runs/error_unet_run1
```

This writes \`metrics.csv\` (per-epoch metrics) and \`best_model.pt\` (selected by best validation masked MSE).

### Evaluation + qualitative figures

Run evaluation on Test and Atlas splits:

```bash
python datahub/eval_error_map_unet.py \
  --run-path ./runs/error_unet_run1 \
  --eval-dir ./data/IXI_2D_unigrad_fiver/ \
  --no-show
```

Qualitative outputs are written under the run directory (\`--run-path\`), including:

- \`training_curves.png\`
- \`test_metrics.json\`
- \`test_error_pred_random.png\` / \`test_error_pred_minmedmax.png\`
- \`atlas_error_pred_random.png\` / \`atlas_error_pred_minmedmax.png\`

### Optional qualitative visualizations

Use the visualization helpers to inspect triplets and fivers without training:

- \`python datahub/visualize_synth_data.py\`
- \`python datahub/visualize_unigrad_data.py\`

## Report assets

Your LaTeX methodology section is provided in:

- \`assets/methodology_three_stage_workflow.tex\`

Figures used for that section are in:

- \`assets/images/synthetic/\`
- \`assets/images/unigrad/\`
- \`assets/images/synth/\`

## Notes

- Run from repository root for consistent relative paths.
- Large outputs are intentionally not committed (see \`.gitignore\`).
- Legacy utilities in \`scripts/\` and legacy docs in \`docs/\` may not apply to the current \`datahub/\` training/evaluation pipeline.

## License

MIT — see \`LICENSE\`.
