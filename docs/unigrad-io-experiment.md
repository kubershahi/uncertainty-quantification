# UniGrad IO experiment — data, error-map U-Net, and runs

End-to-end notes for **3D UniGrad ICON instance optimization (IO)** on IXI, building
`error_map` training data, and training a **3D U-Net** to predict per-voxel IO error
magnitude. IO iteration sweeps are documented briefly at the end.

Artifact roots (this repo):

| Role | Path |
| --- | --- |
| IO NPZ dataset | `datasets/IXI_unigrad_io/` |
| Train / eval scripts | `experiments/regression/unigrad-io/` |
| Run outputs | `assets/runs/unigrad-io/error_unet_run{N}/` |
| Sweep figures | `assets/images/unigrad-io/3d/` |

---

## Pipeline

1. **Data** — `experiments/unigrad-io/create_unigrad_io_data.py` writes shared
   `atlas_valid_mask.npz` and per-subject `Train|Val|Test/*.npz` with `source`,
   `phi_pred`, `phi_predio`, `error_map`, `io_iterations`.
2. **Train** — `train_unigrad_io_unet.py`: 5-channel 3D U-Net → scalar `error_map`
   (masked loss inside atlas foreground).
3. **Eval** — `eval_unigrad_io_unet.py`: training curves, Test metrics JSON, QC PNGs
   (random + easy/normal/hard by mean `error_map`).

Model inputs: robust-normalized **subject**, **atlas**, **`phi_pred / phi_scale`**.
Target: **`error_map`** (voxels). Loss and metrics use **`valid_mask`** only.

---

## Training (`train_unigrad_io_unet.py`)

### Loss and metrics

- **`--loss`**: `mse` (default), `l1`, or `huber` (with **`--huber-delta`**, default `1.0` voxels).
- **`--smooth-weight`**: optional 3D TV on the prediction at **mask transitions** (default `0`).
- **Optimization**: AdamW, `ReduceLROnPlateau` on **primary val loss** matching `--loss`
  (`val_mse`, `val_l1`, or `val_huber`).
- **Checkpoint / early stop**: same primary val metric as `--loss`.
- **Val warmup**: **`--val-start-frac 0.1`** (default) — first full val at
  `ceil(0.1 × epochs)` (50 epochs → epoch 5). All three val losses are computed together
  when val runs. Set **`--val-start-frac 0`** for val from epoch 1.

### `metrics.csv` columns (current format)

`epoch, loss, train_{loss}, val_{loss}, val_mse, val_l1, val_huber, elapsed_s`

The primary `val_{loss}` is not duplicated in the matching extra column (that cell is `nan`).
W&B uses the same names (`train_l1`, `val_l1`, …).

### Example commands

Smoke / MSE baseline:

```bash
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 50 --batch-size 2 --num-workers 4 --out-dir assets/runs/unigrad-io/error_unet_run4
```

Planned **run4** (masked L1, no TV, compile, W&B):

```bash
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 50 --batch-size 2 --num-workers 4 --loss l1 --smooth-weight 0 --val-every 3 --compile --wandb --wandb-project unc-quan --wandb-run-name error_unet_run4 --out-dir assets/runs/unigrad-io/error_unet_run4
```

Use **`--val-start-epoch N`** to override warmup fraction. **`--val-every 3`** reduces val
noise and cost (as in run2).

---

## Evaluation (`eval_unigrad_io_unet.py`)

Default under **`--run-path`**:

- `training_curves.png` — `train_{loss}`, `val_{loss}`, plus the two other val metrics
- `test_metrics.json` — masked Test MSE and L1
- `test_error_pred_random.png`, `test_error_pred_easy_normal_hard.png`

```bash
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run4 --eval-dir datasets/IXI_unigrad_io --no-show
```

Curves only (no GPU Test pass):

```bash
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run4 --curves-only --no-show
```

Legacy runs (old `metrics.csv` without val warmup): hide early val spikes on the plot:

```bash
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run3 --curves-only --no-show --val-plot-min-epoch 5
```

---

## Error-map U-Net runs (1–3)

Summary of completed runs under `assets/runs/unigrad-io/`. Test metrics are **masked**
over the full Test split (115 volumes). QC figures are qualitative (one axial slice;
shared color scale can make predictions look “dim” vs peaky GT).

### Comparison

| Run | Epochs | Batch | Loss | TV (`smooth_weight`) | Val every | Best val (metric) | Test MSE | Test L1 | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| **run1** | 15 | 1 | MSE | 0.05 | 1 | ep 15: val MSE 0.709 | **0.691** | 0.566 | Short smoke; under-trained vs later runs |
| **run2** | 50 | 2 | MSE | 0.02 | 3 | ep 48: val MSE **0.514** | **0.511** | 0.478 | Main 50-epoch MSE baseline; W&B, compile |
| **run3** | 50 | 2 | MSE | **0** | 1 | ep 44: val MSE **0.518** | 0.528 | **0.474** | TV ablation; similar curves/QC to run2 |

### run1 — `error_unet_run1`

- **Config**: 15 epochs, batch 1, MSE + TV 0.05, no `torch.compile` in saved config.
- **Val at ep 15**: val MSE 0.709, val L1 0.610.
- **Test**: MSE 0.691, L1 0.566.
- **Takeaway**: Useful smoke test; not comparable to 50-epoch runs.

### run2 — `error_unet_run2`

- **Config**: 50 epochs, batch 2, MSE + TV 0.02, `val_every=3`, compile, W&B (`unc-quan`).
- **Training**: Train MSE fell steadily (~0.95 → ~0.40); val MSE plateaued ~0.51–0.53 with
  gap vs train → **overfitting** after ~epoch 35–40. Val L1 slightly below val MSE (typical
  when per-voxel errors are often &lt; 1 voxel).
- **Best checkpoint**: epoch 48 (val MSE 0.514).
- **Test**: MSE 0.511, L1 0.478 — best Test MSE of the three runs.
- **QC**: Spatial structure visible but **under-predicted peaks** (“dim” vs GT) and **sharp
  mask-boundary rims** in `error pred`; easy/normal/hard ranking still sensible.

### run3 — `error_unet_run3`

- **Config**: Same as run2 except **`smooth_weight=0`**, **`val_every=1`** (val every epoch).
- **Training**: Similar train/val trends to run2; early val spikes when plotted from epoch 1
  (run2 avoided this with `val_every=3` + first val at epoch 3).
- **Best checkpoint**: epoch 44 (val MSE 0.518).
- **Test**: MSE 0.528, L1 **0.474** (slightly better L1 than run2, slightly worse MSE).
- **Takeaway**: **TV is not the main cause** of dim peaks or rim artefacts — QC and curves
  are similar to run2 with TV off.

### Shared QC / metric interpretation

- **Val L1 &lt; val MSE** is expected when most masked residuals have |error| &lt; 1 voxel.
- **Train metric &lt; val metric** with late-epoch val stall → reduce epochs, tighter early
  stop, or change loss (MSE regresses toward conditional mean → dull peaks).
- See also `docs/unigrad-io-error-unet-next-steps.md` for mask-vs-plot notes and ablation list.

---

## Plan: run4

**Goal**: Improve **peak fidelity** and QC without abandoning the IO setup.

| Setting | Value | Rationale |
| --- | --- | --- |
| `--loss` | **`l1`** | Less mean-seeking than MSE on heavy-tailed `error_map` |
| `--smooth-weight` | **`0`** | run3 showed TV is not the main lever |
| `--val-every` | **`3`** | stabler val curves (run2) |
| `--val-start-frac` | **`0.1`** (default) | skip unstable early val |
| Epochs / batch / compile / W&B | same as run2 | comparable budget |

**Success criteria**

1. Test **L1** ≤ run2/run3 (~0.47) with **visually brighter** peak regions on QC PNGs.
2. Val **L1** used for checkpoint (automatic with `--loss l1`).
3. Re-run full eval on `best_model.pt`; compare `test_error_pred_*.png` to run2 side by side.

**Optional run5** if run4 is still too smooth: `--loss huber --huber-delta 1.0`, or weighted MSE.

Train and eval commands are in the sections above (`error_unet_run4`).

---

## IO iteration sweep (upstream)

Before fixing IO budget for `create_unigrad_io_data.py`, use
`experiments/unigrad-io/sweep_io_iterations.py` to pick iteration count `N` (LNCC elbow,
low `neg_jac_pct`, sensible `error_map` anatomy). Defaults and figure reading:

```bash
python experiments/unigrad-io/sweep_io_iterations.py --mode 3d-pkl --split Train --num-subjects 5 --save-path assets/images/unigrad-io/3d/sweep_io.png --no-show
```

Outputs: `sweep_io_<subject>_images.png`, `_curves.png`, `_metrics.csv` beside `--save-path`.
Pick `N` where LNCC plateaus, folds stay ~0, and row-4 `error_map` highlights real anatomy
(typically **40–80** on IXI). Full sweep protocol was the original content of this file;
see `reports/uniGradICON.pdf` Eq. (1) and `docs/GPU_MEMORY_OPTIMIZATIONS.md`.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/unigrad-io/create_unigrad_io_data.py` | Build `datasets/IXI_unigrad_io/` |
| `experiments/regression/unigrad-io/train_unigrad_io_unet.py` | Train error-map U-Net |
| `experiments/regression/unigrad-io/eval_unigrad_io_unet.py` | Curves + Test QC |
| `experiments/unigrad-io/visualize_unigrad_io_data.py` | NPZ QC |
| `docs/unigrad-io-error-unet-next-steps.md` | Ablations and diagnostics checklist |
