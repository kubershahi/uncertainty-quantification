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
- **Optimization**: AdamW, `ReduceLROnPlateau` on **`val_{loss}`** (matches `--loss`).
- **Checkpoint / early stop**: same **`val_{loss}`** as `--loss`.
- **Early stop**: **`--early-stop-min-delta`** (default `0.005`) — improvements smaller than this
  do not reset patience (used on run4; stopped at epoch 54 with best val at epoch 46).
- **Val warmup**: **`--val-start-frac 0.1`** (default) — first val at `ceil(0.1 × epochs)`.
  Set **`--val-start-frac 0`** for val from epoch 1.

### `metrics.csv` columns (current format)

`epoch, loss, train_{loss}, val_{loss}, elapsed_s`

W&B logs the same two loss columns plus `lr` and `elapsed_s`. Eval curves plot only
**`train_{loss}`** and **`val_{loss}`** (legacy CSVs with `val_mse` / `val_l1` / `val_huber`
still load for old runs).

### Example commands

MSE baseline (run2-style):

```bash
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 50 --batch-size 2 --loss mse --val-every 3 --compile --wandb --wandb-project unc-quan --wandb-run-name error_unet_run2 --out-dir assets/runs/unigrad-io/error_unet_run2
```

**run4b** (L1 re-run: meaningful early stop, val every 3):

```bash
python experiments/regression/unigrad-io/train_unigrad_io_unet.py --data-dir datasets/IXI_unigrad_io --epochs 60 --batch-size 2 --loss l1 --val-every 3 --early-stop-patience 8 --early-stop-min-delta 0.005 --compile --wandb --wandb-project unc-quan --wandb-run-name error_unet_run4b --out-dir assets/runs/unigrad-io/error_unet_run4b
```

Use **`--val-start-epoch N`** to override warmup fraction.

---

## Evaluation (`eval_unigrad_io_unet.py`)

Default under **`--run-path`**:

- `training_curves.png` — `train_{loss}` and `val_{loss}` only
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

## Model architecture (runs 1–4, unchanged)

Implementation: **`UNet3D`** in `train_unigrad_io_unet.py` (`--base-channels`, default **32**).

| Item | Setting |
| --- | --- |
| Inputs (5 ch) | Robust-normalized **subject**, **atlas**, **`phi_pred / phi_scale`** (3 displacement components) |
| Output (1 ch) | **`error_map`** magnitude (voxels), trained only inside **`valid_mask`** |
| Encoder | 4 × `DoubleConv3d` (3×3×3 conv → BN → ReLU ×2) + `MaxPool3d(2)`; channels **32 → 64 → 128 → 256** |
| Bottleneck | `DoubleConv3d` at **512** channels (`base × 16`) |
| Decoder | 4 × `ConvTranspose3d(2, stride=2)` upsample + skip concat + `DoubleConv3d` |
| Head | `Conv3d(base, 1, kernel_size=1)` |
| Regularization | Optional 3D TV on pred at mask edges (`--smooth-weight`; **0** from run3 onward) |

**Runs 1–4** all use this same U-Net; only loss, TV, epochs, and training schedule differ.
**Planned run5+** (if run4 QC still too smooth): document-only ideas below — wider base
(`--base-channels 48`), dropout in `DoubleConv3d`, or shallow extra conv head before `out`;
implement only after run4b QC.

---

## Error-map U-Net runs (1–4)

Summary of completed runs under `assets/runs/unigrad-io/`. Test metrics are **masked**
over the full Test split (115 volumes). QC figures are qualitative (one axial slice;
shared color scale can make predictions look “dim” vs peaky GT).

### Comparison

| Run | Epochs | Batch | Loss | TV (`smooth_weight`) | Val every | Best val (metric) | Test MSE | Test L1 | Notes |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | --- |
| **run1** | 15 | 1 | MSE | 0.05 | 1 | ep 15: val MSE 0.709 | **0.691** | 0.566 | Short smoke; under-trained vs later runs |
| **run2** | 50 | 2 | MSE | 0.02 | 3 | ep 48: val MSE **0.514** | **0.511** | 0.478 | Main 50-epoch MSE baseline; W&B, compile |
| **run3** | 50 | 2 | MSE | **0** | 1 | ep 44: val MSE **0.518** | 0.528 | **0.474** | TV ablation; similar curves/QC to run2 |
| **run4** | 54 (of 60) | 2 | **L1** | **0** | 1 | ep 46: val L1 **0.469** | 0.563 | 0.479 | L1 + early stop; QC ≈ run3 |

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

### run4 — `error_unet_run4` (L1 + early stop)

Artifacts: `assets/runs/unigrad-io/error_unet_run4/` (`training_curves.png`,
`test_error_pred_random.png`, `test_error_pred_easy_normal_hard.png`, `test_metrics.json`).

- **Config**: max 60 epochs, batch 2, **`--loss l1`**, TV 0, `val_every=1`, val from epoch 6,
  **`--early-stop-patience 8`**, **`--early-stop-min-delta 0.005`**, compile, W&B (`unc-quan`).
  Same **`UNet3D`** as runs 1–3 (`base_channels=32`).
- **Training** (see `metrics.csv`, two-curve plot):
  - **`train_l1`**: ~0.62 → **0.37** by epoch 54 (steady drop).
  - **`val_l1`**: noisy; best **0.469** at **epoch 46**; plateaus ~0.47–0.48 after ~epoch 40.
  - **Early stop** fired at **epoch 54** (8 val checks without improvement > 0.005 after the ep-46 best).
  - Train/val gap widens after ~epoch 40 → same overfitting pattern as MSE runs, but stop lands near the val knee.
- **Best checkpoint**: epoch **46** (`val_l1` **0.469**).
- **Test** (masked, 115 volumes, `test_metrics.json`): MSE **0.563**, L1 **0.479**.
- **vs run3** (MSE, TV 0): Test L1 **0.474**, MSE **0.528**. Run4 is **marginally worse** on
  aggregate Test L1/MSE despite a slightly lower best **val** L1; **QC is largely unchanged** vs run3
  (still smooth/blurred `error pred`, correct easy/normal/hard ranking; hard row may look slightly
  brighter — worth a side-by-side with run3 PNGs).
- **Takeaway**: **L1 alone** did not fix peak blur or mask rims. Early stop + min-delta worked as
  intended (54 epochs, best ~46). Next lever is likely **architecture** (run5) or **`val_every=3`**
  (run4b) rather than another long L1 run with the same schedule.

---

## Plan: run4b (optional — stabler val)

Only if you want to re-run L1 with **`--val-every 3`** (less noisy val curve, as in run2) while
keeping the same early-stop settings. Run4 already used min-delta early stop successfully.

| Setting | Value | Rationale |
| --- | --- | --- |
| `--loss` | **`l1`** | Same as run4 |
| `--val-every` | **`3`** | Fewer val spikes on curves |
| `--early-stop-min-delta` | **`0.005`** | Same as run4 |
| `--early-stop-patience` | **`8`** | Same as run4 |

```bash
python experiments/regression/unigrad-io/eval_unigrad_io_unet.py --run-path assets/runs/unigrad-io/error_unet_run4 --eval-dir datasets/IXI_unigrad_io --no-show
```

---

## Plan: run5 (architecture — not implemented)

Only if run4 (or run4b) QC is still peak-blurred:

| Change | Flag / code | Notes |
| --- | --- | --- |
| Wider U-Net | `--base-channels 48` | More capacity for localized errors |
| Dropout | `DoubleConv3d` p=0.1–0.2 | Reduce overfitting on train L1 |
| Huber loss | `--loss huber --huber-delta 1.0` | Between L1 and MSE on heavy tails |

Keep inputs/target/mask pipeline unchanged.

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
see `reports/uniGradICON.pdf` Eq. (1) and `docs/gpu-memory-optimizations.md`.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/unigrad-io/create_unigrad_io_data.py` | Build `datasets/IXI_unigrad_io/` |
| `experiments/regression/unigrad-io/train_unigrad_io_unet.py` | Train error-map U-Net |
| `experiments/regression/unigrad-io/eval_unigrad_io_unet.py` | Curves + Test QC |
| `experiments/unigrad-io/visualize_unigrad_io_data.py` | NPZ QC |
| `docs/unigrad-io-error-unet-next-steps.md` | Ablations and diagnostics checklist |
