# UniGradICON IO Sweep — experiment notes

How to read the outputs of `experiments/unigrad-io/sweep_io_iterations.py`, the
config it runs under, and the recipe for picking a non-overfitting IO
iteration count for the downstream `error_map` U-Net.

## What the experiment does

For each chosen subject, run a single instance-optimization (IO) trajectory of
UniGradICON against the fixed atlas slice (`atlas_slice_111.npy`) and snapshot
the displacement field `phi` at every iteration count given in
`--checkpoints`. The sweep gives us a way to *visualise* how `phi`, the warped
image, and the downstream regression target evolve with IO budget — without
having to restart Adam for each iteration count separately.

For each subject we emit:

| file                                       | content                                                     |
| ------------------------------------------ | ----------------------------------------------------------- |
| `sweep_io_<subject>_images.png`            | 4-row image grid (warped+grid / residual / `||phi||` / `error_map`), one column per checkpoint. |
| `sweep_io_<subject>_curves.png`            | 2-panel metric curves — quality + field health.             |
| `sweep_io_<subject>_metrics.csv`           | per-iter metrics: `iter, io_loss, lncc, mean_phi_px, mean_error_map_px, neg_jac_pct`. |

## IO config (matches the official UniGradICON protocol)

| setting              | value         | source                                                            |
| -------------------- | ------------- | ----------------------------------------------------------------- |
| optimiser            | Adam          | `icon_registration.itk_wrapper.finetune_execute`                  |
| learning rate        | `2e-5`        | `DEFAULT_FINETUNE_LEARNING_RATE`                                  |
| similarity           | LNCC          | paper section 2.3 ("We use 1 − LNCC as similarity measure")               |
| regulariser          | gradICON, λ=1.5 | paper section 2.3 / Eq. (1)                                            |
| input preprocessing  | clip 99th %ile → `[0, 1]`, resample to `175 × 175 × 175` (pseudo-volume = 5-slice replication) | paper section 2.1 |
| atlas slice          | index 111     | this repo (`ATLAS_SLICE_INDEX` in `create_unigrad_io_data.py`)   |
| LNCC window (eval)   | σ = 5  ⇒ 11-px window | `icon_registration.losses.LNCC` default; `--lncc-sigma` flag |

Memory hygiene used during IO (so a 50-iter Adam fit doesn't OOM an 11 GiB GPU):
CPU-side `state_dict` backup, `zero_grad(set_to_none=True)`, eager `del` of
optimiser + loss graph, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(set inside the script). See `docs/GPU_MEMORY_OPTIMIZATIONS.md` for the full
write-up.

## The IO loss (Eq. 1 of the paper)

The quantity Adam descends each step:

L = L_sim(I_A ∘ Φ_AB, I_B)  +  L_sim(I_B ∘ Φ_BA, I_A)  +  λ · ‖∇(Φ_AB ∘ Φ_BA − I)‖²_F

with `L_sim = 1 − LNCC`. So three pieces:

1. **Forward similarity**: warped source matches target.
2. **Backward similarity**: warped target matches source. Symmetry hedge — stops the network from over-tuning one direction.
3. **gradICON regulariser**: penalises the *gradient* of the inverse-consistency error. Soft diffeomorphism prior — forgives a globally constant translation but penalises any spatially varying inverse-error. Designed to be weak enough that a single λ = 1.5 generalises across lung CT / brain MRI / abdominal CT.

The Jacobian-determinant fold count (`neg_jac_pct`, the percent of pixels with `det J < 0` of T = id + phi) is a separate *post-hoc*
diagnostic; the loss does **not** directly penalise `det(J)`.

## How to read the figures

### `sweep_io_<subject>_images.png` — 4-row grid, one column per checkpoint

| row | what          | how to read it                                                       |
| --- | ------------- | -------------------------------------------------------------------- |
| 1   | `warped` + grid | source warped by `phi@N` with deformation grid overlay. Grid lines stay smooth = field is regular; visible folds in the grid = bad.|
| 2   | `residual`    | signed `warped@N − target` (coolwarm). Pixels saturate red/blue where alignment still fails. Should fade with N. |
| 3   | `||phi||`     | per-pixel field magnitude in pixels (viridis). Lights up where IO is moving anatomy. |
| 4   | `error_map`   | `||phi@N − phi@0||_2` per pixel (magma). **This is exactly what the downstream U-Net regresses if `--io-iterations=N` is chosen.** Should light up over real anatomical interfaces (sulci, ventricles), not isolated speckles. |

### `sweep_io_<subject>_curves.png` — 2 panels, 4 curves

**Left panel — Registration quality.**

- `LNCC` (green, ↑ better): local cross-correlation between warped 2D slice and target. Same metric the network was trained on, but evaluated externally on the saved 2D output (sigma=5, native resolution).
- `io_loss` (purple, ↓ better): the **full Eq. 1 value** — bidirectional `1 − LNCC` + 1.5 · gradICON regulariser, in 3D pseudo-volume space. Smoking-gun "is Adam descending?" check.

The two are correlated but not identical (3D vs 2D, with vs without regulariser, symmetric vs forward-only). When they agree, IO is healthy.

**Right panel — Error-map signal vs folds.**

- `mean(error_map)` (orange, signal strength): mean of `||phi@N − phi@0||_2` over pixels. By definition starts at 0 (no signal at iter 0) and grows. The U-Net sees a **bigger** regression target as N grows — but…
- `neg_jac_pct` (red, ↓ better): percentage of pixels with negative Jacobian determinant of the transform. Each non-zero pixel is a folded location (unphysical, locally non-invertible). Should stay essentially zero. If it climbs, IO is generating folds and your `error_map` will contain garbage in those regions.

## Picking the IO iteration count (recipe)

You want the smallest `N` such that:

1. `neg_jac_pct` is still negligible (≤ ~0.1%) → no/few folds.
2. `LNCC` has reached the elbow of its curve → most of the registration gain is captured.
3. `mean(error_map)` has grown to a reasonable fraction of its asymptotic value → enough signal for the U-Net to learn.
4. The row-4 `error_map` image lights up over real anatomy (cortical boundaries, ventricles), not isolated speckles.

If all four agree on the same `N` across all swept subjects, that's your IO
budget. From early runs on IXI 2D, that's typically `N = 40–80`.

Anti-patterns to watch for:

- `LNCC` flat but `mean(error_map)` and `neg_jac_pct` still climbing → IO is just adding folds without improving alignment. **Stop earlier.**
- `io_loss` falling while `LNCC` is flat → the gradICON regulariser term is dropping (more invertible) but 2D similarity isn't moving. Usually benign, but it means you're past the "useful sim improvement" regime.
- `neg_jac_pct` close to zero but `error_map` is all noise → either IO is doing essentially nothing (zero-shot already perfect for this pair) or the dynamic range is just below visibility. Check `mean_error_map_px` in the CSV.

## Reproducing

Defaults: `--checkpoints 0,50,100,150,200,250`, `--seed 42`.

```bash
python experiments/unigrad-io/sweep_io_iterations.py \
    --split Train --num-subjects 5 \
    --save-path ./assets/images/unigrad-io/sweep_io.png --no-show
```

Fixed indices instead of `--num-subjects`:

```bash
python experiments/unigrad-io/sweep_io_iterations.py \
    --split Train --subject-indices 0,17,42,93,128 \
    --save-path ./assets/images/unigrad-io/sweep_io.png --no-show
```

Outputs are `<stem>_<subject_stem>_images.png`, `_curves.png`, `_metrics.csv` beside `--save-path`.

## Related files

- `experiments/unigrad-io/sweep_io_iterations.py` — this experiment.
- `experiments/unigrad-io/create_unigrad_io_data.py` — full-dataset `error_map` generator that picks one IO iteration count based on what the sweep told you.
- `experiments/unigrad-io/visualize_unigrad_io_data.py` — visualiser for the resulting NPZs (`source / target / phi_pred / warped_pred / phi_predio / warped_predio / error_map`).
- `docs/GPU_MEMORY_OPTIMIZATIONS.md` — notes on the memory hygiene used in IO.
- `reports/uniGradICON.pdf` — Eq. (1) and section 2.3.
