# UniGrad IO error-map U-Net — what to try next

Notes after long runs (e.g. 50 epochs) when **validation curves** suggest overfitting while **test QC PNGs** still look *dim* (under-predicted peaks) and *sharp at the brain rim* (mask boundary). See `experiments/regression/unigrad-io/train_unigrad_io_unet.py` and `eval_unigrad_io_unet.py`.

## Quick diagnosis (why this is not “the whole experiment is hopeless”)

- **Masked MSE** on `error_map` encourages predictions close to the **conditional mean** under noise, which often **damps rare high-magnitude voxels** → QC can look “dim” compared to GT.
- **3D TV** in this repo is weighted so differences are penalised mainly where the **atlas mask changes** (mask boundary / transitions), not as a uniform smoother over the entire foreground. Sharp rims can still come from **mask geometry + conv behaviour + unconstrained predictions outside the mask**, not TV alone.

So the artefacts you see are **compatible with loss + model bias**, not proof that UniGrad IO cannot work.

## Things to try next (in a sensible order)

1. **Ablate TV**  
   Re-train with `--smooth-weight 0` (same data, seed, and other hparams). If dim rims and dull peaks barely change, TV is not the main lever.

2. **Align training length with validation**  
   Use **early stopping** with tighter patience, or save **multiple checkpoints** (e.g. every *k* epochs) and evaluate the one at your visual “sweet spot” if `best_model.pt` tracks a noisy val minimum too late.

3. **Loss aimed at peaks**  
   Try **masked L1**, **Huber**, or **weighted MSE** (up-weight large errors) on the error map so the model is less pulled toward the mean than with pure MSE.

4. **QC: mask the *prediction* for display**  
   For figures, multiply **predicted** error by the same `valid_mask` slice (or set out-of-mask to NaN and use masked `imshow`). Training does not supervise outside the mask, so raw pred slices can show arbitrary values there; masking pred makes the plot show only the region the loss cares about.

5. **Capacity / regularisation**  
   Slightly higher `--base-channels`, less weight decay if you add it, etc.—sometimes the bottleneck is simply under-fitting high-frequency structure.

6. **Sanity beyond PNGs**  
   On a few subjects, scatter or histogram **GT vs pred** over **masked voxels only**. If correlation is reasonable but variance of pred is too low, favour **loss / early stopping** changes over abandoning the pipeline.

---

## Training mask vs. masking on plots (especially error GT)

**During training**, the loss uses the atlas `valid_mask` so that only foreground voxels contribute (e.g. masked MSE sums `(pred − target)²` weighted by the mask and normalises by the number of masked voxels). The network is **not** asked to match GT outside that region; gradients from the regression term do not come from there.

**On a plot**, optionally applying the mask to a panel means: *show only the region where the loss was defined* (typically by setting out-of-mask pixels to a background value or NaN for transparency). That is **purely visualization**—it does **not** change the optimisation problem, saved weights, or `test_metrics.json` unless you change eval code to match.

- **Masking error pred in the plot** is useful because predictions **outside the mask are unconstrained** by the main loss; hiding them avoids misleading rim and background structure.

- **Masking error GT in the plot** is **not** the same operation as “how training uses the mask.” Training does not erase GT outside the mask—it **ignores** those voxels in the loss. If your stored `error_map` is already zero (or meaningless) outside the mask, masking GT for display changes little. If there is any non-zero leakage outside the mask, masking GT for display only **matches the figure to the supervised region**; it does **not** re-define the target or replicate “masking GT during training” as a separate training step.

**Summary:** masking on plots is for **human interpretation**; training uses the mask as a **weight on the loss**. Similar *appearance* (foreground only) does not imply the same *mechanism* as training.
