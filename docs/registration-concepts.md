# Registration concepts — project vocabulary and pipelines

Concepts needed to follow **synthetic 2D**, **3D instance optimization (IO)**, and **HCP real-data**
work in this repository. For dataset specifics see `docs/hcp-dataset.md`, `docs/unigrad-synth-experiment.md`,
and `docs/unigrad-io-experiment.md`.

---

## Project goal (one paragraph)

We study **dense registration error** — how wrong a predicted deformation is at each voxel/pixel —
and whether a **U-Net** can predict that error from **observable inputs** (images + predicted
displacement field). When ground truth motion is known (synth) or approximated (IO), we build a
supervised **error map** and train a regressor. That is groundwork for later **uncertainty
quantification** when dense GT is unavailable.

---

## Three data tracks in this repo

```text
                    ┌─────────────────────────────────────┐
                    │  Uncertainty / error-map U-Net      │
                    └─────────────────────────────────────┘
                                      ▲
                    ┌─────────────────┴─────────────────┐
                    │         error_map (supervision)    │
                    └─────────────────┬─────────────────┘
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
   Phase I (GT motion)        UniGradICON zero-shot          (planned)
          │                           │                           │
   IXI 2D TorchIO              IXI 3D IO budget              HCP T1w 3D
   phi_true known              phi_predio ≈ GT               native volumes
          │                           │                           │
   experiments/unigrad-synth/   experiments/unigrad-io/      datasets/hcp/
```

| Track | Dim | Moving / fixed | GT displacement | Error map |
| --- | ---: | --- | --- | --- |
| **Synth** | 2D | `image` → `warped` | `phi_true` in `*_triplet.npz` | ‖φ_pred − φ_true‖ per pixel |
| **IO** | 3D | `source` → `atlas` | `phi_predio` (IO-refined) | ‖φ_predio − φ_pred‖ per voxel |
| **HCP** | 3D | TBD | TBD | TBD |

**Regression scripts:**

- 2D: `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` (`UNet2D`, 4 inputs)
- 3D: `experiments/regression/unigrad-io/train_unigrad_io_unet.py` (`UNet3D`, 5 inputs)

---

## End-to-end pipeline (synth track)

```text
Native MRI slice (IXI_2D)
      ↓
Synthetic deformation (TorchIO)     ← Phase I: create_synth_data.py
      ↓
Warped MRI + phi_true (*_triplet.npz)
      ↓
UniGradICON zero-shot                 ← Phase II: create_unigrad_synth_data.py
      ↓
phi_pred, error_map (*_fiver.npz)
      ↓
U-Net regression                    ← Phase III: train_unigrad_synth_unet.py
      ↓
Predicted error_map
```

**IO track** replaces TorchIO GT with ICON **instance optimization** on real 3D pairs; see
`docs/unigrad-io-experiment.md`.

---

## Coordinate spaces

### Native space (HCP `T1w/`, IXI volumes before MNI)

Each subject stays in its own anatomical frame after minimal preprocessing. **Nonlinear inter-subject
registration is still an open problem** — suitable for learning registration and error structure.

### MNI / template space (HCP `MNINonLinear/`)

Brains warped to a common template. Useful for group studies but **reduces anatomical variability**.
This project **does not** use MNINonLinear for HCP downloads.

### AC-PC alignment

Rigid rotation/translation so the AC–PC axis is standardized. **Not** nonlinear registration.
HCP `T1w/` volumes are AC-PC aligned and distortion corrected but remain in **native** space.

---

## Displacement vs deformation

For voxel/pixel position **x**:

- **Displacement** **u(x)** = (dx, dy[, dz]) — how far the point moves
- **Deformation** **φ(x)** = **x + u(x)** — where the output grid samples from

UniGradICON may output a **position map** (φ); scripts convert to pixel/voxel displacement by
subtracting the identity grid and scaling (see `create_unigrad_synth_data.py`).

---

## Backward (pull) warping

Registration warps the **moving** image onto the **fixed** grid:

1. For each output voxel, compute source coordinate φ(x)
2. Sample moving image at φ(x) (interpolation)
3. Write intensity to output voxel

Non-integer φ(x) requires **interpolation** (trilinear for MRI intensity; nearest-neighbor for labels).

---

## Ground truth and error map (this project)

### Synth (2D)

| Field | Meaning |
| --- | --- |
| `phi_true` | Known synthetic displacement (TorchIO), **pixels** |
| `phi_pred` | UniGradICON zero-shot prediction, **pixels** |
| `phi_diff` | φ_pred − φ_true (component-wise) |
| `error_map` | Per-pixel ‖φ_pred − φ_true‖₂ (scalar magnitude) |

Stored in `*_fiver.npz` under `Train|Val|Test/` (`create_unigrad_synth_data.py`).

### IO (3D)

| Field | Meaning |
| --- | --- |
| `phi_pred` | Zero-shot ICON displacement |
| `phi_predio` | After **instance optimization** (IO iterations) |
| `error_map` | Per-voxel ‖φ_predio − φ_pred‖₂, **voxels** |

Only voxels inside **`valid_mask`** (atlas foreground) are used for loss and metrics. See
`create_unigrad_io_data.py`.

### What the U-Net predicts

The network does **not** predict φ directly. It predicts **error_map** (or its magnitude) from:

| Track | Typical inputs |
| --- | --- |
| Synth 2D | normalized `image`, `warped`, `phi_pred / phi_scale` |
| IO 3D | normalized `source`, `atlas`, `phi_pred / phi_scale` |

`phi_scale` (default 64) puts displacement channels on a similar numeric scale as intensities.

---

## Zero-shot vs instance optimization (IO)

| Stage | Description |
| --- | --- |
| **Zero-shot** | Single forward pass of pretrained UniGradICON → `phi_pred` |
| **IO** | Extra gradient steps on the pair → `phi_predio`, lower LNCC / better alignment |
| **Error map** | Measures **how much IO improved** over zero-shot (per voxel) |

IO iteration count is chosen via `sweep_io_iterations.py` before building `datasets/IXI_unigrad_io/`.

---

## Units and masks

| Context | Displacement / error units |
| --- | --- |
| IXI 2D synth | **pixels** on the slice grid |
| IXI 3D IO | **voxels** in volume index space |
| HCP (future) | **voxels** (NIfTI grid) |

**Masks:**

- Synth: `valid_mask` — interior margin away from slice boundary (`INTERIOR_MARGIN` in `create_synth_data.py`)
- IO: `valid_mask` — shared atlas foreground (`atlas_valid_mask.npz`)
- HCP: `brainmask_fs.nii.gz` for brain-only evaluation (download + QC)

**QC flags:** `qc_passed` on triplets/fivers; synth Phase II skips failed triplets by default.

---

## Key NPZ / file schemas (quick reference)

**`*_triplet.npz` (Phase I):** `image`, `warped`, `phi` (2, H, W), optional `valid_mask`, `qc_passed`

**`*_fiver.npz` (Phase II synth):** `image`, `warped`, `phi_true`, `phi_pred`, `phi_diff`, `error_map`, `valid_mask`, `qc_passed`

**IO `*.npz` (Phase II 3D):** `source`, `phi_pred`, `phi_predio`, `error_map`, `io_iterations`, …

---

## Related files

| File | Role |
| --- | --- |
| `experiments/unigrad-synth/create_synth_data.py` | Phase I triplets |
| `experiments/unigrad-synth/create_unigrad_synth_data.py` | Phase II fivers |
| `experiments/unigrad-io/create_unigrad_io_data.py` | Phase II IO volumes |
| `experiments/regression/unigrad-synth/` | 2D error-map U-Net |
| `experiments/regression/unigrad-io/` | 3D error-map U-Net |
| `docs/hcp-dataset.md` | HCP download and layout |
| `reports/uniGradICON.pdf` | ICON / IO equations |
