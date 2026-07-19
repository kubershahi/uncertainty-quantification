# Registration concepts — project vocabulary and pipelines

Concepts for the **3D** tracks in this repository: **HCP synth** (TorchIO GT + UniGradICON
error maps + U-Net regression) and **IXI IO** (instance optimization). For dataset specifics see
`docs/hcp-dataset.md`, `docs/unigrad-synth-experiment.md`, and `docs/unigrad-io-experiment.md`.

---

## Project goal (one paragraph)

We study **dense registration error** — how wrong a predicted deformation is at each voxel —
and whether a **3D U-Net** can predict that error from **observable inputs** (images + predicted
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
   HCP TorchIO GT              UniGradICON IO              HCP T1w (raw)
   u_gt known                  phi_predio ≈ GT             native volumes
          │                           │                           │
   synth-data-gen/torchio/     error-map-gen/unigrad-io/   datasets/hcp/
   + error-map-gen/unigrad-synth/ + regression/unigrad-io/
```

| Track | Dim | Moving / fixed | GT displacement | Error map |
| --- | ---: | --- | --- | --- |
| **Synth (HCP 3D)** | 3D | `source` → `moving` | `u_gt` in synth NPZ (source lattice) | ‖u_gt − u_pred‖ (`create_unigrad_synth_data.py`) |
| **IO (IXI 3D)** | 3D | `source` → `atlas` | `phi_predio` (IO-refined) | ‖φ_predio − φ_pred‖ |
| **HCP raw** | 3D | T1w NIfTI | — | QC / future registration |

**Regression scripts (both 3D U-Net, 5 input channels):**

- HCP synth: `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py`
- IXI IO: `experiments/regression/unigrad-io/train_unigrad_io_unet.py`

---

## Shared vocabulary (all phases)

### Coordinate spaces

**Native space (HCP `T1w/`, IXI volumes before MNI).** Each subject stays in its own anatomical
frame after minimal preprocessing. Nonlinear inter-subject registration remains an open problem —
suitable for learning registration and error structure.

**MNI / template space (HCP `MNINonLinear/`).** Brains warped to a common template. Useful for group
studies but reduces anatomical variability. This project **does not** use MNINonLinear for HCP
downloads.

**AC-PC alignment.** Rigid rotation/translation so the AC–PC axis is standardized — **not**
nonlinear registration. HCP `T1w/` volumes are AC-PC aligned and distortion corrected but remain in
**native** space.

### Displacement **u** vs position map **φ**

Registration stores geometry on the **fixed** (output) grid. At each fixed-grid location **x**:

| Symbol | Name in this repo | What it stores | Shape (3D) |
| --- | --- | --- | --- |
| **u(x)** | displacement / **u vectors** | offset **(dx, dy, dz)** in voxels | `(3, X, Y, Z)` |
| **φ(x)** | **position map** (not displacement) | absolute sample coordinate **x + u(x)** | `(3, X, Y, Z)` |

```text
φ(x) = x + u(x)
u(x) = φ(x) − x
```

For output-grid voxel **x** on the **fixed** image:

- **u(x)** is the **offset** into the **moving** volume.
- Sample **moving(x + u(x))** to get the anatomy that should match **fixed(x)**.

```text
registered_moving(x) ≈ moving(x + u(x)) ≈ fixed(x)
```

**Example.** Fixed and moving share the same grid. At fixed voxel **x = (50, 80, 40)**:

- Suppose **u(x) = (+3, −1, 0)** voxels → **φ(x) = (53, 79, 40)**.
- Intensity for fixed(x) is looked up in moving near (53, 79, 40).

The array **u** is three channels; at every **x** the three values form one **vector**. QC plots
often label a “u vectors” panel.

| Term | Use for |
| --- | --- |
| **Position map φ** | Absolute coordinates to sample from |
| **Displacement field / u vectors** | Offsets **u(x)** |
| **“Deformation field”** | Ambiguous — prefer φ or u explicitly |

UniGradICON often exposes `phi_AB_vectorfield`. After subtracting the identity map you get a
displacement; scripts convert it to voxel **u** (`phi_vectorfield_to_volume_voxels`,
`phi_dhw_to_u_xyz`).

| Say | Mean |
| --- | --- |
| **u** / **u vectors** / displacement | Offsets on fixed/source grid; keys `u_gt`, `u_pred` |
| **‖u‖** / **‖u_gt‖** / **‖u_pred‖** | Magnitude of the displacement at each voxel |
| **φ** / **position map** | Absolute sample coordinates; ICON output before `− identity` |
| **identity map / identity grid** | φ₀(x) = x — “do nothing” warp |
| **Backward / pull warp** | For each fixed **x**, sample moving at **φ(x)** |

### Backward (pull) warping

1. For each fixed voxel **x**, compute **φ(x) = x + u(x)**
2. Sample the moving image at **φ(x)** (trilinear interpolation)
3. Write that intensity into the registered output at **x**

```text
fixed grid x  ──►  φ(x) = x + u(x)  ──►  sample moving  ──►  registered_moving(x)
```

### Units and masks (3D)

| Context | Displacement / error units |
| --- | --- |
| HCP 3D synth | **voxels** (`u_gt` / `u_pred`; TorchIO params in mm) |
| IXI 3D IO | **voxels** in volume index space |

**Masks:**

- HCP synth: `source_mask` / `moving_mask` from `brainmask_fs`; `identity_grid_mask` for valid `u_gt`
- IO: `valid_mask` — shared atlas foreground (`atlas_valid_mask.npz`)
- HCP raw: `brainmask_fs.nii.gz` for download QC

---

## Phase I — Synthetic GT displacement (HCP)

Implementation: `experiments/synth-data-gen/torchio/create_synth_data.py`. Detail:
`docs/hcp-dataset.md`.

### Pipeline

```text
HCP T1w NIfTI (LAS, native space)
      ↓
TorchIO warp + identity-grid → u_back
      ↓
InvertDisplacementField → u_gt (source lattice)
      ↓
source, moving (z-scored), u_gt, masks  → datasets/synth-data/torchio/hcp/
```

**Convention:**

```text
moving(x + u_gt(x)) ≈ source(x)
```

### Implicit transforms (TorchIO / SimpleITK)

Transforms are applied **implicitly** (no full deformation field in RAM).

- **Rigid / affine:** global matrix **T** in mm; resampling uses **T⁻¹**.
- **Elastic:** sparse B-spline control grid (`ELASTIC_NUM_CONTROL_POINTS = 7`); `max_displacement`
  in **mm**.

For every output voxel **x**: index → world mm → backward map → fractional source index **φ(x)** →
sample source (trilinear) or pad.

### Recovering dense `u_gt`

**Step A — Identity-grid → temporary `u_back` (moving lattice).** Warp a 3-channel identity grid
with the MRI. At each **x** on the moving lattice: `u_back(x) = φ_back(x) − x`, so
`moving(x) ≈ source(x + u_back(x))`.

**Step B — Invert → stored `u_gt` (source lattice).** SimpleITK `InvertDisplacementField` (voxel
units) yields `u_gt` with `moving(x + u_gt(x)) ≈ source(x)`. Cleanup: OOB via `identity_grid_mask`
→ 12-voxel border → p99.9 clip on ‖u_gt‖.

### Intensity vs geometry (Phase I)

Masked z-score on `source` / `moving` does **not** change `u_gt`. Warp raw T1 first; extract and
invert displacement; then z-score images with shared μ, σ from the source brain mask.

### NPZ schema (Phase I)

`source`, `moving`, `u_gt` `(3, X, Y, Z)`, `source_mask`, `moving_mask`, `identity_grid_mask`,
`source_affine`, `deformation_class`, `subject_id`. See `docs/hcp-dataset.md`.

---

## Phase II — Predicted displacement and error map

Two tracks share the idea “compare a registration prediction to a reference,” but the reference
differs.

### HCP synth track

```text
Phase I NPZ
      ↓
UniGradICON net(moving, source) → u_pred (source lattice)
      ↓
u_error_map = ‖u_gt − u_pred‖
      ↓
datasets/error-map/unigrad-synth/hcp/
```

| Field | Meaning |
| --- | --- |
| `u_gt` | TorchIO GT after invert; **voxels** |
| `u_pred` | UniGradICON; same lattice; **voxels** |
| `u_error_map` | Per-voxel ‖u_gt − u_pred‖₂ |
| `error_map_mask` / `source_mask` | Valid voxels for metrics / U-Net loss |

`u_pred` cleanup matches Phase I (OOB → border → p99.9). Scripts:
`create_unigrad_synth_data.py`, `visualize_unigrad_data.py`.

### IO track (IXI 3D)

```text
source ↔ atlas pair
      ↓
Zero-shot ICON → phi_pred
      ↓
Instance optimization → phi_predio
      ↓
error_map = ‖phi_predio − phi_pred‖
```

| Stage | Description |
| --- | --- |
| **Zero-shot** | Single forward pass → `phi_pred` |
| **IO** | Extra gradient steps on the pair → `phi_predio` |
| **Error map** | How much IO improved over zero-shot (per voxel) |

Only voxels inside **`valid_mask`** (atlas foreground) are used for loss and metrics. IO iteration
count: `sweep_io_iterations.py`. See `docs/unigrad-io-experiment.md`.

### Intensity vs geometry (Phase II)

UniGradICON predicts φ / u in **voxel index space** from intensity-normalized pairs. Z-score does
not redefine geometry. Use the **same** μ, σ for both images in a pair when applicable.

---

## Phase III — Error map regression

Train a **3D U-Net** to predict the dense error map from inputs available at inference (images +
predicted displacement), without re-running registration. Scripts:

- HCP: `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py`
- IO: `experiments/regression/unigrad-io/train_unigrad_io_unet.py`

### What the U-Net predicts

The network does **not** predict φ or u. It predicts a **scalar error map** (magnitude) per voxel.

| Track | Inputs (5 channels) | Target | Loss mask |
| --- | --- | --- | --- |
| HCP synth | `source`, `moving`, `u_pred` × 3 (`/ u_scale`) | ‖u_gt − u_pred‖ | `source_mask` |
| IXI IO | `source`, `atlas`, `phi_pred` × 3 (`/ phi_scale`) | ‖φ_predio − φ_pred‖ | `valid_mask` |

W&B / `run_config.json` may list inputs as three names (`source`, `moving`, `u_pred`) or five
(`…_x/_y/_z`). Both mean the same stacking: **1 + 1 + 3 = 5 channels**. Displacement is stored as
`(3, X, Y, Z)` and concatenated along the channel axis — not split into separate arrays and
re-appended.

`u_scale` / `phi_scale` (default 64) puts displacement channels on a similar numeric scale as
intensities.

### Convolutional neural networks and U-Net (3D)

#### 1. Core mechanics of 2D vs 3D convolution

The difference is the dimensions along which the kernel slides.

**2D convolution (spatial).** Input `(C, H, W)`. Kernel slides left–right and top–bottom. Example:
RGB `(3, 128, 128)` with a filter of shape `(3, 5, 5)` → one 2D feature map.

**3D convolution (volumetric).** Input `(C, D, H, W)` or `(C, X, Y, Z)`. Kernel slides in three
spatial directions. Example: MRI `(1, 128, 128, 128)` with filter `(1, 3, 3, 3)` preserves continuity
across neighboring slices. **This project’s error-map U-Nets are 3D.**

#### 2. Kernels, filters, and layer transitions

- **Kernel:** weight window on **one** input channel (e.g. `3×3×3`).
- **Filter:** one kernel per input channel, stacked; produces **one** output feature map. Filter
  depth always matches the layer’s input channel count.

Output channel count = number of independent filters in that layer.

Example (2D shapes for brevity; 3D adds a depth axis the same way):

- Layer 1: input `(3, H, W)`, want 16 maps with `5×5` → weight tensor `(16, 3, 5, 5)`.
- Layer 2: input `(16, H, W)`, want 20 maps → weight tensor `(20, 16, 5, 5)`.

**How one feature-map cell is filled:** place a filter on a local patch; multiply each kernel with
its input channel; **sum all products** (+ bias) into one scalar; slide by stride and repeat.

#### 3. The U-Net architecture

Encoder–decoder with skip connections for dense voxel-wise regression.

```text
Contracting Path (Encoder)                 Expanding Path (Decoder)
[Input] ---> [DoubleConv] ---------------> [Concat + DoubleConv] ---> [1×1×1 Out]
                  |                              ^
               [Pool]                         [Upconv]
                  v                              |
             [DoubleConv] ----------------------> [Concat + DoubleConv]
                  |                              ^
               [Pool]                         [Upconv]
                  v                              |
             ===> [Bottleneck DoubleConv] ===
```

**Contracting path (encoder).** Double convolution (two `Conv3d` + norm + ReLU) extracts local
structure; **max pooling** (`2×2×2`) halves spatial size while channels typically double — larger
receptive field.

**Bottleneck.** Lowest resolution, highest channel depth; most abstract context.

**Expanding path (decoder).** **Transpose convolution** doubles spatial size and halves channels;
**skip connections** concatenate encoder feature maps (`torch.cat` on channel axis) to restore fine
spatial detail lost in pooling; another double conv fuses them.

**Output.** A `1×1×1` convolution maps final features to **1** channel (the predicted error map).

In this repo, `UNet3D` uses four pool/unpool stages (spatial ÷16 at the bottleneck). Volumes are
padded so each spatial dim is a multiple of 16. Default width is set by `--base-channels` (HCP synth
default **16** on 24 GB GPUs; IO often uses **32**).

#### 4. Training notes (HCP synth)

- Masked MAE or MSE: `sum(loss * mask) / (sum(mask) + ε)` over `source_mask` (`--train-loss`).
- Val metrics always logged: MAE, MSE, RMSE, Pearson r. Early stop / LR plateau / best
  checkpoint use `--val-loss` (default `mae`).
- Default val schedule: first validation at epoch 5, then every 5 epochs; early stop on the
  selected val metric.
- Optional ablation `--mask-u-pred`: zero `u_pred` outside `source_mask` before the U-Net.

---

## Key NPZ schemas (quick reference)

**HCP Phase I** (`datasets/synth-data/torchio/hcp/`): `source`, `moving`, `u_gt` `(3, X, Y, Z)`,
masks, `source_affine`, `deformation_class`, `subject_id`.

**HCP Phase II** (`datasets/error-map/unigrad-synth/hcp/`): Phase I keys + `u_gt_igm`, `u_pred`,
`u_error_map`, `error_map_mask`.

**IO** (`datasets/error-map/unigrad-io/ixi/`): `source`, `phi_pred`, `phi_predio`, `error_map`,
`io_iterations`, plus shared `atlas_valid_mask.npz`.

---

## Related files

| File | Role |
| --- | --- |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | Phase I — HCP synth NPZ |
| `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` | Phase II — HCP error maps |
| `experiments/error-map-gen/unigrad-io/create_unigrad_io_data.py` | Phase II — IO volumes |
| `experiments/regression/unigrad-synth/` | Phase III — HCP error-map U-Net |
| `experiments/regression/unigrad-io/` | Phase III — IO error-map U-Net |
| `docs/hcp-dataset.md` | HCP download, layout, synth generation |
| `docs/unigrad-synth-experiment.md` | HCP three-phase overview |
| `docs/unigrad-io-experiment.md` | IO track overview |
| `reports/uniGradICON.pdf` | ICON / IO equations |
