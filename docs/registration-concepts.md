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
   experiments/synth-data-gen/torchio/   experiments/error-map-gen/unigrad-io/      datasets/hcp/
```

| Track | Dim | Moving / fixed | GT displacement | Error map |
| --- | ---: | --- | --- | --- |
| **Synth (IXI 2D)** | 2D | `image` → `warped` | `phi_true` in `*_triplet.npz` | ‖φ_pred − φ_true‖ per pixel |
| **Synth (HCP 3D)** | 3D | `source` → `moving` | `u` in `hcp_synth/*.npz` | *Planned* ‖u_pred − u‖ per voxel |
| **IO** | 3D | `source` → `atlas` | `phi_predio` (IO-refined) | ‖φ_predio − φ_pred‖ per voxel |
| **HCP raw** | 3D | T1w NIfTI | — | QC / future registration |

**Regression scripts:**

- 2D: `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py` (`UNet2D`, 4 inputs)
- 3D: `experiments/regression/unigrad-io/train_unigrad_io_unet.py` (`UNet3D`, 5 inputs)

---

## End-to-end pipeline (synth tracks)

### IXI 2D (legacy)

```text
Native MRI slice (IXI_2D)
      ↓
Synthetic deformation (TorchIO)     ← create_synth_data.py (2D triplets)
      ↓
Warped MRI + phi_true (*_triplet.npz)
      ↓
UniGradICON zero-shot                 ← create_unigrad_synth_data.py
      ↓
phi_pred, error_map (*_fiver.npz)
      ↓
U-Net regression                    ← train_unigrad_synth_unet.py
```

### HCP 3D (current Phase I)

```text
HCP T1w NIfTI (LAS, native space)
      ↓
TorchIO warp + identity-grid u      ← create_synth_data.py
      ↓
source, moving (z-scored), u, mask  → datasets/synth-data/torchio/hcp/
      ↓
(planned) UniGradICON + error-map U-Net
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

## Displacement **u** vs position map **φ**

Registration stores geometry on the **fixed** (output) grid. At each fixed-grid location **x**:

| Symbol | Name in this repo | What it stores | Shape (3D) |
| --- | --- | --- | --- |
| **u(x)** | displacement / **u vectors** | offset **(dx, dy, dz)** in voxels | `(3, X, Y, Z)` |
| **φ(x)** | **position map** (not displacement) | absolute sample coordinate **x + u(x)** | `(3, X, Y, Z)` |

Relation (component-wise):

```text
φ(x) = x + u(x)
u(x) = φ(x) − x
```

### How to read **u(x)** (intuition)

For output-grid voxel **x** on the **fixed** image:

- **u(x)** is the **offset** into the **moving** volume.
- Sample **moving(x + u(x))** to get the anatomy that should match **fixed(x)**.

So the aligned (warped) moving image is:

```text
registered_moving(x) ≈ moving(x + u(x)) ≈ fixed(x)
```

**Example (2D, one voxel).** Fixed and moving share the same grid. At fixed voxel **x = (50, 80)**:

- Suppose **u(x) = (+3, −1)** voxels.
- Then **φ(x) = (53, 79)**.
- Meaning: intensity for fixed(50, 80) is looked up in the moving image at ≈ (53, 79).
- Matching anatomy for that fixed location lives ~3 voxels right and 1 voxel down in moving space.

**Example (vector view).** The array **u** is a stack of three channels. At every **x**, the three values form one **vector** pointing from **x** toward the moving sample location. Calling these **u vectors** is fine and matches QC plots that label a “u vectors” panel (arrow / RGB-as-vector viz of the three components).

### Is **φ** a “deformation field”?

Prefer **position map** (or warping map / deformation map) for **φ**:

| Term | Use for | Why |
| --- | --- | --- |
| **Position map φ** | Absolute coordinates to sample from | Clear: values are locations, not offsets |
| **Displacement field / u vectors** | Offsets **u(x)** | Clear: each voxel stores a vector offset |
| **“Deformation field”** | Ambiguous colloquialism | People use it for **either** φ **or** u |

Mathematically both **φ** and **u** are maps **x ↦ ℝ³**, so both can be called “fields.” In practice, **field** almost always means **u** (a vector of increments at each point). **φ** stores **positions**, so “deformation / position map” is the accurate label here — it is **not** the same object as a displacement field until you subtract the identity grid.

UniGradICON internals often expose something named like `phi_AB_vectorfield`. After subtracting the identity map you get a **displacement**; scripts then convert that to voxel **u** (see `create_unigrad_synth_data.py`: `phi_vectorfield_to_volume_voxels`, then `phi_dhw_to_u_xyz`).

### Naming cheat sheet (this repo)

| Say | Mean |
| --- | --- |
| **u** / **u vectors** / displacement | Offsets on fixed grid; HCP keys `u`, `u_gt`, `u_pred` |
| **‖u‖** | Magnitude of the displacement vector at each voxel |
| **φ** / **position map** | Absolute sample coordinates; ICON output before `− identity` |
| **identity map / identity grid** | φ₀(x) = x — “do nothing” warp |
| **Backward / pull warp** | For each fixed **x**, sample moving at **φ(x)** |

---

## Backward (pull) warping

Registration warps the **moving** image onto the **fixed** grid:

1. For each fixed (output) voxel **x**, compute sample coordinate **φ(x) = x + u(x)**
2. Sample the moving image at **φ(x)** (interpolation)
3. Write that intensity into the registered output at **x**

Non-integer **φ(x)** requires **interpolation** (trilinear for MRI intensity; nearest-neighbor for labels).

```text
fixed grid x  ──►  φ(x) = x + u(x)  ──►  sample moving  ──►  registered_moving(x)
                        ▲
                        └── u(x) is the offset (u vector) on the fixed grid
```

---

## 3D synthetic deformation and displacement extraction

This pipeline generates synthetic non-rigid registration pairs by passing a 3D volume through random
spatial transformations while tracking the underlying geometric displacement. Implementation:
`experiments/synth-data-gen/torchio/create_synth_data.py`. See also `docs/unigrad-synth-experiment.md`.

### 1. The implicit transformation pipeline

To remain memory-efficient, TorchIO (via SimpleITK) **never materializes a full-resolution
deformation field** in RAM. It applies transforms **implicitly** using a physical-space backward-
warping loop.

**Rigid / affine transforms**

A global transformation matrix **T** is sampled (rotation, scale, translation in mm). For
resampling, SimpleITK needs the mapping from **output** space → **input** space, so the pipeline
uses **T⁻¹** (computed once per sample).

**Elastic transforms**

A sparse, low-resolution grid of B-spline **control points** is sampled (in this repo:
`ELASTIC_NUM_CONTROL_POINTS = 7` per axis → a 7×7×7 control mesh). Continuous cubic B-splines
interpolate these points to yield a backward displacement vector at any physical coordinate.
`max_displacement` is in **mm**.

**Voxel-by-voxel execution loop**

For every integer voxel slot **x** in the target (destination) volume:

1. **Index → world:** voxel index **x** is mapped to physical millimeters via `source_affine`.
2. **Backward mapping:** evaluate **T⁻¹** or the B-spline field to find the physical coordinate
   where that tissue originated in the source volume.
3. **World → index:** map that source physical coordinate back to a **fractional voxel index**
   **φ(x)** via the inverse affine.
4. **Boundary and sample:** if φ(x) is out of bounds, use the pad value (e.g. 0 or image minimum);
   otherwise **sample** the source with trilinear interpolation and write the result to the target
   voxel.

**Display note:** `create_synth_data.py` uses `default_pad_value="minimum"` for vacated voxels after
global affine warps, then masked z-score sets outside-brain voxels to 0. The fixed brain mask is not
warped with the image, so QC figures can show **dark voids** in the moving panel where anatomy moved
away — this is a padding/intensity artifact, not a change to stored **u** (geometry).

This is **backward (pull) warping**: the output grid is fixed; we look up where each output voxel
came from in the input.

```text
output voxel x  ──affine──►  world mm  ──T⁻¹ / B-spline──►  source world mm  ──inv affine──►  φ(x)
                                                                                              │
                                                                                    sample source at φ(x)
                                                                                              ▼
                                                                                        moving(x)
```

### 2. Recovering the dense displacement field (u)

Because the transform is computed implicitly, TorchIO does **not** natively return a dense
voxel-by-voxel displacement field **u** for training.

**The identity-grid trick**

A companion 3-channel volume is warped alongside the MRI:

| Step | Description |
| --- | --- |
| **Initialization** | Build an *identity grid* with the same shape as the MRI. Channel 0/1/2 store raw voxel indices **i, j, k** at array position `[i, j, k]` (not tissue intensities). |
| **Implicit warping** | Pass this grid through the **same** backward-warping pipeline as the MRI (same `source_affine`, same transform). The warp rearranges coordinate *values* as if they were intensities. |
| **Vector subtraction** | The warped grid holds source indices **φ(x)** at each output position **x**. Displacement is **u(x) = φ(x) − x** (component-wise). |

In this repo we store **u** in voxel units (`u_unit="vox"`). The position map is **φ(x) = x + u(x)**.

The recovered **u** is a dense field in voxel index space, compatible with backward lookup and
with spatial transformer layers (e.g. PyTorch `grid_sample` convention).

**Code reference**

```text
subject = Subject(
    mri  = ScalarImage(tensor=source,  affine=source_affine),
    grid = ScalarImage(tensor=identity_grid, affine=source_affine),
)
transformed = transform(subject)
u = transformed.grid - identity_grid
```

---

## Intensity normalization vs displacement (and downstream φ)

Geometry and intensity are **separate**:

| | Displacement **u** / **φ** | Image intensities |
| --- | --- | --- |
| **Represents** | Where each output voxel samples from (index-space geometry) | MRI signal strength |
| **Units** | Voxels (index space) | Scanner units → masked z-score for training |
| **Changed by `(I−μ)/σ`?** | **No** | **Yes** |

### Phase I (synth generation)

`u` comes from the **identity-grid trick**, not from intensity values. In `create_synth_data.py`:

1. Warp **raw** T1; extract **u** from the coordinate grid.
2. QC on raw intensities.
3. Masked z-score **source** and **moving** with shared μ, σ from the source brain mask; save **u**
   unchanged.

Z-score rescales scalar values per voxel. It does **not** move voxel indices, so it cannot alter **u**.

### Phase II (downstream registration → `phi_pred`)

UniGradICON (and similar models) predict **φ** or **u** in **voxel index space** from normalized
image pairs. Masked z-score with **shared** μ, σ:

- Puts intensities in a stable range for the network.
- Is an affine intensity transform; deformable registration targets **spatial** alignment, which is
  invariant to affine intensity rescaling in principle (e.g. LNCC).
- Does **not** change the voxel grid or the geometric correspondence that **φ** describes.

Use the **same** μ, σ for `source` and `moving`. Independent z-score per volume would change
relative appearance and is not used here.

**Summary:** z-score affects **what the network reads** (intensity); it does not redefine **where**
voxels map. Stored GT **u**, and downstream **phi_pred**, both describe geometry on the fixed voxel
grid.

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

IO iteration count is chosen via `sweep_io_iterations.py` before building `datasets/error-map/unigrad-io/ixi/`.

---

## Units and masks

| Context | Displacement / error units |
| --- | --- |
| IXI 2D synth | **pixels** on the slice grid |
| HCP 3D synth | **voxels** (`u` in `hcp_synth/*.npz`; TorchIO params in mm) |
| IXI 3D IO | **voxels** in volume index space |

**Masks:**

- IXI 2D synth: `valid_mask` — interior margin away from slice boundary
- HCP 3D synth: `mask` from `brainmask_fs`; interior margin in QC (`INTERIOR_MARGIN`)
- IO: `valid_mask` — shared atlas foreground (`atlas_valid_mask.npz`)
- HCP raw: `brainmask_fs.nii.gz` for download QC (`visualize_hcp_data.py`)

**QC flags:** `qc_passed` on triplets/fivers; synth Phase II skips failed triplets by default.

---

## Key NPZ / file schemas (quick reference)

**`*_triplet.npz` (IXI 2D):** `image`, `warped`, `phi` (2, H, W), optional `valid_mask`, `qc_passed`

**`hcp_synth/*.npz` (HCP 3D):** `source`, `moving`, `u` (3, X, Y, Z), `mask`, `source_affine`,
`source_spacing`, `u_unit`, `deformation_class` (`none` | `rigid` | `affine` | `elastic` |
`affine_elastic`), `subject_id`, `qc_passed`. Filenames: `<subject_id>_<suffix>.npz` where suffix
is `none`, `rig`, `aff`, `ela`, or `aela`. See `docs/unigrad-synth-experiment.md` § Deformation
classes and file nomenclature.

**`*_fiver.npz` (Phase II synth):** `image`, `warped`, `phi_true`, `phi_pred`, `phi_diff`, `error_map`, `valid_mask`, `qc_passed`

**IO `*.npz` (Phase II 3D):** `source`, `phi_pred`, `phi_predio`, `error_map`, `io_iterations`, …

---

## Related files

| File | Role |
| --- | --- |
| `experiments/synth-data-gen/torchio/create_synth_data.py` | HCP 3D synth NPZ (Phase I) |
| `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py` | Phase II fivers |
| `experiments/error-map-gen/unigrad-io/create_unigrad_io_data.py` | Phase II IO volumes |
| `experiments/regression/unigrad-synth/` | 2D error-map U-Net |
| `experiments/regression/unigrad-io/` | 3D error-map U-Net |
| `docs/hcp-dataset.md` | HCP download and layout |
| `reports/uniGradICON.pdf` | ICON / IO equations |
