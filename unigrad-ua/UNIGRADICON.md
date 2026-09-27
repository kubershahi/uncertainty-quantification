# uniGradICON: how it was pretrained

Everything needed to continue pretraining, fine-tune, or extend uniGradICON, with the source of each fact. Sources: the paper `reports/uniGradICON.pdf` (arXiv 2403.05780v1, 9 Mar 2024) cited as `paper p<N>`; the code at [uncbiag/uniGradICON](https://github.com/uncbiag/uniGradICON) commit `3e83a92` cited as `train.py:<line>`, `dataset.py:<line>`, `__init__.py:<line>` (`src/unigradicon/__init__.py`); and the `icon_registration` 1.1.7 package cited as `IR/<file>:<line>`. Numbers marked **measured** were reproduced by building the network on CPU.

Companion document: [`RECON.md`](RECON.md) (uncertainty-head literature and design).

---

## 1. Identity

| Item | Value | Source |
| --- | --- | --- |
| Paper | Tian, Greer, Kwitt, Vialard, San José Estépar, Bouix, Rushmore, Niethammer. *uniGradICON: A Foundation Model for Medical Image Registration*. MICCAI 2024 | paper p1 |
| Version read | arXiv 2403.05780v1; says "twelve datasets", the table has thirteen rows | paper p1, p12 |
| Code | `unigradicon` 1.0.4 (tags up to 1.0.5), on top of `icon_registration` ≥ 1.1.6 | `setup.cfg`, `requirements.txt` |
| Built on | GradICON (Tian et al., CVPR 2023, [arXiv 2206.05897](https://arxiv.org/abs/2206.05897)); ICON (Greer et al., ICCV 2021, [arXiv 2105.04459](https://arxiv.org/abs/2105.04459)) | paper p3 |
| Weights | `https://github.com/uncbiag/uniGradICON/releases/download/unigradicon_weights/Step_2_final.trch` | `__init__.py:263–279` |
| Weights cache | `network_weights/unigradicon1.0/Step_2_final.trch`, **relative to the current working directory**; override with `weights_location=` | `__init__.py:266–273` |
| Sibling models | multiGradICON (`multigradicon_weights` release); uniCARL preview (`v1.0.4` release, 160³) | `__init__.py:244–261`, `unicarl.py:17,31` |
| CLI | `unigradicon-register`, `unigradicon-warp`, `unigradicon-jacobian`, `unicarl-register` | `setup.cfg:30–34` |
| Local install | editable in conda env `biag` from `~/Documents/GitHub/uniGradICON` | — |

## 2. Pretraining corpus

Paper Table 6 (p12). Datasets 1–4 train the model, 5–7 are in-distribution tests, 8–13 are out-of-distribution (OOD) tests and fine-tuning data.

| # | Dataset | Region | Patients | Per patient | Pairs | Type | Modality | Role |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 1 | COPDGene | Lung | 899 | 2 | 899 | intra-patient | CT | train |
| 2 | OAI | Knee | 2,532 | 1 | 3,205,512 | inter-patient | MRI | train |
| 3 | HCP | Brain | 1,076 | 1 | 578,888 | inter-patient | MRI | train |
| 4 | L2R-Abdomen (AbdomenCTCT) | Abdomen | 30 | 1 | 450 | inter-patient | CT | train |
| 5 | DirLab-COPDGene | Lung | 10 | 2 | 10 | intra-patient | CT | in-distribution test |
| 6 | OAI-test | Knee | 301 | 1 | 301 | inter-patient | MRI | in-distribution test |
| 7 | HCP-test | Brain | 32 | 1 | 100 | inter-patient | MRI | in-distribution test |
| 8 | L2R-NLST-val | Lung | 10 | 2 | 10 | intra-patient | CT | OOD type 1 |
| 9 | L2R-OASIS-val | Brain | 20 | 1 | 19 | inter-patient | MRI | OOD type 1 |
| 10 | IXI-test | Brain | 115 | 1 | 115 | atlas–patient | MRI | OOD type 1 |
| 11 | L2R-CBCT-val | Lung | 3 | 3 | 6 | intra-patient | CT/CBCT | OOD type 3 |
| 12 | L2R-CTMR-val | Abdomen | 3 | 2 | 3 | intra-patient | CT/MRI | OOD type 3 |
| 13 | L2R-CBCT-train | Lung | 3 | 11 | 22 | intra-patient | CT/CBCT | fine-tuning |

- **Pair counts** in the table are n²/2 (2532²/2 = 3,205,512), not n(n−1)/2.
- **L2R-Abdomen:** the paper follows the official Learn2Reg split, in which the validation images are part of the training set (p12). The OOD type-2 experiment uses a separate model retrained **without** dataset 4 (p6).
- **Pairing at train time:** COPDGene pairs are inspiration/expiration of the same patient; OAI, HCP and Abdomen draw two random images from the dataset on every `__getitem__`, ignoring the index (`dataset.py:153–158`).

**Access routes**

| Dataset | Route | Notes |
| --- | --- | --- |
| COPDGene | dbGaP phs000179 / COPDGene ancillary-study application | restricted |
| OAI | NIMH Data Archive, collection 2343 (paper p9) | controlled access |
| HCP | ConnectomeDB / `s3://hcp-openaccess` | **already downloaded** by this repo (`scripts/download_hcp.sh`, 1113 subjects) |
| L2R AbdomenCTCT | [Learn2Reg datasets page](https://learn2reg.grand-challenge.org/Datasets/) | public |

**On-disk layout `training/dataset.py` expects** (pre-serialised torch tensors, not NIfTI), root `DATASET_DIR = "./data/uniGradICON/"` (`dataset.py:11`):

| Dataset class | Path | Per-image normalisation |
| --- | --- | --- |
| `COPDDataset` | `half_res_preprocessed_transposed_SI/lungs_train_2xdown_scaled` (+ `lungs_seg_…` with `ROI_only=True`) | min–max to [0,1], multiplied by the lung mask (`dataset.py:44–53`) |
| `OAIDataset` | `OAI/knees_big_2xdown_train_set` | clip to [min, p99], scale to [0,1] |
| `HCPDataset` | `HCP/brain_train_2xdown_scaled` | clip to [min, p99], scale to [0,1] (`dataset.py:141–148`) |
| `L2rAbdomenDataset` | `AbdomenCTCT/imagesTr/*` | `(clip(HU, −1000, 1000) + 1000) / 2000` (`dataset.py:182`) |

The `2xdown` names mean the stored volumes are already half resolution; every class then resizes trilinearly to 175³. The script to build these tensors from raw data is **not** in the repo.

## 3. Preprocessing (inference)

| Modality | Intensity | Source |
| --- | --- | --- |
| CT | clamp HU to [−1000, 1000], rescale to [0,1] | `__init__.py:309–327`, paper p3 |
| MRI | clamp to [min, p99], rescale to [0,1] | same |
| Both | trilinear resize to 175³ (anisotropic spacing allowed); the output field is interpolated back to the original grid for evaluation | `IR/itk_wrapper.py:66–71`, paper p3 |

Not done: affine pre-alignment (the paper says it is not needed, p5), skull stripping, cropping. An optional segmentation multiplies the image before registration (`__init__.py:326`). This repo's equivalent is `preprocess_volume_for_unigrad` in [`create_unigrad_synth_data.py:110–122`](../experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py#L110-L122).

## 4. Architecture

`make_network(input_shape, include_last_step=True)` (`__init__.py:218–232`) builds a two-step, multi-resolution GradICON network. All four UNets are `tallUNet2`.

| Step | Wrapper | Resolution | Input | UNet params (measured) |
| --- | --- | --- | --- | ---: |
| 1a | `DownsampleRegistration` ×2 → `FunctionFromVectorField` | 44³ | A, B | 17,670,229 |
| 1b | `TwoStepRegistration` inside `DownsampleRegistration` | 88³ | A warped by 1a, B | 17,670,229 |
| 1c | `TwoStepRegistration` | 175³ | A warped by 1a∘1b, B | 17,670,229 |
| 2 | `TwoStepRegistration(inner, FunctionFromVectorField(tallUNet2))` | 175³ | A warped by step 1, B | 17,670,229 |
| **Total** | | | | **70,680,916** |

- **`tallUNet2`** = `UNet2(5, [[2,16,32,64,256,512],[16,32,64,128,256]], dim=3)` (`IR/networks.py:529–534`). Input `cat([A, B])` (2 channels). Down path: stride-2 3³ convolutions with average-pooled residual shortcuts; up path: 4³ transposed convolutions, BatchNorm3d after each; leaky ReLU; skip concatenation (`IR/networks.py:255–287`).
- **Output layer:** `lastConv = Conv3d(18 → 3, k=3)`, weights and bias **initialised to zero**, output divided by 10 (`IR/networks.py:249–253, 286–287`). 18 = 2 skip channels + 16 decoder channels.
- **Normalisation and stochasticity:** 20 BatchNorm3d layers, **no dropout** (measured).
- **Composition:** `TwoStepRegistration.forward` warps A with φ, predicts ψ on (warped A, B), and returns `x ↦ φ(ψ(x))` (`IR/network_wrappers.py:206–217`). `FunctionFromVectorField` returns `x ↦ x + u(x)`, taking a shortcut (no interpolation) when `x` is the identity grid (`IR/network_wrappers.py:111–120`). So the final displacement is `u(x) = u_ψ(x) + u_φ(x + u_ψ(x))`.

**Output convention (this matters for any uncertainty on the field).**

| Quantity | Meaning | Units |
| --- | --- | --- |
| `net.phi_AB` | function from coordinates on **B's lattice** (fixed) into A's space; `warped_A(x) = A(φ_AB(x))` | normalised |
| `net.phi_AB_vectorfield` | `phi_AB(identity_map)`, shape `(B, 3, 175, 175, 175)` | normalised [0,1] coordinates |
| `net.phi_BA`, `net.phi_BA_vectorfield` | the reverse direction, computed in the same forward pass | normalised |
| `net.identity_map` | non-persistent buffer, range [0, 1], spacing `1/(175−1)` (`IR/network_wrappers.py:50–59`) | normalised |
| displacement `u` | `phi_AB_vectorfield − identity_map` | normalised; ×174 = voxels at 175³; ×(N−1) per axis = native voxels |

This repo converts to native voxels with `p · (N_native − 1)` per axis after trilinear resampling ([`create_unigrad_synth_data.py:125–141`](../experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py#L125-L141)).

## 5. Objective

Paper Eq. 1 (p4), implemented in `GradientICONSparse.forward` (`__init__.py:20–188`):

`L = L_sim(A∘φ_AB, B) + L_sim(B∘φ_BA, A) + λ · ‖∇(φ_AB∘φ_BA) − I‖²_F`

| Term | Code value | Source |
| --- | --- | --- |
| λ | 1.5 | `__init__.py:218` |
| Similarity | `1 − mean(LNCC)`, Gaussian window σ = 5, kernel 21 (`sigma*4+1`) | `IR/losses.py:577–606` |
| Symmetry | similarity in both directions; warping with `zero_boundary=True` | `__init__.py:83, 90, 128` |
| Gradient inverse consistency | sample points `Iε = id + 2·randn/175`, subsampled `[::2]³`; finite-difference Jacobian of `φ_AB∘φ_BA(Iε) − Iε` with δ = 0.001 along x, y, z; mean squared | `__init__.py:131–175` |
| Returned | `ICONLoss(all_loss, inverse_consistency_loss, similarity_loss, transform_magnitude, flips)`; `flips` = count of voxels with negative Jacobian determinant | `IR/losses.py:21–24, 830–840` |

Alternatives available without code changes: `make_sim("lncc2")` = squared LNCC (for multimodal), `make_sim("mind")` = MIND-SSC (r = 2, d = 2) (`__init__.py:234–242`); optional Jacobian intensity-conservation term for CT (`apply_intensity_conservation_loss`, `__init__.py:115–126`). `IR/losses.py` also has `GradientICON` (dense, line 117), `BendingEnergyNet` (363) and `DiffusionRegularizedNet` (492).

## 6. Training recipe

`training/train.py`, function `train_two_stage` (lines 197–265), called with `epochs=[801, 201]`, `eval_period=20`, `save_period=20` (line 297).

| Item | Value | Source |
| --- | --- | --- |
| Stage 1 | 3-UNet network (`include_last_step=False`), 801 epochs → `Step_1_final.trch` | `train.py:199–226` |
| Stage 2 | 4-UNet network; stage-1 weights loaded into `regis_net.netPhi`; **all** parameters trained 201 epochs → `Step_2_final.trch` | `train.py:231–265` |
| Paper wording | "800 epochs" and "200 epochs"; lr 5e-5; GradICON defaults, no further tuning | paper p4, footnote 7 |
| Sampling | `ConcatDataset` of the four datasets, `data_num = 1000` each (COPDGene capped at its 899 pairs) → 3,899 pairs per epoch, roughly balanced per dataset | `train.py:28–37`, `dataset.py:29–31` |
| Batch | 4 per GPU × 4 GPUs = 16, `torch.nn.DataParallel`, `drop_last=True` | `train.py:23–25, 279–285` |
| Iterations | 3,899 / 16 = 243 per epoch → ≈ 194,600 (stage 1) + ≈ 48,800 (stage 2) ≈ **243,500** | derived |
| Optimiser | Adam, lr 5e-5, no scheduler, no weight decay | `train.py:215, 248` |
| Augmentation (both images, under `no_grad`) | random axis permutation + sign flip shared by A and B; independent 0.05·N(0,1) noise on each image's 3×4 affine | `train.py:39–85` |
| Checkpoints | `regis_net.state_dict()` + optimiser every 20 epochs; `--resume_from` restores stage 1 only | `train.py:137–145, 207–219` |
| "Validation" | every 20 epochs one batch from **the training set** is visualised in TensorBoard; there is no held-out split | `train.py:147–150, 286–292` |
| Logging | `footsteps` output directory + TensorBoard | `train.py:112–121` |
| cuDNN | `enabled = True`, `benchmark = True` | `train.py:202–203` |

## 7. Compute

The paper reports **no** GPU type, count, batch size or wall-clock time. What the code implies:

- 4 GPUs (`device_ids = [1, 0, 2, 3]`), 16 pairs per step at 175³, each pair evaluated in **both** directions plus three extra composed passes for the finite-difference Jacobian.
- ≈ 243,500 optimiser steps in total.

**Order-of-magnitude estimate (assumption, not a measurement):** if one 4-GPU step takes 1–2 s, pretraining is 68–135 h of wall-clock, i.e. roughly 270–540 GPU-hours. Replace this with a measured number before planning: time 20 steps of `train_kernel` on one Nautilus GPU at batch 2 (the step time scales roughly linearly in batch).

**What fits on one GPU for fine-tuning:** a head-only probe needs activations of the frozen network only up to the head, and gradients only through the head, so batch 1–2 at 175³ fits in 24–48 GB. Full fine-tuning of all 70.7 M parameters at 175³ needs the full activation graph of four UNets in two directions; expect batch 1 per 40–80 GB GPU and gradient accumulation. Keep BatchNorm in `eval()` at these batch sizes.

## 8. Instance optimisation (IO) and fine-tuning

**IO** is test-time fine-tuning of the whole network on the one pair being registered.

| Item | Value | Source |
| --- | --- | --- |
| Function | `finetune_execute(model, A, B, steps)`; `finetune_execute_mask(…)` with masks | `IR/itk_wrapper.py:12–39` |
| Optimiser | Adam, lr 2e-5, over **all** `model.parameters()` | `IR/itk_wrapper.py:14` |
| Loss | the full GradICON loss (`loss_tuple[0]`) | `IR/itk_wrapper.py:18–19` |
| Steps | 50 by default (`--io_iterations`, `"None"` disables) | `__init__.py:349` |
| Mode | model stays in `eval()`, so BatchNorm uses running statistics | `__init__.py:260, 278` |
| **Caveat** | `finetune_execute` does **not** restore the weights afterwards (the `load_state_dict` line is commented out, `IR/itk_wrapper.py:23`); only the mask variant restores. Registering a second pair with the same model object continues from the first pair's IO weights. Reload or deep-copy between pairs. | measured |

This repo runs its own IO loop in [`create_unigrad_io_data.py:169–190`](../experiments/error-map-gen/unigrad-io/create_unigrad_io_data.py#L169-L190).

**IO gains reported in the paper** (zero-shot → IO):

| Task | Metric | Zero-shot | +IO | Source |
| --- | --- | ---: | ---: | --- |
| COPDGene | mTRE (mm) ↓ | 2.26 | 1.40 | p6 |
| OAI | Dice (%) ↑ | 68.9 | 70.3 | p6 |
| HCP | Dice (%) ↑ | 76.2 | 78.9 | p6 |
| L2R-Abdomen | Dice (%) ↑ | 48.3 | 52.2 | p6 |
| NLST | mTRE (mm) ↓ | 2.07 | 1.77 | p7 |
| OASIS | Dice (%) ↑ | 79.0 | 79.6 | p7 |
| IXI | Dice (%) ↑ | 70.6 | 71.3 | p7 |
| CBCT | Dice (%) ↑ | 57.0 | 59.9 | p8 |
| CT–MR | Dice (%) ↑ | 50.0 | 66.8 | p8 |
| Abdomen, model trained without Abdomen | Dice (%) ↑ | 34.1 | 45.3 | p7 |

IO usually increases folding: `%|J|<0` on Abdomen goes 0.31 → 0.96 and on CT–MR 0.04 → 0.61 (p6–8).

**Fine-tuning** (the only one reported): on L2R-CBCT-train for 4,000 epochs with the same lr and hyperparameters; CBCT Dice 60.3, 63.7 with fine-tune + IO, above the Learn2Reg top-1 test score of 63.2 (p8).

## 9. Evaluation protocol and headline numbers

**OOD taxonomy** (paper Table 2, p6): type 1 = same anatomy, different source (NLST, OASIS, IXI); type 2 = unseen anatomy (Abdomen with the model trained without it); type 3 = unseen modality (CBCT, CT–MR). Metrics: mTRE (mm), Dice (%), `%|J|<0`.

**In-distribution (Table 1, p6)**

| Method | COPDGene mTRE ↓ | OAI Dice ↑ | HCP Dice ↑ | L2R-Abd Dice ↑ |
| --- | ---: | ---: | ---: | ---: |
| Initial | 23.36 | 7.6 | 53.4 | 25.9 |
| Best task-specific (GradICON + IO) | 1.31 | 71.2 | 80.5 | — |
| SyN | 1.79 | 65.7 | 75.8 | 25.2 |
| Universal VoxelMorph-SVF (same corpus, MSE) | 19.21 | 55.0 | 44.2 | 33.8 |
| **uniGradICON** | 2.26 | 68.9 | 76.2 | 48.3 |
| **uniGradICON + IO** | 1.40 | 70.3 | 78.9 | 52.2 |

**OOD (Tables 3–5, p7–8)**

| Task | uniGradICON | Comparison |
| --- | ---: | --- |
| NLST mTRE (mm) | 2.07 | SyN 3.04; L2R test top-1 1.44, top-5 2.04 |
| OASIS Dice | 79.0 | SyN 75.6; L2R test top-1 82, top-5 78 |
| IXI Dice | 70.6 | VoxelMorph 73.2, TransMorph 75.4, SyN 64.5 |
| Abdomen Dice, trained without Abdomen | 34.1 (45.3 with IO) | L2R top-5 49, top-1 69 |
| CBCT Dice | 57.0 | SyN 57.4; L2R top-5 56.9 |
| CT–MR Dice | 50.0 | SyN 45.0; L2R top-5 71, top-1 75 |

Learn2Reg numbers are on the test sets, uniGradICON's on the validation sets; the authors say they are "not directly comparable" (p6).

**Stated limitations** (p9, p13): few in-distribution tasks; multimodal performance (suggested: modality-agnostic features, more multimodal data, LNCC² or NMI); only intensities used, no segmentations; a larger network would likely help; no universal affine network; orientation handling. **The paper never mentions uncertainty.**

## 10. multiGradICON and uniCARL (pointers)

- **multiGradICON** (Demir et al., WBIR 2024, [arXiv 2408.00221](https://arxiv.org/abs/2408.00221); `training/train_multi.py`): corpus COPDGene, BraTS-Reg, AbdomenCTCT (plus intensity-inverted CT), HCP, ABCD FA/MD, OAI multi-sequence, L2R MR–CT, UK Biobank; `use_label=True` computes the similarity on a random other modality of the same subject; `WeightedRandomSampler` with weight `regions · C(n_mod+1, 2)`, 4,000 samples per epoch; same `[801, 201]` schedule plus 100 anatomy-balanced fine-tuning epochs (`train_multi.py:41–84, 317–354`).
- **uniCARL**: preview model at 160³ in `unicarl.py`; not documented in the paper.

## 11. API cheat-sheet

| Call | What it does |
| --- | --- |
| `get_unigradicon(loss_fn=icon.LNCC(sigma=5), apply_intensity_conservation_loss=False, weights_location=None)` | builds the 4-UNet network, downloads weights if missing (relative to cwd), strict-loads into `net.regis_net`, moves to `config.device`, `eval()` |
| `make_network(input_shape, include_last_step, lmbda, loss_fn, …)` | same network without weights (used for training and to count parameters on CPU) |
| `make_sim("lncc" \| "lncc2" \| "mind")` | similarity objects for IO or training |
| `preprocess(itk_image, modality="ct" \| "mri", segmentation=None)` | intensity normalisation (§3) |
| `icon.itk_wrapper.register_pair(model, A, B, finetune_steps=None \| int)` | ITK-level registration with resize to 175³ and optional IO; returns ITK transforms |
| `loss = net(A, B)` (5D tensors at 175³) | forward pass; sets `net.phi_AB`, `net.phi_BA`, `net.phi_AB_vectorfield`, `net.warped_image_A`; returns `ICONLoss` |

**Device caveat:** `icon_registration.config.device` is CUDA if available, else CPU; there is no MPS branch (`IR/config.py:3–6`), and `GradientICONSparse` moves its random tensors to `config.device` explicitly (`__init__.py:133, 154–161`). Moving the model to MPS on a Mac causes a device mismatch; use CPU for smoke tests.

## 12. Extension points for a probabilistic output

What an uncertainty-aware variant has to touch. The design choice between these is argued in [`RECON.md`](RECON.md) §3–4.

| Point | Location | What to know |
| --- | --- | --- |
| Final field producer | `net.regis_net.netPsi` = `FunctionFromVectorField(tallUNet2)` of step 2 | produces only the **last residual** `u_ψ`; the composed field is `u_ψ(x) + u_φ(x + u_ψ(x))` (§4) |
| Output layer | `net.regis_net.netPsi.net.lastConv`, `Conv3d(18→3)`, zero-init, output ÷ 10 | where 3 log-variance channels go (replace with 18→6), or whose **input** an add-on head reads |
| Head input without code changes | `register_forward_hook` on `lastConv`: its input is the 18-channel full-resolution feature map (2 skip + 16 decoder), its output ×0.1 is `u_ψ` | `UNet2.forward` returns only the field and `FunctionFromVectorField` stores nothing |
| Sampling point | `FunctionFromVectorField.forward` (`IR/network_wrappers.py:111–120`) | a reparameterised sample `μ + σ⊙ε` would replace `tensor_of_displacements` here, before the closure is built |
| Loss | `GradientICONSparse.forward` (`__init__.py:177`, `all_loss = λ·ICON + sim`) | a prior / KL term on σ joins `all_loss` here; `train_kernel` in `train.py:87–94` uses `loss_object.all_loss` unchanged |
| Checkpoint | strict `net.regis_net.load_state_dict` (`__init__.py:276`), 188 keys `netPhi.*` / `netPsi.*` | keep a new head **outside** `regis_net` to load the released weights strictly; a widened `lastConv` needs a mapped state dict (copy the 3 old output channels, zero-init the new ones) |
| BatchNorm | 20 layers, running statistics from pretraining | freeze (`eval()`) for probing and small-batch fine-tuning |
| Both directions | `phi_AB` and `phi_BA` computed in every forward pass | σ_AB and σ_BA are separate outputs; inverse-consistency error `φ_AB∘φ_BA − id` is available for free as a baseline |
| Stochastic layers | none (no dropout) | MC dropout needs injected dropout layers and retraining |

## 13. Known gaps and bugs

| Issue | Where |
| --- | --- |
| Paper says "twelve datasets", Table 6 has thirteen rows | paper p1 vs p12 |
| Paper says 800 + 200 epochs, code runs 801 + 201 | paper p4 vs `train.py:297` |
| Optimiser, batch size, GPUs, wall-clock and IO settings not in the paper | paper p4 |
| No held-out validation during training; the "eval" batch comes from the training set | `train.py:286–292` |
| Script that builds the pre-serialised training tensors is not released | `dataset.py` |
| `finetune_execute` leaves IO weights in the model | `IR/itk_wrapper.py:23` |
| Weights download path is relative to cwd (a new copy per working directory) | `__init__.py:266–273` |
| No MPS support; `config.device` hard-coded in the loss | `IR/config.py:3–6`, `__init__.py:133` |
| `dataset_multi.py:185` sets `img_num = len(self.imgs)`, counting modality keys rather than cases | multiGradICON |
| `ACDCDataset` calls `glob(` and `torch.pad`, which fail | `dataset.py:246, 265` |
| `setup.cfg` says 1.0.4 while tag 1.0.5 exists | `setup.cfg` |
