# Uncertainty-aware uniGradICON: literature and design recon

Goal: make uniGradICON output, in one forward pass, a per-voxel uncertainty on its deformation next to the deformation itself. Then train it from the released weights by probing (backbone frozen), fine-tuning, or probing followed by fine-tuning. This document surveys what exists, fixes what the uncertainty is *of*, compares the designs, and ends with a recommendation and an experiment ladder. Model facts come from [`UNIGRADICON.md`](UNIGRADICON.md). Literature search: about 50 queries plus full reads of the key papers, run on **2026-09-26**. Every URL below was opened.

**Hard constraint throughout:** a real clinical pair never has a ground-truth deformation `u_gt`. The uncertainty therefore has to be produced **without** any error signal at inference, and it must be trainable without one. `u_gt` is used only where it can be manufactured (synthetic warps of real anatomy): to *measure* whether the uncertainty tracks true error, and to *calibrate* its scale.

---

## Literature review: must-read papers

Read in tier order. "Read" gives the depth used for this recon: **full** = the whole paper was read, **abstract** = abstract, method and headline numbers only, **known** = foundational work cited without re-extracting its numbers. Results are the paper's own numbers. Where a cell says "not extracted", the paper has results but this recon did not pull them.

### Tier 1: read first (they define the method, the baseline and the evaluation)

| # | Paper | Read | Summary | Main contribution | Results | Why we read it |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | **uniGradICON**, Tian et al., MICCAI 2024, [arXiv 2403.05780](https://arxiv.org/abs/2403.05780) | full | One GradICON network trained on four datasets (lung CT, knee MRI, brain MRI, abdomen CT) at 175³; zero-shot on new tasks, optional instance optimisation (IO). | First universal deep registration model that works across anatomies and modalities without retraining. | In-distribution, zero-shot → IO: COPDGene mTRE 2.26 → 1.40 mm, OAI Dice 68.9 → 70.3, HCP Dice 76.2 → 78.9, Abdomen Dice 48.3 → 52.2. A universal VoxelMorph trained on the same data reaches HCP Dice 44.2. No uncertainty anywhere. | The model we extend. Details in [`UNIGRADICON.md`](UNIGRADICON.md). |
| 2 | **Tian, Hu & Iglesias**, *Uncertainty Estimation for Pretrained Medical Image Registration Models via Transformation Equivariance*, under review (ICLR 2026 submission), [arXiv 2509.23355](https://arxiv.org/abs/2509.23355) | full | Perturb the input by translations, shears, scales or B-splines (N = 50), map each prediction back, report the variance. Training-free; runs on uniGradICON, SynthMorph and TransMorph. | Test-time uncertainty for any pretrained registration network, with a proof that the variance splits into "intrinsic spread" + "bias jitter". | On uniGradICON, single cases: r 0.71–0.78 against linear ground truth, **r 0.43** against nonlinear; nAURC 0.29–0.47. 12–28 s per pair vs 0.20 s for one pass. Maps agree spatially with MC dropout. | **The baseline to beat.** Same model and grid; its failure mode (a deterministic network gives pure bias jitter) is our opening. |
| 3 | **Dalca, Balakrishnan, Guttag & Sabuncu**, *Unsupervised learning of probabilistic diffeomorphic registration*, MedIA 2019, [arXiv 1903.03545](https://arxiv.org/abs/1903.03545) | full | Probabilistic VoxelMorph: the network outputs mean and diagonal variance of a velocity field; one reparameterised sample per step; Laplacian smoothness prior; image likelihood. | The variational template for unsupervised registration with a learned posterior variance. | Brain atlas registration Dice 0.754 with 0.2 folding voxels, vs ANTs 0.749. Diagonal vs smoothed covariance: no difference at λ = 20. **No quantitative uncertainty result** ("beyond the scope of this paper"). | Our objective is this one adapted to uniGradICON. It also warns that diagonal noise on a *displacement* (not velocity) field is harmful. |
| 4 | **PULPo**, Siegert, Fischer, Heinrich & Baumgartner, MICCAI 2024, [arXiv 2407.10567](https://arxiv.org/abs/2407.10567) | abstract | Hierarchical variational registration: one latent per level of a Laplacian pyramid, KL per level, NCC likelihood. | Calibrated uncertainty from unsupervised probabilistic registration; shows the Dalca model collapses to near-deterministic. | Voxel-level NCC between variance and error: OASIS 0.533 vs 0.210 (Dalca), BraTS 0.497 vs 0.264. Landmark-level: 0.302 vs −0.171. Dalca's max variance 3.4e-5 vs PULPo 1.4. | **Variance collapse is our main technical risk**; this paper names it and fixes it with level weighting. |
| 5 | **Hu et al.**, *Hierarchical Uncertainty Estimation for Learning-based Registration in Neuroimaging*, ICLR 2025, [arXiv 2410.09299](https://arxiv.org/abs/2410.09299), [code](https://github.com/HuXiaoling/Regre4Regis) | full | Network regresses atlas coordinates plus 3 per-axis log-variances with Gaussian NLL; the variance is propagated into affine, B-spline and Demons fits. | A learned Gaussian head beats MC dropout by a wide margin, and uncertainty-weighted fitting improves registration. | Correlation with error: head r 0.476 / ρ 0.601 vs MC dropout r 0.108 / ρ 0.181. 3 variance channels vs 1: Dice 0.790 vs 0.714. Demons trained with direct deformation: Dice 0.767 → 0.799 with uncertainty. | Justifies per-axis variance and dismisses MC dropout. Supervised against NiftyReg; ours is label-free. They list pairwise registration as future work. |
| 6 | **IO-CUE**, Bramlage & Curio, 2025, [arXiv 2506.00918](https://arxiv.org/abs/2506.00918), [code](https://github.com/biggzlar/IO-CUE) | full | Post-hoc variance network on a **frozen** regressor, fed the input and the frozen prediction, trained with a detached Gaussian NLL. | Theory for why a frozen-mean variance head is the right sequential estimate, and why it needs both input and prediction. | NYU depth: ECE 0.00, NLL −2.33, Spearman 0.53, vs 5-model ensemble 0.01 / −1.72 / 0.40. Flip augmentation of the probe set raised OOD AUROC 0.6 → 0.99. | **Theory behind our probe stage.** Caveat: it assumes an MSE-trained base; uniGradICON's is LNCC + GradICON. |
| 7 | **Luo et al.**, *Are Registration Uncertainty and Error Monotonically Associated?*, MICCAI 2020, [arXiv 1908.07709](https://arxiv.org/abs/1908.07709) | abstract | Gaussian-process registration on brain-shift ultrasound (23 cases); correlates posterior std with landmark error. | Shows registration uncertainty need not rise with error; asks for this to be tested before uncertainty is trusted. | Point-wise Spearman 0.29 (manual landmarks), 0.40 (automatic); patch-wise "consistently low". | The question our synthetic bench answers at scale; sets expectations for real-landmark correlations. |
| 8 | **Kumar et al.**, *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (LP-FT), ICLR 2022, [arXiv 2202.10054](https://arxiv.org/abs/2202.10054) | abstract | Theory and experiments on linear probing vs fine-tuning vs probe-then-fine-tune. | Fine-tuning with a random head distorts features and hurts OOD; probing first fixes this. | Over 10 distribution shifts: fine-tuning vs probing moved ID accuracy 83 → 85 % but OOD 66 → 59 %; LP-FT +1 % ID and +10 % OOD over fine-tuning. | Our training order. Never tested for registration or for uncertainty, so any result is new. |

### Tier 2: read next (design alternatives, error-prediction lineage, uses)

| # | Paper | Read | Summary | Main contribution | Results | Why we read it |
| ---: | --- | --- | --- | --- | --- | --- |
| 9 | **Eppenhof & Pluim**, *Error estimation of deformable image registration of pulmonary CT scans using CNNs*, J. Med. Imaging 2018, [DOI](https://doi.org/10.1117/1.JMI.5.2.024003) | abstract | 3D patch CNN trained on synthetically deformed lung CT predicts a dense registration-error map. | Synthetic-deformation supervision for dense error prediction, validated on real landmarks. | RMSD 0.51 mm vs synthetic error, 0.66 mm vs landmark TRE. | The direct ancestor of this repo's Phase III, and a model for sim-to-real validation. |
| 10 | **Luo et al.**, *On the Applicability of Registration Uncertainty*, MICCAI 2019, [arXiv 1803.05266](https://arxiv.org/abs/1803.05266) | abstract | Analysis of what registration uncertainty can and cannot say. | Transformation uncertainty and label uncertainty are different quantities; report the former. | Conceptual analysis; not extracted. | The reason §0 separates the four kinds of uncertainty. |
| 11 | **Zhang et al.**, *Heteroscedastic Uncertainty Estimation Framework for Unsupervised Registration*, MICCAI 2024, [arXiv 2312.00836](https://arxiv.org/abs/2312.00836) | abstract | A registration network and an intensity-variance network trained alternately; the variance reweights the image loss by relative signal-to-noise; β-NLL on the variance net. | Stable heteroscedastic training for unsupervised registration. | Evaluated by sparsification and AUSE; numbers not extracted. | The label-free intensity-variance baseline, and the alternating-optimiser recipe. |
| 12 | **AC-CAR**, Wang, Luo, Du & Qin, IEEE TMI 2026, [arXiv 2601.05981](https://arxiv.org/abs/2601.05981), [code](https://github.com/Yinsong0510/AC-CAR) | full | Contrast-agnostic registration with a variance decoder that shares the registration encoder; β-NLL on the intensity residual. | Uncertainty as a cheap add-on to a registration backbone. | Variance branch costs +1.2 M parameters (2.83 → 4.05 M) and +0.04 s (2.538 → 2.579 s). Uncertainty judged only qualitatively (sparsification plots, no AUSE, no baseline). | Evidence that a shared-encoder head is cheap; a warning that uncertainty claims need numbers. |
| 13 | **CONReg**, Gheiji et al., J. Imaging Inform. Med. 2026, [DOI](https://doi.org/10.1007/s10278-026-01878-3) | abstract | U-Net predicts the displacement plus lower and upper quantiles; split-conformal calibration gives 95 % keypoint intervals. | Distribution-free coverage guarantees for registration. | Empirical coverage 0.92–0.98; keypoints flagged as certain had significantly lower TRE (p < 0.05). | The calibration layer for §3.5. |
| 14 | **Loi et al.**, *Inverse consistency error for validating deformable image registration*, Phys. Imaging Radiat. Oncol. 2026, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12925039/) | abstract | Uses the forward–backward inconsistency map as an error proxy in radiotherapy. | A free, label-free error proxy for any bidirectional registration. | R = 0.85 vs true error, 0.68 vs TRE; underestimates deformations above 15 mm. | Free baseline: uniGradICON computes both directions every pass. |
| 15 | **Chen et al.**, *From Registration Uncertainty to Segmentation Uncertainty*, ISBI 2024, [arXiv 2403.05111](https://arxiv.org/abs/2403.05111), [code](https://github.com/junyuchen245/Registration_Uncertainty) | abstract | A small auxiliary network on a fixed registration network turns registration uncertainty into segmentation uncertainty. | Auxiliary heads on frozen registration networks, and a downstream use of sampled fields. | Epistemic and aleatoric segmentation uncertainty track label-propagation error; numbers not extracted. | Precedent for a frozen-backbone head; the label-propagation evaluation in §5. |
| 16 | **Salari et al.**, *FocalErrorNet*, MICCAI 2023, [arXiv 2307.14520](https://arxiv.org/abs/2307.14520) | abstract | Focal-modulation network regresses MRI–ultrasound registration error, with MC dropout on its own estimate. | Error prediction with uncertainty on the prediction itself. | 0.59 ± 0.57 mm vs 1.69 ± 1.37 mm for a CNN baseline. | Error-prediction baseline family on real clinical data. |
| 17 | **RUGI**, González, Bates, Ng & Tang, 2026, [arXiv 2609.28081](https://arxiv.org/abs/2609.28081) | abstract | Recursive refinement of pretrained VoxelMorph / TransMorph / CycleMorph, gated by uncertainty or residual. | Uncertainty used to decide *where and whether* to refine. | 27–37 % lower MSE on cardiac MRI and echo. Notes that Tian 2025 did not use their map for refinement. | The template for σ-guided IO (E5). |
| 18 | **The LUMirage**, Jena, Chaudhari & Gee, 2025, [arXiv 2512.15505](https://arxiv.org/abs/2512.15505) | abstract | Independent re-evaluation of zero-shot claims in the LUMIR brain challenge. | Shows deep registration fails silently under contrast, resolution and preprocessing shifts. | Significant drops on T2, T2* and FLAIR (Cohen's d 0.7–1.5); models fail to run on 0.6 mm images. | Defines the OOD conditions σ should flag (E4). |
| 19 | **Chaudhary et al.**, *Uncertainty-Aware Test-Time Adaptation for Inverse Consistent Diffeomorphic Lung Image Registration*, 2024, [arXiv 2411.07567](https://arxiv.org/abs/2411.07567) | abstract | MC-dropout uncertainty guides test-time adaptation of an inverse-consistent lung network. | Uncertainty-guided test-time adaptation. | Lung Dice 0.966 vs 0.953 for VoxelMorph and TransMorph on 675 COPDGene subjects. | Closest existing "uncertainty-guided IO"; same data as uniGradICON's lung corpus. |

### Tier 3: background (read as needed)

| # | Paper | Read | Main contribution | Results | Why |
| ---: | --- | --- | --- | --- | --- |
| 20 | **GradICON**, Tian et al., CVPR 2023, [arXiv 2206.05897](https://arxiv.org/abs/2206.05897) | known | Gradient inverse-consistency regulariser that yields approximately diffeomorphic maps without explicit smoothness terms. | Not extracted; uniGradICON reuses its network and defaults unchanged. | The loss our head trains against. |
| 21 | **Kendall & Gal**, NeurIPS 2017, [arXiv 1703.04977](https://arxiv.org/abs/1703.04977) | known | Aleatoric vs epistemic uncertainty; heteroscedastic NLL heads. | Not extracted. | The vocabulary and the NLL form. |
| 22 | **β-NLL**, Seitzer et al., ICLR 2022, [arXiv 2203.09168](https://arxiv.org/abs/2203.09168) | known | Plain Gaussian NLL under-fits the mean in high-variance regions; weighting by σ^{2β} fixes it. | Not extracted. | Used by Zhang 2024 and AC-CAR; relevant once μ is unfrozen. |
| 23 | **Krebs et al.**, *Learning a Probabilistic Model for Diffeomorphic Registration*, IEEE TMI 2019, [arXiv 1812.07460](https://arxiv.org/abs/1812.07460) | known | Conditional VAE over a low-dimensional latent deformation. | Not extracted. | Alternative latent-variable formulation. |
| 24 | **Rivetti et al.**, PMB 2024, [DOI](https://doi.org/10.1088/1361-6560/ad4c4f) | abstract | Multivariate-normal output over the displacement field. | Beats MC dropout and MC B-spline; KL 0.15. | A non-diagonal alternative to §3.3. |
| 25 | **Sokooti et al.**, MedIA 2019, [arXiv 1905.07624](https://arxiv.org/abs/1905.07624) | abstract | Regression forests predicting local registration error from transformation and dissimilarity features. | Quantitative local error on chest CT; numbers not extracted. | Classical error-prediction precedent. |
| 26 | **Bierbrier, Gueziri & Collins**, MedIA 81:102531, 2022 | abstract | Taxonomy and scoping review of registration error and confidence estimation. | 20 of 570 screened studies included. | Framing and terminology for the related-work section. |
| 27 | **Gal & Ghahramani**, ICML 2016, [arXiv 1506.02142](https://arxiv.org/abs/1506.02142) | known | Dropout as approximate Bayesian inference (MC dropout). | Not extracted. | The weak baseline in Hu 2025; needs injected dropout in uniGradICON. |

**Suggested order for a first pass:** 1 → 2 → 3 → 4 → 5 → 6, which covers the model, the baseline, the objective, its failure mode, the head, and the probe theory. Then 7 and 8 for the evaluation question and the training order, and 13 and 14 for calibration and the free baseline.

---

## 0. What is uncertain about what

uniGradICON returns `phi_AB`, a map from coordinates on **B's lattice** (the fixed image) into A's space, so `warped_A(x) = A(φ_AB(x))`. `phi_AB_vectorfield` is that map on the 175³ grid, in normalised [0, 1] coordinates. The displacement is `u(x) = φ_AB(x) − x`. Because `x` is a fixed grid, **Var[φ(x)] = Var[u(x)]** and `‖φ_gt − φ_pred‖ = ‖u_gt − u_pred‖`: uncertainty "of φ" and uncertainty "of u" are the same number. Every result must still pin down the following:

| Axis | Choices | uniGradICON specifics | Our convention |
| --- | --- | --- | --- |
| Lattice | fixed (B) vs moving (A) | `φ_AB` lives on B, `φ_BA` on A | report on the fixed lattice (B), matching this repo's `source` convention |
| Direction | forward vs inverse | both computed in every forward pass | a σ for each direction; report `σ_AB` |
| Units | normalised → 175³ voxels (×174) → native voxels (×(N−1) per axis) → mm | spacing `1/174` | σ is rescaled **per axis** before any magnitude, exactly like `u` in `create_unigrad_synth_data.py:125–141` |
| Quantity | (a) transformation variance `Var[u(x)]`; (b) expected error `E‖u_gt − u‖`; (c) intensity-residual variance `Var[A∘φ − B]`; (d) label uncertainty | these are four different things ([Luo 2019](https://arxiv.org/abs/1803.05266)) | **(a) is the output**; (b) is how it is *judged*; (c) is a baseline; (d) is a downstream use |
| Which field | final composed `u = u_ψ(x) + u_φ(x + u_ψ(x))` vs the last step's residual `u_ψ` | a head on the last UNet sees only `u_ψ` | σ is defined on the **composed** field; how to get there is §3.4 |
| Scalar vs per axis | `tr Σ`, `√tr Σ`, 3 diagonal entries | — | 3 per-axis log-variances ([Hu 2025](https://arxiv.org/abs/2410.09299): 3 channels beat 1, Dice 0.790 vs 0.714); report `√tr Σ` in voxels for maps |

---

## 1. Has this been done?

**No trained uncertainty output exists for any registration foundation model**, in any of the probe, fine-tune or probe-then-fine-tune variants. We checked uniGradICON, multiGradICON, SynthMorph, BrainMorph, TransMorph variants, DINO-Reg, SAME++ and IMPACT.

Queries run (2026-09-26): "uniGradICON uncertainty", "registration foundation model uncertainty", "uncertainty-aware image registration foundation model", "probabilistic uniGradICON", "GradICON uncertainty", "multiGradICON", "SynthMorph / BrainMorph uncertainty", "DINO-Reg OR SAME++ OR IMPACT uncertainty", "test-time adaptation registration uncertainty pretrained foundation", "registration failure detection foundation model", "pretrained registration network post-hoc variance head frozen backbone", "fine-tuning uniGradICON", "uniGradICON OOD". One search summary claimed uniGradICON "provides a probabilistic version". Neither the paper nor the README supports this.

**Closest work**

| Paper | What it does | Relevance |
| --- | --- | --- |
| **Tian, Hu & Iglesias 2025/26**, *Uncertainty Estimation for Pretrained Medical Image Registration Models via Transformation Equivariance*, [arXiv 2509.23355](https://arxiv.org/abs/2509.23355) (v2 31 Jan 2026) | Test-time, training-free. Perturb the input by τ, map the prediction back, report `√tr Var_τ[τ∘f(A∘τ, B)]`. N = 50 per perturbation type. Runs on uniGradICON at 175³. | **Must-compare baseline.** Same model, same grid. Details below. |
| **Hu et al., ICLR 2025**, *Hierarchical Uncertainty Estimation for Learning-based Registration in Neuroimaging*, [arXiv 2410.09299](https://arxiv.org/abs/2410.09299), [code](https://github.com/HuXiaoling/Regre4Regis) | Gaussian head (μ + 3 log σ²) on a coordinate-regression network trained from scratch with NLL against NiftyReg ground truth; propagates σ to affine / B-spline / Demons fits. | Strongest evidence that a learned Gaussian head beats MC dropout (r 0.476 vs 0.108; ρ 0.601 vs 0.181). Future work in their words: "extend the framework to pairwise registration of any two input images." |
| **Chaudhary et al. 2024**, [arXiv 2411.07567](https://arxiv.org/abs/2411.07567) | MC-dropout uncertainty guides test-time adaptation of an inverse-consistent lung network (COPDGene). | Not a foundation model; uses a dataset from uniGradICON's corpus. |
| **Hu, Yu et al., IJCNN 2025**, [arXiv 2505.06527](https://arxiv.org/abs/2505.06527), [code](https://github.com/Promise13/fm_sam) | Retrains uniGradICON with sharpness-aware minimisation for better generalisation. | Only other uniGradICON retraining found; no uncertainty. |
| **multiGradICON**, [arXiv 2408.00221](https://arxiv.org/abs/2408.00221); **BrainMorph**, [arXiv 2405.14019](https://arxiv.org/abs/2405.14019); **SAME++**, [arXiv 2311.14986](https://arxiv.org/abs/2311.14986) | Foundation or large registration models. | No uncertainty component. |

### Tian, Hu & Iglesias in detail (the paper to answer)

**Status:**
- Submitted to ICLR 2026. The OpenReview PDF header reads "Under review".
- No decision or reviews are visible without logging in: [OpenReview forum](https://openreview.net/forum?id=4dARndGskg).
- arXiv v2 carries no venue note, and Semantic Scholar lists no venue.
- The code was promised "upon acceptance"; no repository exists on GitHub.
- We cannot say whether it was accepted or rejected. The retitled v2 after the ICLR decision window fits a resubmission, but that is inference.

**Method:**
- Perturbations, N = 50 each:
  - translation: up to 1 % of the image shape;
  - shear: U(−0.02, 0.02);
  - scale: U(0.9, 1.1);
  - B-spline: 10 px control grid, displacements U(−12.5, 12.5) px.
- The map is `u(y) = √tr S(y)`, where S is the sample covariance of `g_n = τ_n ∘ φ̂_n`.
- Error is `‖φ_gt − φ̂‖₂` per voxel on the 175³ grid (voxels), averaged in an ROI.

**Decomposition (Lemma 4.1).**
- Variance = intrinsic spread `E_τ[J_τ Σ_ε J_τᵀ]` + bias jitter `Cov_τ[J_τ μ_ε]`.
- For a deterministic network `Σ_ε = 0`, so the map is **pure bias jitter**.
- Their App. C.3 shows why it can fail: true error is `‖μ_ε‖² + tr Σ_ε`, and a network that is "more wrong than uncertain" (large, stable bias) produces a small map.
- v1 also listed a limitation dropped in v2: the map "may be blind to errors that are themselves transformation-equivariant (e.g., in translation- or rotation-equivariant registration networks)". GradICON-regularised uniGradICON is approximately equivariant by design.

**Ground truth used in their evaluation.** All at 175³:
- translation 10 %;
- affine (shear ±0.1, scale 0.8–1.2);
- nonlinear: a composition of two B-spline fields (10 px, ±12.5 px);
- "real": an ANTs affine + nonrigid transform re-applied to the source.

Data: 11-repository brain set (220 volumes, 110 used), ACDC (100), L2R Abdomen (30), IXI, BraTS-Reg, ThoraxCBCT.

**Numbers (uniGradICON).** Single-case values from the text:

| Ground truth | Pearson r | Spearman ρ | nAURC |
| --- | --- | --- | --- |
| translation / affine / real | 0.71–0.78 | 0.49–0.50 | 0.29–0.47 across ground-truth types |
| **nonlinear deformation** | **0.43** | 0.48 | (same range) |

Dataset-level numbers are in figures only (Figs. 5, 6, 8), so the protocol has to be rerun, not copied.

**Runtime:** 12.1–27.7 s per pair against 0.20 s for one uniGradICON pass (RTX 4500 Ada).

**Stated limitations (v2):**
- only first- and second-order moments;
- "does not disentangle different sources of uncertainty in the manner of Bayesian approaches";
- not evaluated on optimisation-based registration.

**What this means for us:**
- A single-pass learned σ is 60–140× cheaper.
- It can in principle carry the systematic error that their map cannot.
- It can be evaluated on **their** protocol: same perturbation distributions, 175³ lattice, errors in voxels, ROI-averaged r / ρ, and nAURC.

---

## 2. Prior art closest to what this repo has done (error prediction)

This repo's Phase III (external `UNet3D` regressing `‖u_gt − u_pred‖` from source, moving and `u_pred`; HCP test r = 0.874, masked MAE 0.338 voxels) belongs to the **error-prediction** lineage.

| Paper | Method | Result | Relation to us |
| --- | --- | --- | --- |
| Eppenhof & Pluim 2018, *J. Med. Imaging* 5(2):024003, [DOI](https://doi.org/10.1117/1.JMI.5.2.024003) | patch 3D CNN trained on synthetically deformed lung CT, dense error map | RMSD 0.51 mm vs synthetic error, 0.66 mm vs landmark TRE | same recipe; also validated on real landmarks |
| Sokooti et al. 2019, *MedIA*, [arXiv 1905.07624](https://arxiv.org/abs/1905.07624), [code](https://github.com/hsokooti/RegUn) | regression forest on transformation and dissimilarity features | local error on chest CT | classical precursor |
| Sokooti, Yousefi et al. 2021, *IEEE Access* 9:62008, [IEEE](https://ieeexplore.ieee.org/document/9408621/) | multi-resolution ConvLSTM classifying 0–3 / 3–6 / >6 mm error, synthetic DVFs | multi-resolution helps | — |
| Salari et al., MICCAI 2023 (FocalErrorNet), [arXiv 2307.14520](https://arxiv.org/abs/2307.14520); IEEE IUS 2023, [arXiv 2308.10784](https://arxiv.org/abs/2308.10784) | focal-modulation / Swin UNETR error regressors for MRI–ultrasound, MC dropout on the estimate | 0.59 ± 0.57 mm vs 1.69 ± 1.37 mm CNN baseline | dense error maps on real clinical data |
| Chen et al., ISBI 2024, [arXiv 2403.05111](https://arxiv.org/abs/2403.05111), [code](https://github.com/junyuchen245/Registration_Uncertainty) | auxiliary network on a **fixed** registration network | epistemic + aleatoric segmentation uncertainty tracks label-propagation error | auxiliary head on a frozen network |
| Loi et al. 2026, *Phys. Imaging Radiat. Oncol.* 37:100916, [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12925039/) | inverse-consistency error map as an error proxy | R = 0.85 vs true error, 0.68 vs TRE; underestimates > 15 mm | **free baseline** on uniGradICON (both directions are computed) |
| Bierbrier, Gueziri & Collins 2022, *MedIA* 81:102531 (DOI 10.1016/j.media.2022.102531, inferred) | taxonomy and scoping review | 20 of 570 studies | framing |
| RegQCNET 2020, [arXiv 2005.06835](https://arxiv.org/abs/2005.06835) | scalar affine-QC regressor | — | global, affine only |

**Where we sit.**
- No error-map paper targets a foundation registration model.
- Our r = 0.874 is against **nonlinear** synthetic ground truth. That is the regime where Tian 2025 reports r = 0.43 for uniGradICON, but the ground-truth distributions differ (TorchIO classes at native resolution vs B-spline compositions at 175³), so the two numbers are not yet comparable.
- Error regression needs `u_gt`-like supervision, which is why it is the **baseline**, not the primary route.

---

## 3. Ways to make uniGradICON output (μ, σ) directly

### 3.1 Options

| Route | Output | Objective | Needs an error signal? | Captures | Fit to uniGradICON | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **Variational Gaussian over `u`** ([Dalca 2019](https://arxiv.org/abs/1903.03545), [Krebs 2019](https://arxiv.org/abs/1812.07460), [PULPo 2024](https://arxiv.org/abs/2407.10567), [Rivetti 2024](https://doi.org/10.1088/1361-6560/ad4c4f)) | μ (3) + log σ² (3) per voxel | registration loss on a reparameterised sample + KL to a smoothness prior | **no** | transformation variance consistent with the images under the model | head on the last UNet's features; frozen μ keeps registration quality | **primary** |
| Heteroscedastic intensity residual ([Zhang 2024](https://arxiv.org/abs/2312.00836), [AC-CAR 2026](https://arxiv.org/abs/2601.05981), [Gong 2022](https://openaccess.thecvf.com/content/WACV2022/papers/Gong_Uncertainty_Learning_Towards_Unsupervised_Deformable_Medical_Image_Registration_WACV_2022_paper.pdf)) | log σ²_I of `A∘φ − B` | β-NLL ([Seitzer 2022](https://arxiv.org/abs/2203.09168)) on the residual | no | image-level ambiguity, **not** transformation variance | cheap (+1.2 M params in AC-CAR) | label-free baseline |
| MC dropout ([Gal 2016](https://arxiv.org/abs/1506.02142)), deep ensembles, last-layer Laplace, SWAG | sample variance | retraining with dropout / several models | no | epistemic | uniGradICON has **no dropout**; 4 × 70.7 M for ensembles | baseline only (weak: Hu 2025 r 0.108) |
| Test-time equivariance (Tian 2025) | `√tr Var_τ` | none | no | bias jitter | 50 passes | training-free baseline |
| Inverse-consistency error | `‖φ_AB∘φ_BA − id‖` | none | no | inconsistency | free | training-free baseline |
| Error-prediction head ([DeVries 2018](https://arxiv.org/abs/1802.04865), [Yoo 2019](https://arxiv.org/abs/1905.03677), [IO-CUE 2025](https://arxiv.org/abs/2506.00918), our Phase III) | `ê(x)` | regression / NLL against `u_gt` | **yes** (synthetic only) | expected error | head on features or external | supervised baseline |
| Conformal / quantile layer ([CONReg 2026](https://doi.org/10.1007/s10278-026-01878-3)) | calibrated intervals | calibration split | yes, once, on a calibration set | coverage guarantee | sits on top of any σ | **calibration step** (§3.5) |
| Deep evidential regression ([Amini 2020](https://arxiv.org/abs/1910.02600)) | NIG parameters | evidential NLL | yes | both, in principle | no registration precedent found | not pursued |

### 3.2 Output layout: 6 channels, not 9

The mean *is* the displacement, so a Gaussian output is 3 + 3 = 6 channels.

| Layout | Output | Pretrained field | Checkpoint | Probe trains |
| --- | --- | --- | --- | --- |
| A. widen `lastConv` 18 → 6 | μ (3) + log σ² (3) | μ channels copied from the old 18→3 weights (keep the ÷10 scaling), σ channels new | mapped state dict | the 3 new output channels |
| **B. add-on head 18 → 3 (recommended)** | existing `u_ψ` (3) + log σ² (3) | untouched | strict load; head lives outside `regis_net` | the head |
| C. add-on head 18 → 6 | existing `u_ψ` (3) + Δμ (3) + log σ² (3) = 9 | refined by a residual Δμ | strict load | the head |

- **A and B are the same model while the mean is frozen.** B is simpler and keeps the released weights loadable as-is.
- **C** is the fine-tune variant: the head may also correct the field where it is confident the field is wrong.

**Head input.** At minimum, the 18-channel input of `net.regis_net.netPsi.net.lastConv`, captured by a forward hook (2 skip channels + 16 decoder channels at 175³). IO-CUE argues for also feeding the raw inputs and the frozen prediction:
- without the input, the head learns only marginal variance;
- the prediction carries "quasi-epistemic" signal (distance from the model's output manifold);
- a head comparable in capacity to the base helps.

So the head input is `[features(18), A∘φ (warped moving), B, u (3)]` = 23 channels, and the head is a small 3D U-Net (a few M params). This is also where this repo's `UNet3D` (5 in / 1 out) already sits: it is IO-CUE's input layout without the features.

### 3.3 Objective for the variational route

The formulation follows Dalca 2019 (Eq. 6), adapted to uniGradICON's loss:

`L = sim(A ∘ φ_ũ, B) + sim(B ∘ φ_ṽ, A) + λ · GradICON(φ_μ) + β · KL( N(μ, Σ) ‖ p(u) )`, with `ũ = μ + σ ⊙ (G ∗ ε)`, `ε ~ N(0, I)`

Five points need care:

1. **Noise must be spatially smooth.** Dalca: "diagonal covariances would likely have negative effects … if z was modelled as the displacement field itself". uniGradICON predicts displacements, not velocities. White per-voxel noise on `u` also makes the finite-difference Jacobian in the GradICON term explode. So:
   - sample smoothed noise `G ∗ ε` (Gaussian kernel, as in Dalca's non-diagonal variant);
   - apply GradICON to **μ** (unchanged from pretraining), and the similarity term to the **sample**.
2. **Prior.** `p(u) = N(0, (λ_p L)⁻¹)` with L the grid Laplacian, as in Dalca. For diagonal Σ the KL is `½[λ_p tr(D Σ) − Σ log σ² + μᵀ Λ μ] + const`; with μ frozen, only the first two terms act on σ. The `μᵀΛμ` term is dropped in the probe (μ fixed) and in fine-tuning (GradICON already regularises μ).
3. **Collapse is the main risk.** PULPo reports that Dalca's model trained with NCC let σ "converge to a very small number making the model almost deterministic" (max var 3.4e-5 vs 1.4 for PULPo). With LNCC as the "likelihood", the variance scale is set by the ratio β / similarity weight, which has no physical meaning. Mitigations:
   - monitor the log σ statistics from step 1;
   - sweep β;
   - use PULPo-style level weighting;
   - rely on the calibration step (§3.5) for absolute scale.
4. **Both directions.** The same head runs on the reverse pass, giving `σ_AB` and `σ_BA`.
5. **K = 1 sample** per iteration (Dalca) is enough for training. At test time, K samples cost only a re-composition, not new network passes (§3.4).

### 3.4 Cascade: from the last residual to the composed field

The head sits on step 2, whose residual is `u_ψ`. The composed field is `u(x) = u_ψ(x) + u_φ(x + u_ψ(x))`. Sampling `ũ_ψ = μ_ψ + σ_ψ ⊙ (G∗ε)` and re-composing gives samples of the composed field without re-running any UNet. So the composed variance is available by **Monte Carlo over composition only** (cheap), or to first order analytically as `(I + J_φ) Σ_ψ (I + J_φ)ᵀ`.

What this does **not** capture is uncertainty in steps 1a–1c. Whether that matters (most of the large motion is resolved in step 1) is an empirical question and a small contribution in itself.

### 3.5 Calibration (the one place `u_gt` enters)

Because the LNCC-based objective leaves σ's scale arbitrary, fit **one** calibration map on the synthetic bench, where `u_gt` is known, and freeze it:
- a global temperature `s` with `σ_cal = s·σ`, fitted by NLL of `u_gt` under `N(μ, σ_cal²)`; or
- a split-conformal quantile per deformation class ([CONReg 2026](https://doi.org/10.1007/s10278-026-01878-3): coverage 0.92–0.98 on keypoint intervals).

The calibration set is a held-out split of the synthetic data. It is never used at inference, and real pairs never need `u_gt`. Whether a synthetic-fitted calibration transfers to real data is itself a measured result (§5).

---

## 4. Training strategies

All start from the released `Step_2_final.trch`; **no pretraining from scratch** (≈ 243,500 steps on 4 GPUs, [`UNIGRADICON.md`](UNIGRADICON.md) §6–7).

| Strategy | Trainable | Registration quality | Compute | Evidence |
| --- | --- | --- | --- | --- |
| **Probe** | head only; 70.7 M backbone frozen, BatchNorm in `eval()` | unchanged by construction (μ frozen) | backbone under `no_grad`, gradients through head + warp + LNCC only → one GPU, batch 1–2 | IO-CUE: sequential fitting of a variance model on a frozen mean is the MLE when the mean is an MSE fit. Caveat: uniGradICON's mean minimises LNCC + GradICON, not MSE, so σ will partly absorb bias. |
| **LP-FT** | probe first, then everything at lr ≤ 5e-5 (the pretraining lr), BatchNorm frozen | must be monitored (Dice / TRE of μ) | full graph of 4 UNets × 2 directions → batch 1 per 40–80 GB GPU with accumulation | [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054): FT vs LP moved ID accuracy 83 → 85 % but OOD 66 → 59 %; LP-FT gained ≈ 1 % ID and ≈ 10 % OOD over FT. **Never tested for registration or for uncertainty.** |
| Full FT | head + backbone from the start | same risk, larger | same | comparison arm only |
| Parameter-efficient (adapters / LoRA on 3D convs) | adapters + head | small drift | between probe and FT | no registration precedent found; [Zhang, Zhao & Tao 2026](https://arxiv.org/abs/2603.26393) keep a backbone frozen and tune light U-Nets by IO (Learn2Reg LUMIR25, 4th) |

**Interaction with instance optimisation (IO):**
- IO changes all weights (Adam 2e-5, 50 steps) and `finetune_execute` does not restore them ([`UNIGRADICON.md`](UNIGRADICON.md) §8). Deep-copy the model per pair.
- Options, from simplest to hardest:
  1. run IO on μ as today, then evaluate the frozen head on the IO'd features;
  2. use σ as a gate: skip or shorten IO where σ is low;
  3. weight the IO similarity by σ⁻¹, or mask it, following [RUGI 2026](https://arxiv.org/abs/2609.28081), which cuts MSE by 27–37 % on pretrained VoxelMorph / TransMorph / CycleMorph.

---

## 5. Evaluating uncertainty when `u_gt` does not exist

| Check | Data | What it measures | `u_gt` needed | Notes |
| --- | --- | --- | --- | --- |
| **Synthetic bench** | HCP synth (this repo: 1113 subjects, 857 / 109 / 147, five classes: none, rigid, affine, elastic, affine + elastic) | Pearson / Spearman of σ vs `‖u_gt − μ‖` in the brain mask; sparsification and AUSE; NLL and coverage of `u_gt` under `N(μ, σ_cal²)`; per class and per region | yes (manufactured) | answers [Luo 2020](https://arxiv.org/abs/1908.07709) ("Are registration uncertainty and error monotonically associated?"; their ρ_s ≈ 0.29 on real landmarks) at scale |
| **Tian 2025 protocol** | the same brains resampled to 175³ with their translation / affine / B-spline ground truth | r, ρ, AURC, nAURC | yes (manufactured) | head-to-head with the published baseline |
| Landmark TRE | DIR-Lab COPDgene (10 pairs × 300 landmarks), L2R NLST (inhale / exhale, ≈ 2,000 automatic + 100+ manual keypoints per case) | sparsification of σ against landmark error; conformal coverage | no | lung is in-distribution (COPDGene) and type-1 OOD (NLST) |
| Label-propagation error | HCP `aparc+aseg` (downloaded), L2R AbdomenCTCT (13 organs), L2R OASIS (35 labels), IXI (TransMorph labels) | does high σ coincide with boundary disagreement after warping labels | no | sparse and indirect: label error ≠ transformation error (Luo 2019) |
| OOD flagging | uniGradICON's type 1 / 2 / 3 splits; [LUMirage 2025](https://arxiv.org/abs/2512.15505) conditions (T2, T2*, FLAIR, 0.6 mm, preprocessing changes) | does mean σ rise when registration quality falls | no | AUROC of image-level σ vs failure (Dice drop) |
| Registration quality of μ | HCP-test, L2R-Abdomen, DIR-Lab | Dice, TRE, `%|J|<0` must not regress | no | guards LP-FT and FT |

**Metrics to add to this repo:** sparsification curves and AUSE, nAURC (Tian's definition `(Unc − Oracle)/(Random − Oracle)`), Gaussian NLL, coverage at 1σ / 2σ, and ECE for regression. Existing masked MAE / RMSE / Pearson carry over.

**Expected ranges (to set honest targets):**
- On synthetic data: Hu 2025's learned head reaches r 0.48 / ρ 0.60, and Tian 2025's uniGradICON reaches r 0.43 on nonlinear ground truth.
- On real landmarks, correlations are weak to moderate everywhere: Luo 2020 ρ_s 0.29 / 0.40, PULPo landmark NCC 0.23–0.30.

---

## 6. Robustness and refinement uses

| Use | Idea | Evidence so far |
| --- | --- | --- |
| **σ-guided IO** | refine only where σ is high; stop early where σ is low; weight IO's similarity by σ⁻¹ | uniGradICON's IO raises folding (Abdomen `%|J|<0` 0.31 → 0.96, CT–MR 0.04 → 0.61). [RUGI 2026](https://arxiv.org/abs/2609.28081): uncertainty-gated recursive refinement of pretrained networks, −27–37 % MSE. [Chaudhary 2024](https://arxiv.org/abs/2411.07567): uncertainty-guided test-time adaptation, lung Dice 0.966 vs 0.953. Tian 2025 did not use their map for refinement (noted by RUGI). |
| **Failure / OOD flagging** | image-level summary of σ predicts registration failure before anyone looks | LUMirage: zero-shot models drop sharply on T2 / T2* / FLAIR (Cohen's d 0.7–1.5), fail at 0.6 mm, and are sensitive to preprocessing, silently |
| Precision-weighted smoothing | `K ⋆ (σ⁻² ⊙ μ) / (K ⋆ σ⁻²)` (Hu 2025's Demons formula) as a cheap post-hoc correction | Hu 2025: Demons Dice 0.767 → 0.799 with uncertainty |
| Downstream propagation | sample fields → label / dose / segmentation uncertainty | [Chen ISBI 2024](https://arxiv.org/abs/2403.05111) |
| Augmented probe data | train the head on blur / contrast / flip / modality-perturbed pairs so σ grows off-distribution | IO-CUE: flip augmentation raised OOD AUROC 0.6 → 0.99 |

---

## 7. Dataset for the proof of concept

Criteria: in uniGradICON's pretraining corpus (so the backbone is in-distribution and σ is not confounded with domain shift), small enough for one GPU, with a way to judge σ.

| Dataset | In corpus | Size | Judge σ by | Access | Role |
| --- | --- | --- | --- | --- | --- |
| **HCP T1w** | yes (train) | 1113 subjects already on the PVC; ~0.7 mm; `aparc+aseg` included in native T1w space | synthetic `u_gt` (Phase I/II exist) + label propagation | done | **primary**: probe training on real inter-subject pairs, synthetic bench, label error |
| **L2R AbdomenCTCT** | yes (train) | 30 train + 20 test, 13 organ labels, 2 mm, 192×160×256, ≈ 1 GB | labels | [Learn2Reg](https://learn2reg.grand-challenge.org/Datasets/) | second anatomy / modality (CT); tiny |
| IXI | type-1 OOD | 115 test (TransMorph split), atlas | labels; this repo's IO delta | local (IXI track) | OOD check, and IO-delta comparison with the existing IXI experiment |
| L2R OASIS | type-1 OOD | 416 T1w, 35 labels, 1 mm, ≈ 1.3 GB | labels | Learn2Reg | OOD brain |
| DIR-Lab COPDgene | in-distribution test | 10 pairs × 300 landmarks | TRE | [request form](https://www.dir-lab.com/ReferenceData.html) | landmark ground truth in the corpus's own anatomy |
| L2R NLST | type-1 OOD | 210 inhale / exhale cases, keypoints | TRE | Learn2Reg | landmark OOD |
| LUMIR / LUMIR25 | no | 3,384 T1w, 1 mm, ≈ 52 GB; validation labels private | leaderboard only | [GitHub](https://github.com/JHU-MedImage-Reg/LUMIR_L2R) | later, if a leaderboard result is wanted |
| OAI / OAI-ZIB | OAI yes (train) | OAI-ZIB: 507 DESS knees, 4 bone / cartilage labels | labels | NDA account + data-use agreement; OAI-ZIB public | optional third anatomy |

**Recommendation:** HCP only, for E0–E3.
- Probe training: 200 subjects (≈ 20k random inter-subject pairs are available), sampled like uniGradICON's loader.
- Synthetic bench: the existing 147-subject Test split, plus a 109-subject Val split reserved for calibration.
- Real-data check: `aparc+aseg` on 32 held-out subjects (mirroring HCP-test).

Add L2R AbdomenCTCT and IXI in E4. Request DIR-Lab access now, because it takes time.

---

## 8. What transfers from this repo

| Asset | Path | New role |
| --- | --- | --- |
| Phase I TorchIO synth (`u_gt` on the fixed lattice, five classes, subject-level splits) | `experiments/synth-data-gen/torchio/create_synth_data.py` | **calibration and evaluation bench**; not supervision |
| Phase II uniGradICON runner (p99 clip, 175³ resize, `net(moving, source)`, `phi − id`, per-axis rescale to native voxels) | `experiments/error-map-gen/unigrad-synth/create_unigrad_synth_data.py:110–141` | reuse for loading, preprocessing and unit conversion of μ **and σ** (σ rescales per axis like `u`) |
| IO loop | `experiments/error-map-gen/unigrad-io/create_unigrad_io_data.py:169–190` | σ-guided IO (E5); IO delta as a baseline |
| Masked metrics, orthogonal-slice figures, min / median / max case selection | `experiments/regression/unigrad-synth/eval_unigrad_synth_unet.py`, `experiments/error-map-gen/unigrad-synth/visualize_unigrad_data.py` | extend with sparsification, AUSE, nAURC, NLL, coverage |
| `UNet3D` error regressor, trained run `error_unet_run1` (r 0.874) | `experiments/regression/unigrad-synth/train_unigrad_synth_unet.py`, `assets/runs/regression/unigrad-synth/hcp/error_unet_run1/` | supervised error-prediction baseline; its 5-channel input is IO-CUE's layout without features |
| IXI IO track (‖φ_IO − φ_pred‖) | `experiments/error-map-gen/unigrad-io/`, `experiments/regression/unigrad-io/` | label-free proxy baseline; OOD brain set |
| Nautilus PVC, venv, GPU deployment | `deploy/nautilus/` | runs everything; the probe fits one GPU |

**What must change:**
- σ is produced at 175³ in normalised units on the fixed lattice, and must go through the same per-axis rescale and trilinear resampling as `u` before comparison with `u_gt` in native voxels. Variance scales by `(N−1)²` per axis.
- The synthetic bench's `u_gt` is in native voxels. Tian's protocol lives at 175³; both are reported.
- The head needs the uniGradICON network object (hook on `lastConv`), not only saved NPZ fields. Phase II has to be re-run with the head attached, or features saved.

---

## 9. Recommendation (proposal for the advisor)

**Claim to aim for:** a single forward pass of uniGradICON can output a calibrated per-voxel transformation uncertainty. It is learned from the model's own registration objective, with no ground-truth deformations, from the released weights by probing. It tracks true error better than test-time equivariance (Tian 2025) and inverse-consistency error, at 1/60 of the cost. It flags OOD failure and guides instance optimisation.

**Design**

| Choice | Recommendation | Why |
| --- | --- | --- |
| Output | 6 channels: frozen `u_ψ` (3) + per-axis log σ² (3), layout B | strict checkpoint load; μ quality preserved; 3 channels beat 1 (Hu 2025) |
| Head | small 3D U-Net on `[lastConv input (18), warped A, B, u_ψ (3)]`, zero-initialised output | IO-CUE's input argument; zero-init starts at the prior |
| Objective | LNCC (both directions) on a smoothed-noise sample + β·KL to a Laplacian prior on σ; GradICON on μ | label-free; avoids white-noise Jacobian blow-up; Dalca / PULPo template |
| Composed σ | Monte Carlo over re-composition (K = 20–50, no extra UNet passes); first-order formula as a check | the last step alone is not the output field |
| Calibration | one temperature (and a per-class conformal quantile) fitted on the synthetic Val split | fixes the arbitrary LNCC scale; the only use of `u_gt` |
| Training order | probe → LP-FT → full FT (comparison) | Kumar 2022; untested in registration, so it is a finding either way |
| Data | HCP (in corpus) for E0–E3; AbdomenCTCT, IXI, DIR-Lab, NLST for E4 | §7 |

**Experiment ladder**

| Step | What | Output |
| --- | --- | --- |
| E0 | Baselines on the HCP synthetic bench: inverse-consistency error (free), Tian 2025 re-implemented (N = 50, their four perturbation types), IO delta, our `UNet3D` error regressor | table of r / ρ / AUSE / nAURC per deformation class |
| E1 | Probe: σ head only, trained on real HCP inter-subject pairs with the §3.3 objective; β sweep; collapse monitoring | σ maps; calibration fitted on synthetic Val; E0 table extended |
| E2 | LP-FT from E1 | same table + Dice / TRE of μ (must not drop) |
| E3 | Full FT from the checkpoint | same, for the LP vs LP-FT vs FT comparison |
| E4 | OOD: IXI, L2R AbdomenCTCT, OASIS, NLST, DIR-Lab; LUMirage-style shifts (contrast, resolution) | sparsification vs landmark / label error; AUROC of image-level σ vs failure; does synthetic calibration transfer |
| E5 | σ-guided IO: gate / weight IO by σ | TRE / Dice gain vs plain IO, `%|J|<0` |

**Where the novelty is, tied to what each paper left open**

| Open point (source) | How this work answers it |
| --- | --- |
| Tian 2025 v1: framework "naturally extends to probabilistic registration models"; v2 limitation: "does not disentangle different sources of uncertainty"; dropped v1 limitation: blind to transformation-equivariant errors | a learned probabilistic output on the same model, compared on their protocol, in one pass |
| Tian 2025: r = 0.43 vs nonlinear ground truth on uniGradICON; the map is pure bias jitter for a deterministic net | σ is trained on residual evidence, not jitter; tested on nonlinear ground truth at scale |
| Hu et al. ICLR 2025: "extend the framework to pairwise registration of any two input images" | pairwise, any anatomy, foundation model, label-free |
| Dalca 2019: "Analysis of uncertainty is beyond the scope of this paper"; PULPo 2024: σ collapse | quantitative uncertainty analysis with collapse diagnostics and calibration |
| Luo 2020: are uncertainty and error monotonically associated? | tested per deformation class and region with known `u_gt` (1113 subjects) |
| IO-CUE 2025 limitation: requires an MSE-fit base model | a frozen base trained with LNCC + GradICON; measures what the probe absorbs |
| Kumar 2022 LP-FT: never tested for dense regression or registration | first probe / LP-FT / FT comparison for uncertainty on a registration foundation model |
| uniGradICON paper: no uncertainty; IO increases folding | σ-guided IO |
| LUMirage 2025: silent zero-shot failure under contrast / resolution shift | image-level σ as a failure flag |

Nobody else has published this as of 2026-09-26. The nearest risk is the Tian group itself (Tian is also uniGradICON's first author).

---

## 10. Open questions for the advisor

1. Is the claim **transformation variance calibrated against error**, or **expected error**? This decides whether the error-prediction head is the baseline or a co-primary.
2. Is a synthetic-only calibration step acceptable, given that `u_gt` is never used at inference?
3. Should the Tian 2025 baseline be re-implemented now (no code released), or should we wait for their code?
4. Compute budget on Nautilus for LP-FT and full FT (batch 1 on 40–80 GB GPUs, 4 UNets × 2 directions).
5. Dataset access to request now: DIR-Lab COPDgene (form), and whether OAI via NDA is worth the paperwork.
6. Target venue and timing, which decides how far down the E0–E5 ladder the first paper goes.
7. Is it acceptable to modify `icon_registration` / uniGradICON in a fork, or must everything stay as hooks and wrappers around the released package?

---

## References (opened 2026-09-26, except Gal & Ghahramani 2016, added from its well-known arXiv id)

| Short name | Citation | Link |
| --- | --- | --- |
| uniGradICON | Tian et al., MICCAI 2024 | [arXiv 2403.05780](https://arxiv.org/abs/2403.05780) |
| GradICON | Tian et al., CVPR 2023 | [arXiv 2206.05897](https://arxiv.org/abs/2206.05897) |
| ICON | Greer et al., ICCV 2021 | [arXiv 2105.04459](https://arxiv.org/abs/2105.04459) |
| multiGradICON | Demir et al., WBIR 2024 | [arXiv 2408.00221](https://arxiv.org/abs/2408.00221) |
| Tian 2025 | Tian, Hu, Iglesias, under review (ICLR 2026 submission) | [arXiv 2509.23355](https://arxiv.org/abs/2509.23355), [OpenReview](https://openreview.net/forum?id=4dARndGskg) |
| Hu 2025 | Hu et al., ICLR 2025 | [arXiv 2410.09299](https://arxiv.org/abs/2410.09299) |
| Chaudhary 2024 | Chaudhary et al., arXiv 2024 | [arXiv 2411.07567](https://arxiv.org/abs/2411.07567) |
| SAM-uniGradICON | Hu, Yu et al., IJCNN 2025 | [arXiv 2505.06527](https://arxiv.org/abs/2505.06527) |
| BrainMorph | Wang et al., MELBA 2025 | [arXiv 2405.14019](https://arxiv.org/abs/2405.14019) |
| SAME++ | Tian, Li et al., arXiv 2023 | [arXiv 2311.14986](https://arxiv.org/abs/2311.14986) |
| Eppenhof 2018 | Eppenhof & Pluim, J. Med. Imaging 5(2) | [DOI 10.1117/1.JMI.5.2.024003](https://doi.org/10.1117/1.JMI.5.2.024003) |
| Sokooti 2019 | Sokooti et al., MedIA 2019 | [arXiv 1905.07624](https://arxiv.org/abs/1905.07624) |
| Sokooti 2021 | Sokooti, Yousefi et al., IEEE Access 9 | [IEEE 9408621](https://ieeexplore.ieee.org/document/9408621/) |
| Bierbrier 2022 | Bierbrier, Gueziri, Collins, MedIA 81:102531 | DOI 10.1016/j.media.2022.102531 (inferred) |
| RegQCNET | de Senneville, Manjón, Coupé, PMB 2020 | [arXiv 2005.06835](https://arxiv.org/abs/2005.06835) |
| FocalErrorNet | Salari et al., MICCAI 2023 | [arXiv 2307.14520](https://arxiv.org/abs/2307.14520) |
| Salari 2023 | Salari et al., IEEE IUS 2023 | [arXiv 2308.10784](https://arxiv.org/abs/2308.10784) |
| Chen 2024 | Chen et al., ISBI 2024 | [arXiv 2403.05111](https://arxiv.org/abs/2403.05111) |
| Loi 2026 | Loi et al., Phys. Imaging Radiat. Oncol. 37:100916 | [PMC12925039](https://pmc.ncbi.nlm.nih.gov/articles/PMC12925039/) |
| IO-CUE | Bramlage & Curio, arXiv 2025 | [arXiv 2506.00918](https://arxiv.org/abs/2506.00918) |
| CONReg | Gheiji et al., J. Imaging Inform. Med. 2026 | [DOI 10.1007/s10278-026-01878-3](https://doi.org/10.1007/s10278-026-01878-3) |
| AC-CAR | Wang, Luo, Du, Qin, IEEE TMI 2026 | [arXiv 2601.05981](https://arxiv.org/abs/2601.05981) |
| Zhang 2024 | Zhang et al., MICCAI 2024 | [arXiv 2312.00836](https://arxiv.org/abs/2312.00836) |
| Rivetti 2024 | Rivetti et al., PMB 69(11):115045 | [DOI 10.1088/1361-6560/ad4c4f](https://doi.org/10.1088/1361-6560/ad4c4f) |
| PULPo | Siegert et al., MICCAI 2024 | [arXiv 2407.10567](https://arxiv.org/abs/2407.10567) |
| TransMorph | Chen et al., MedIA 2022 | [arXiv 2111.10480](https://arxiv.org/abs/2111.10480) |
| Gong 2022 | Gong et al., WACV 2022 | [CVF](https://openaccess.thecvf.com/content/WACV2022/papers/Gong_Uncertainty_Learning_Towards_Unsupervised_Deformable_Medical_Image_Registration_WACV_2022_paper.pdf) |
| Dalca 2019 | Dalca et al., MedIA 2019 | [arXiv 1903.03545](https://arxiv.org/abs/1903.03545) |
| Krebs 2019 | Krebs et al., IEEE TMI 2019 | [arXiv 1812.07460](https://arxiv.org/abs/1812.07460) |
| Kendall & Gal 2017 | NeurIPS 2017 | [arXiv 1703.04977](https://arxiv.org/abs/1703.04977) |
| Gal & Ghahramani 2016 | ICML 2016 | [arXiv 1506.02142](https://arxiv.org/abs/1506.02142) |
| β-NLL | Seitzer et al., ICLR 2022 | [arXiv 2203.09168](https://arxiv.org/abs/2203.09168) |
| Evidential | Amini et al., NeurIPS 2020 | [arXiv 1910.02600](https://arxiv.org/abs/1910.02600) |
| DeVries 2018 | DeVries & Taylor, arXiv 2018 | [arXiv 1802.04865](https://arxiv.org/abs/1802.04865) |
| Learning loss | Yoo & Kweon, CVPR 2019 | [arXiv 1905.03677](https://arxiv.org/abs/1905.03677) |
| BINDER | Cerri et al., arXiv 2026 | [arXiv 2609.19875](https://arxiv.org/abs/2609.19875) |
| Luo 2019 | Luo et al., MICCAI 2019 | [arXiv 1803.05266](https://arxiv.org/abs/1803.05266) |
| Luo 2020 | Luo et al., MICCAI 2020 | [arXiv 1908.07709](https://arxiv.org/abs/1908.07709) |
| RUGI | González et al., arXiv 2026 | [arXiv 2609.28081](https://arxiv.org/abs/2609.28081) |
| LP-FT | Kumar et al., ICLR 2022 | [arXiv 2202.10054](https://arxiv.org/abs/2202.10054) |
| Frozen-backbone IO | Zhang, Zhao, Tao, Learn2Reg 2025 | [arXiv 2603.26393](https://arxiv.org/abs/2603.26393) |
| Gradient-projection IO | Zhang, Zhao, Tao, Learn2Reg 2024 | [arXiv 2410.15767](https://arxiv.org/abs/2410.15767) |
| LUMirage | Jena, Chaudhari, Gee, arXiv 2025 | [arXiv 2512.15505](https://arxiv.org/abs/2512.15505) |
| Beyond LUMIR | Chen et al., MedIA (accepted) | [arXiv 2505.24160](https://arxiv.org/abs/2505.24160) |
| Surrogate supervision | Liu et al., arXiv 2025 | [arXiv 2509.09869](https://arxiv.org/abs/2509.09869) |
