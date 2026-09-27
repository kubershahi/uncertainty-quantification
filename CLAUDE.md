# CLAUDE.md

Guidance for Claude Code in this repo.

## What this is

Groundwork for **uncertainty quantification (UQ) for medical image registration**: learn whether dense registration error is predictable from inference-time observables, so later UQ methods can be calibrated against real error rather than model self-consistency. The foundation registrator is [UniGradICON](https://github.com/uncbiag/uniGradICON). Owner: Kuber Shahi (UCSD).

**Read `README.md` first**, then the doc for the track you touch (`docs/`). Write-up: `reports/Uncertainty_Quantification.pdf`; LaTeX source `reports/latex/neurips_2026.tex` is **local and gitignored**. `commands.sh` (kubectl / tmux / latex cheat-sheet) is local and gitignored too.

## Status (2026-09-27)

Both tracks share one three-phase pattern: **I** dense reference displacement → **II** UniGradICON prediction + error map vs the reference → **III** 3D U-Net regresses the error map from observable inputs.

1. ✅ **Primary — HCP synth (`unigrad-synth`).** HCP S1200 T1w (native, ~0.7 mm). Phase I: TorchIO warps (`none` 5 % · `rigid` 20 % · `affine` 25 % · `elastic` 25 % · `affine_elastic` 25 %), identity-grid warp → SimpleITK `InvertDisplacementField` → `u_gt`; subject-level split **75/10/15**. Phase II: `net(moving, source)` at 175³ → `u_pred` resampled to native; `u_error_map = ‖u_gt − u_pred‖`. Phase III: `UNet3D` 5→1, GroupNorm, `base_channels=16`, inputs `[source, moving, u_pred/64]`, masked MAE in `source_mask`, AdamW + `ReduceLROnPlateau`, AMP + `torch.compile` on by default. **`error_unet_run1` Test (n = 147): MAE 0.338 · RMSE 0.485 · Pearson r 0.874 (voxels)**; best val MAE 0.318 at epoch 69. Report written.
2. ✅ **Secondary — IXI instance optimization (`unigrad-io`).** Atlas–subject pairs (`datasets/IXI/{Train,Val,Test}/*.pkl` + `atlas.pkl`). Zero-shot `phi_pred`, then 50-step IO → `phi_predio`; target `‖phi_predio − phi_pred‖` in atlas `valid_mask`. `UNet3D` 5→1, BatchNorm, `base_channels=32`, inputs `[subject, atlas, phi_pred/64]`, no pad-to-16. Runs (Test, n = 115, MSE / L1): run1 smoke MSE+TV 0.05 → 0.691 / 0.566; run2 MSE+TV 0.02 → 0.511 / 0.478; run3 TV 0 → 0.528 / 0.474 (TV has no measurable effect); **run4 L1 + early stop → 0.563 / 0.479**.
3. ▶ **Next — uncertainty-aware uniGradICON (`unigrad-ua/`).** Make uniGradICON output a per-voxel transformation uncertainty in one pass: a head on the last UNet's features adds 3 per-axis log-variances to the frozen 3-channel displacement (6 channels), trained label-free from the released weights with the registration loss on a smoothed-noise sample plus a KL prior, by probe → LP-FT → full FT. `u_gt` is never available for real pairs, so the HCP synth set (Phase I/II) becomes the **calibration and evaluation bench**, and the Phase III regressor becomes a baseline, alongside inverse-consistency error and the test-time equivariance method of Tian, Hu & Iglesias 2025 (arXiv 2509.23355). No trained uncertainty head on any registration foundation model was found (search 2026-09-26). Pretraining facts: `unigrad-ua/UNIGRADICON.md`; literature, design, novelty and experiment ladder E0–E5: `unigrad-ua/RECON.md`. Documents only so far; no code or runs.

## How to work

- **Small committable batches**, one concern per commit. **Kuber commits himself — Claude never runs `git commit` (or add/reset/push), even when told "then we commit".** Hand over `git add … && git commit -m "<short subject>" -m "<succinct body>"` with a plain-language explanation. Succinct everywhere.
- **Cluster commands are documented, not run by Claude.** Kubernetes / GPU work goes on NRP Nautilus. Give Kuber the command and the expected output; he pastes the result back. Edit here → commit → `git pull` on the PVC. Never edit code on the cluster.
- **Never delete files** (cleanup, `.DS_Store`, regenerable-looking figures, untracked `assets/`) without first listing exact paths and getting approval. Don't edit `README.md` or `docs/` unless asked.
- Prefer editing the one script over parallel scripts. Touch the Test split once. Spell out acronyms on first use in prose. Don't start data generation, downloads or training unless asked.

## Delegation

- Delegate multi-file investigations, broad searches, independent parallel work and large edits to subagents; do small edits and answers that depend on this conversation directly.
- One subagent per task; plan first only when the task is complicated; run independent subagents in parallel.
- Give each subagent the context it lacks (the rules above, the track names, the displacement convention) and ask for a short report.
- Work from the report; open the files only to verify a claim before it goes into a commit, a doc or the report.

## Model routing (pass `model` on every agent call)

- fable: architecture, hard bugs, reviews
- opus: edits, tests, docs, refactors
- sonnet: lookups and summaries

## Conventions (code and file style)

- **Tracks and names.** `synth-data-gen/torchio` (Phase I), `error-map-gen/{unigrad-synth,unigrad-io}` (II), `regression/{unigrad-synth,unigrad-io}` (III). Scripts are `create_*` / `visualize_*` / `train_*_unet` / `eval_*_unet` / `sweep_*`. Directories and labels are lowercase and `-`-joined; Python identifiers, NPZ keys and CSV columns use `_`. Documents under `docs/` are lowercase `-`-joined (`hcp-dataset.md`). Documents inside a track directory such as `unigrad-ua/` are `UPPERCASE.md` (`UNIGRADICON.md`, `RECON.md`); add a `README.md` index there once it holds three documents. The root files are `README.md` and `CLAUDE.md`.
- **Paths** (all relative to the repo root; argparse defaults match these):

  | Role | Path |
  | --- | --- |
  | HCP raw | `datasets/hcp/` |
  | HCP synth NPZ (I) | `datasets/synth-data/torchio/hcp/{Train,Val,Test}/` |
  | HCP error-map NPZ (II) | `datasets/error-map/unigrad-synth/hcp/` |
  | IXI volumes | `datasets/IXI/` + `atlas.pkl` |
  | IXI IO NPZ (II) | `datasets/error-map/unigrad-io/ixi/` |
  | QC figures | `assets/images/{synth-data,error-map}/…` |
  | Trained runs | `assets/runs/regression/unigrad-synth/hcp/error_unet_run<N>/`, `assets/runs/regression/unigrad-io/error_unet_run<N>/` |

  `datasets/` and `models/` are gitignored (only `.gitkeep` is tracked). `best_model.pt` and `wandb/` are gitignored. Runs keep `run_config.json`, `metrics.csv`, `test_metrics.json` and the QC PNGs in git.
- **Run names** `error_unet_run<N>` for the directory; W&B `--wandb-run-name unigradsynth_unet_run<N>`. Never reuse a run directory for different weights; start a new `run<N+1>`.
- **Units and conventions.** Displacements and error maps are in **voxels** (index space). HCP convention: `moving(x + u(x)) ≈ source(x)` on the source/fixed lattice. Vocabulary (φ vs u): `docs/registration-concepts.md`. Loss and metrics are always masked (`source_mask` for HCP, atlas `valid_mask` for IXI). Put units on colorbar labels, not repeated in plot titles.
- **Python.** Black / isort / ruff at line length 100, py39 target, 4-space indent, LF. Each script starts with `#!/usr/bin/env python3`, then a module docstring: purpose, NPZ input/output key schema, model I/O, then `Example:` lines. **Example commands are single-line and copy-pasteable** (no `\` continuations), run from the repo root, with values matching the file's argparse defaults. Also: `from __future__ import annotations`, PEP 604 type hints on signatures, `pathlib.Path`, f-strings, module-level `DEFAULT_*` constants, `parse_args()` + `main(argv)`, `argparse.BooleanOptionalAction` for on-by-default switches, no bare `except`, `set_seed()` covering `random/numpy/torch/cuda`. AMP uses `torch.amp.autocast("cuda")` / `torch.amp.GradScaler("cuda")`. Reuse the 3D IO helpers in `create_unigrad_io_data.py`; keep sweep-only plotting in `sweep_io_iterations.py`.
- **Shell / Kubernetes.** Start scripts with `set -euo pipefail`. The header gives a one-line purpose and an example invocation. Use `# shellcheck source=/dev/null` before a `source`. Knobs are UPPERCASE environment variables (`FORCE=1`, `SUBJECT_LIST_FILE`, `PARALLEL_JOBS`). `.sh` files and `deploy/nautilus/**/*.yaml` must stay LF (`.gitattributes`). Never commit kubeconfig or credentials. Local overrides go in `deploy/nautilus/local/` (gitignored).
- **Documents.** Use tables over prose for numbers. Every command block runs from the repo root. Report masked metrics with n volumes and split. No personal names in tracked docs other than this file.

## Environments

- **Laptop:** conda env **`biag`** (`/opt/miniconda3/envs/biag`, python 3.11, torch 2.10, torchio 0.22.1). **The shell default is `grad`**, so run `conda activate biag` or call `/opt/miniconda3/envs/biag/bin/python`. UniGradICON is an **editable install from the sibling clone** `~/Documents/GitHub/uniGradICON` (`3e83a92`). Read `src/unigradicon/` there when the API misbehaves; the pretraining script is `training/train.py` (+ `dataset.py`), and `icon_registration` 1.1.7 lives in the env's `site-packages`. Apple Silicon with MPS: the scripts select CUDA or CPU only (`create_unigrad_io_data.py` hard-codes CUDA), so treat the laptop as smoke-test only.
- **NRP Nautilus (Kubernetes):** namespace comes from the kubectl context (manifests set none). PVC **`unc-files`** (RWX, rook-cephfs, 2 Ti) is mounted at **`/files`**. Repo at `/files/repo/uncertainty-quantification`, venv at `/files/venvs/unc` (built by `deploy/nautilus/scripts/setup_venv.sh`, which installs torch cu124 and then `pip install unigradicon torchio wandb …` from PyPI, not editable). `scripts/env.sh` exports the path roots, moves `$HOME` to `/files/home/root` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Workloads: `unc-dev` (CPU PVC admin pod), `unc-jupyter` (Jupyter Lab), `unc-heavy` (1-GPU deployment), `job-create-unigrad-io-data` (IXI Phase II). Long runs go in tmux. GPU memory tips: `docs/gpu-memory-optimizations.md`.
- **HCP download:** `scripts/download_hcp.sh` runs `aws s3 cp` from `s3://hcp-openaccess/HCP_1200` (T1w brain, `aparc+aseg`, `brainmask_fs`) and needs ConnectomeDB AWS credentials in `~/.aws`. Pass `SUBJECT_LIST_FILE=scripts/hcp_subjects_test10.txt` for a 10-subject test.

## Verified facts and known stale spots

- The README's train and eval commands for HCP are exactly the scripts' defaults plus `--wandb`, `--mode both` and `--no-show`. `training_curves.png` is written by **eval**, not train. Eval `--mode {figures,metrics,both}` writes `test_metrics.json`, `test_random_orthogonal/` and `test_mmm_orthogonal/`.
- The HCP U-Net inputs `u_pred` unmasked by default; `--mask-u-pred` zeroes it outside `source_mask`, for ablations. Phase II also writes `error_map_mask`, but nothing downstream reads it.
- `UNet3D` / `DoubleConv3d` are copy-pasted between the two `train_*_unet.py` files. They differ in the norm layer and the HCP pad-to-16/crop. Keep the two copies in sync, or factor them out on purpose in one commit.
- The IXI `run_config.json` files carry old absolute paths from the cluster (`/files/repo/…/IXI_unigrad_io`, `assets/runs/unigrad-io/…`). They record what was run, so leave them as they are.
- The IXI tree has no `3d/` level (it was dropped from paths and defaults on 2026-09-26). The env var `UNIGRAD_IO_3D_OUT` in `env.sh` keeps its old name.
- `scripts/unigradicon_registration_demo.py` is a notebook export (`!pip` magics) and does not run as plain Python.
- uniGradICON: `phi_AB_vectorfield` is φ_AB on the **fixed (B) lattice** in normalised [0, 1] coordinates (spacing 1/174 at 175³); `u = φ − identity_map`, rescaled to native voxels by `(N−1)` per axis. 70,680,916 parameters (4 × `tallUNet2`), 20 BatchNorm3d, **no dropout**; `lastConv` 18→3, zero-init, output ÷10.
- `get_unigradicon()` downloads weights to `network_weights/unigradicon1.0/` **relative to the cwd** and strict-loads them into `net.regis_net`; keep any new head outside `regis_net`.
- `icon_registration`'s `finetune_execute` (instance optimisation) **does not restore the weights** afterwards; deep-copy or reload the model between pairs.
- `icon_registration.config.device` is CUDA or CPU only and the loss hard-codes it: no MPS.
- Uncertainty-head attachment point: forward hook on `net.regis_net.netPsi.net.lastConv`; its input is the 18-channel full-resolution feature map of the last (step-2) UNet, which predicts only the residual on top of the step-1 field.

## Layout

```
README.md, CLAUDE.md, LICENSE, pyproject.toml, requirements.txt, .editorconfig, .gitattributes
experiments/synth-data-gen/torchio/          Phase I HCP synth + HCP/synth QC viz
experiments/error-map-gen/unigrad-synth/     Phase II HCP UniGradICON error maps + viz
experiments/error-map-gen/unigrad-io/        Phase II IXI zero-shot + IO error maps, IO-iteration sweep, atlas viz
experiments/regression/unigrad-{synth,io}/   Phase III train / eval 3D U-Net
docs/                                        concepts, HCP dataset, per-track experiment notes, GPU memory, viz guide
assets/images/  assets/runs/regression/      QC figures · trained-run artefacts (no checkpoints)
scripts/                                     HCP download + subject list, registration demo / viz helpers
deploy/nautilus/                             PVC, pods, deployments, IO data job, venv / env scripts
unigrad-ua/                                  uncertainty-aware uniGradICON: UNIGRADICON.md (pretraining brief), RECON.md (literature + design)
reports/                                     Uncertainty_Quantification.pdf, uniGradICON.pdf (paper)
datasets/ models/ reports/latex/ commands.sh .cursor/   local, gitignored
```
