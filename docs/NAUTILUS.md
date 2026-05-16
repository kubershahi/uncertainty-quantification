# Nautilus (Kubernetes) deployment

## Layout on `/files`

```text
/files/
  repo/uncertainty-quantification/   # git clone
  venvs/unc/                          # pip env (setup once)
  datasets/IXI_2D/                    # Train/ Val/ Test/ Atlas/
  outputs/IXI_2D_unigrad_io/            # create_unigrad_io_data.py
  runs/                               # U-Net checkpoints (later)
```

## Prerequisites

- `kubectl` configured for Nautilus (download kubeconfig from the portal; keep it **local only**).
- Access to a namespace where you can create PVCs and GPU pods.

```bash
# Replace with your namespace (do not commit this value to a public repo)
export KUBE_NS=your-namespace-here
kubectl config set-context --current --namespace="${KUBE_NS}"
```

## 1. Claim storage (once)

```bash
kubectl apply -f deploy/nautilus/pvc.yaml
kubectl get pvc unc-files
```

If `rook-cephfs` is wrong for your namespace, check `kubectl get storageclass` and edit
`pvc.yaml`.

## 2. Start dev pod

```bash
kubectl apply -f deploy/nautilus/pod-dev.yaml
kubectl wait --for=condition=Ready pod/unc-dev --timeout=20m
kubectl exec -it unc-dev -- bash
```

If the pod stays **Pending**, loosen GPU names in `pod-dev.yaml` or remove the
`nodeAffinity` block temporarily.

### More CPU/RAM for long jobs (e.g. `sweep_io_iterations.py`)

Bare Pods (`pod-dev.yaml`) hit NRP limits (~16 CPU, ~32 Gi including shm). For heavier
interactive runs, apply **`deployment-heavy.yaml`**: it uses the same PVC but requests **up to
16 CPU / 64 Gi** RAM (matches the batch Job manifest pattern).

```bash
kubectl apply -f deploy/nautilus/deployment-heavy.yaml
kubectl rollout status deployment/unc-heavy --timeout=25m
kubectl exec -it deployment/unc-heavy -- bash
```

Stop it when idle: `kubectl delete deployment unc-heavy` (PVC is unchanged).

## 3. Inside the pod — one-time setup

The dev image **`pytorch/pytorch:…-devel`** usually does **not** ship with `git`. Install it once in the pod (Debian/apt), then clone onto the PVC so the repo survives pod restarts.

```bash
# Install git (~30s; harmless if apt already stale)
sudo apt-get update && sudo apt-get install -y git
# dev container often runs as root — if sudo is missing and you are root, omit sudo:
#   apt-get update && apt-get install -y git

# Clone repo (HTTPS; use SSH URL + keys only if you prefer)
ORG=your-org   # or USER for a personal fork
mkdir -p /files/repo
cd /files/repo
git clone "https://github.com/${ORG}/uncertainty-quantification.git"
cd uncertainty-quantification

# Venv on PVC (~15–20 min, only once)
bash deploy/nautilus/scripts/setup-venv.sh

# GPU check
source /files/venvs/unc/bin/activate
python experiments/resource_checks/diagnose_torch_gpu.py
```

**No apt / git install blocked?** Unpack from GitHub’s zip on the PVC (no git needed):

```bash
mkdir -p /files/repo && cd /files/repo
curl -fsSL -o repo.zip https://github.com/your-org/uncertainty-quantification/archive/refs/heads/main.zip
unzip -q repo.zip && mv uncertainty-quantification-main uncertainty-quantification
cd uncertainty-quantification   # unzip may need apt install unzip curl
```

## 4. Transfer IXI_2D onto PVC

From your **laptop** (while `unc-dev` is running):

```bash
# On machine that has IXI_2D/
tar czf ixi_2d.tar.gz -C /path/to/parent IXI_2D
kubectl cp ixi_2d.tar.gz unc-dev:/files/datasets/ixi_2d.tar.gz
kubectl exec unc-dev -- bash -c \
  'mkdir -p /files/datasets && cd /files/datasets && tar xzf ixi_2d.tar.gz && \
   rm -f ixi_2d.tar.gz && ls IXI_2D/Train | wc -l'
```

Expected layout: `/files/datasets/IXI_2D/{Train,Val,Test,Atlas}/`.

## 5. Smoke test (inside pod)

```bash
source /files/venvs/unc/bin/activate
cd /files/repo/uncertainty-quantification
bash deploy/nautilus/scripts/run-io-data-smoke.sh
```

## 6. Full data generation

**Option A — tmux in dev pod** (easy to watch logs):

```bash
tmux new -s io
source /files/venvs/unc/bin/activate
cd /files/repo/uncertainty-quantification
python experiments/unigrad-io/create_unigrad_io_data.py \
  --ixi-root /files/datasets/IXI_2D \
  --output-path /files/outputs/IXI_2D_unigrad_io
# Ctrl-b d to detach; kubectl exec back in later
```

**Option B — Kubernetes Job** (keeps running if you disconnect):

```bash
# From laptop
kubectl apply -f deploy/nautilus/job-unigrad-io-data.yaml
kubectl logs -f job/unc-unigrad-io-data
```

The script skips slices whose `.npz` already exists (resumable). Use
`--overwrite` only when regenerating.

## 7. Cleanup

```bash
kubectl delete pod unc-dev          # dev pod only; PVC kept
kubectl delete job unc-unigrad-io-data
```

## Troubleshooting

### `PODs without controllers are limited to 16 cores and 32 GB of RAM`

NRP blocks **bare Pods** (created with `kind: Pod` and no parent) above **16 CPU** and **32 Gi RAM** (including memory-backed volumes like `/dev/shm`).

- **Controller** = a Kubernetes object that owns Pods: `Deployment`, `StatefulSet`, `Job`, `CronJob`, etc. Your `unc-dev` manifest is a standalone Pod, so the stricter cap applies.
- **Fix (dev pod):** `pod-dev.yaml` is sized under the cap (8 CPU, 24 Gi container RAM + 4 Gi shm). Re-apply after pulling the latest manifest.
- **Need more RAM for a long run?** Use `job-unigrad-io-data.yaml` (`kind: Job`) or a `Deployment` with 1 replica instead of a bare Pod.

### `git: command not found` inside `unc-dev`

The PyTorch CUDA image drops `git` to stay small. Fix: `sudo apt-get update && sudo apt-get install -y git` inside the pod, then clone. Or fetch the repo ZIP with `curl`/`wget` over HTTPS (see step 3). **PVC note:** reinstalling git is only needed inside a fresh container — your clone under `/files/repo/` persists.

### `kubectl cp` tarball instead

You can tar the repo locally and skip git on the cluster:

```bash
# laptop: from repo root, excluding huge dirs
tar czf uq-repo.tar.gz --exclude=.git --exclude=data --exclude=venv .
kubectl cp uq-repo.tar.gz unc-dev:/files/repo/uq-repo.tar.gz
# pod:
mkdir -p /files/repo/uncertainty-quantification && cd /files/repo/uncertainty-quantification && tar xzf ../uq-repo.tar.gz
```

## Quick reference

| Task | Command |
|------|---------|
| Activate env | `source /files/venvs/unc/bin/activate` |
| Env vars | `source deploy/nautilus/scripts/env.sh` |
| IO sweep | `python experiments/unigrad-io/sweep_io_iterations.py --ixi-root /files/datasets/IXI_2D ...` |
| Train U-Net | `python experiments/train_error_map_unet.py` (point `--data-dir` at outputs path) |

## Vertex vs Nautilus

| Vertex | Nautilus (this setup) |
|--------|------------------------|
| Custom Docker image per revision | Stock `pytorch/pytorch` image |
| Deps in image | Deps in `/files/venvs/unc` on PVC |
| New image per experiment | New pod/job, same image + same `/files` |
