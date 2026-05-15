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

## 3. Inside the pod — one-time setup

```bash
# Clone repo (HTTPS or copy SSH key into pod first)
mkdir -p /files/repo
cd /files/repo
git clone https://github.com/<your-org>/uncertainty-quantification.git
cd uncertainty-quantification

# Venv on PVC (~15–20 min, only once)
bash deploy/nautilus/scripts/setup-venv.sh

# GPU check
source /files/venvs/unc/bin/activate
python datahub/resource_checks/diagnose_torch_gpu.py
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
python datahub/unigrad-io/create_unigrad_io_data.py \
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

## Quick reference

| Task | Command |
|------|---------|
| Activate env | `source /files/venvs/unc/bin/activate` |
| Env vars | `source deploy/nautilus/scripts/env.sh` |
| IO sweep | `python datahub/unigrad-io/sweep_io_iterations.py --ixi-root /files/datasets/IXI_2D ...` |
| Train U-Net | `python datahub/train_error_map_unet.py` (point `--data-dir` at outputs path) |

## Vertex vs Nautilus

| Vertex | Nautilus (this setup) |
|--------|------------------------|
| Custom Docker image per revision | Stock `pytorch/pytorch` image |
| Deps in image | Deps in `/files/venvs/unc` on PVC |
| New image per experiment | New pod/job, same image + same `/files` |
