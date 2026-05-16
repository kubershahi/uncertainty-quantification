# Nautilus (NRP Kubernetes)

Runbook for **PVC-mounted GPU workloads**: repo + venv + datasets under `/files`, manifests without hard-coded namespaces (use local kube context).

**Related repo choices:** pipeline code lives in **`experiments/`** (formerly `datahub/`). Manifests prefer **Ampere/Ada** GPUs via `nvidia.com/gpu.product`; **V100 is excluded** — widen the list in YAML if jobs stay Pending.

---

## `/files` layout

```text
/files/
  repo/uncertainty-quantification/
  venvs/unc/
  datasets/IXI_2D/          # 2D slices (Train/Val/Test/Atlas)
  datasets/IXI/             # optional raw *.pkl volumes for 3D sweep
  outputs/IXI_2D_unigrad_io/
  runs/
```

---

## Prerequisites

- `kubectl` + kubeconfig from the portal (keep local; do not commit).
- Namespace with PVC + GPU quota.

```bash
kubectl config set-context --current --namespace=<your-namespace>
```

---

## 1. PVC (once)

```bash
kubectl apply -f deploy/nautilus/pvc.yaml
kubectl get pvc unc-files
```

Adjust `storageClassName` in `pvc.yaml` if `rook-cephfs` is wrong (`kubectl get storageclass`).

---

## 2. Pods

| Goal | Manifest | Shell |
|------|-----------|--------|
| Interactive dev (limits ~16 CPU / ~32 Gi incl. shm) | `pod-dev.yaml` | `kubectl exec -it unc-dev -- bash` |
| More CPU/RAM (e.g. long IO sweep) | `deployment-heavy.yaml` | `kubectl exec -it deployment/unc-heavy -- bash` |
| Batch IO NPZ generation | `job-unigrad-io-data.yaml` | `kubectl logs -f job/unc-unigrad-io-data` |

```bash
kubectl apply -f deploy/nautilus/pod-dev.yaml
kubectl wait --for=condition=Ready pod/unc-dev --timeout=20m
```

Heavy interactive pod:

```bash
kubectl apply -f deploy/nautilus/deployment-heavy.yaml
kubectl rollout status deployment/unc-heavy --timeout=25m
kubectl exec -it deployment/unc-heavy -- bash
```

Delete deployment when idle: `kubectl delete deployment unc-heavy`.

---

## 3. One-time setup in the container

Image may lack `git`:

```bash
apt-get update && apt-get install -y git   # root shell typical
```

Clone onto PVC:

```bash
mkdir -p /files/repo && cd /files/repo
git clone https://github.com/<org>/uncertainty-quantification.git
cd uncertainty-quantification
bash deploy/nautilus/scripts/setup-venv.sh
source /files/venvs/unc/bin/activate
python experiments/resource_checks/diagnose_torch_gpu.py
```

**Zip instead of git:** unpack GitHub `main.zip` under `/files/repo/` (may need `apt-get install -y unzip curl`).

---

## 4. Copy data to PVC

From laptop — **create parent dirs first** (`kubectl cp` unpack needs them):

```bash
kubectl exec unc-dev -- mkdir -p /files/datasets
kubectl cp ixi_2d.tar.gz unc-dev:/files/datasets/ixi_2d.tar.gz
kubectl exec unc-dev -- bash -c \
  'cd /files/datasets && tar xzf ixi_2d.tar.gz && rm -f ixi_2d.tar.gz'
```

---

## 5. Smoke test

```bash
source /files/venvs/unc/bin/activate
cd /files/repo/uncertainty-quantification
bash deploy/nautilus/scripts/run-io-data-smoke.sh
```

Paths come from `deploy/nautilus/scripts/env.sh` (`IXI_ROOT`, `UNIGRAD_IO_OUT`, …).

---

## 6. Full IO generation

**Interactive (tmux):**

```bash
tmux new -s io
source /files/venvs/unc/bin/activate
cd /files/repo/uncertainty-quantification
python experiments/unigrad-io/create_unigrad_io_data.py \
  --ixi-root /files/datasets/IXI_2D \
  --output-path /files/outputs/IXI_2D_unigrad_io
```

**Job:**

```bash
kubectl apply -f deploy/nautilus/job-unigrad-io-data.yaml
kubectl logs -f job/unc-unigrad-io-data
```

Runs are **resumable** (existing `.npz` skipped unless `--overwrite`).

---

## 7. Sync laptop → cluster

From repo root on laptop after committing:

```bash
git push
kubectl exec unc-dev -- bash -lc 'cd /files/repo/uncertainty-quantification && git pull --ff-only'
```

(Uncommitted edits: stream a tarball or use `kubectl cp` for specific files.)

---

## 8. Cleanup

```bash
kubectl delete pod unc-dev
kubectl delete deployment unc-heavy
kubectl delete job unc-unigrad-io-data
```

PVC persists unless deleted separately.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| Bare Pod admission (`…limited to 16 cores and 32 GB`) | Use `deployment-heavy.yaml` or `job-unigrad-io-data.yaml`, or shrink `pod-dev.yaml` resources + shm. |
| Pod **Pending** (GPU) | Check node labels: `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\t}{.metadata.labels.nvidia\.com/gpu\.product}{\n}{end}'` — add matching strings to YAML `values:`. |
| `kubectl cp` / tar errors | Ensure destination dirs exist (`mkdir -p /files/datasets`). |
| `git` missing in container | `apt-get install -y git`. |
| `set: pipefail` / script errors | Shell scripts must use **LF** line endings (see `.gitattributes`). |

---

## Quick reference

| Task | Command |
|------|---------|
| Activate venv | `source /files/venvs/unc/bin/activate` |
| Env exports | `source deploy/nautilus/scripts/env.sh` |
| IO sweep (2D) | `python experiments/unigrad-io/sweep_io_iterations.py --ixi-root /files/datasets/IXI_2D …` |
| IO sweep (3D pickles) | `--mode 3d-pkl --ixi-root /files/datasets/IXI --atlas-pkl /files/datasets/IXI/atlas.pkl …` |
| Train U-Net | `python experiments/train_error_map_unet.py` |

---

## Vertex vs Nautilus

| Vertex | Nautilus |
|--------|----------|
| Custom image per revision | Stock `pytorch/pytorch` |
| Deps baked in image | Venv on PVC |
| Per revision image | New Pod/Job, same `/files` |
