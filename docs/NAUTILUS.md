# Nautilus (NRP Kubernetes)

Runbook for this repo on Nautilus: **PVC at `/files`**, git clone, shared venv, **3D UniGrad IO** data under `datasets/`, figures under `assets/`. Set your namespace once:

```bash
kubectl config set-context --current --namespace=<your-namespace>
```

---

## `/files` layout

```text
/files/
  home/root/                         # persistent HOME (git config, ~/.ssh)
  repo/uncertainty-quantification/   # git clone
    datasets/IXI/                    # Train|Val|Test/*.pkl + atlas.pkl
    datasets/IXI_unigrad_io/         # generated 3D .npz (gitignored)
    assets/images/unigrad-io/        # IO sweep PNGs/CSVs (committed)
  venvs/unc/                         # PyTorch + UniGradICON + JupyterLab
```

---

## Manifests & scripts

| File | K8s name | Use |
|------|----------|-----|
| `deploy/nautilus/pvc.yaml` | `unc-files` | RWX volume (once per namespace) |
| `deploy/nautilus/pod-pvc-admin.yaml` | Pod `unc-dev` | PVC admin: git, SSH, `kubectl cp` (no GPU; same `pytorch/pytorch` image as other pods) |
| `deploy/nautilus/deployment-gpu.yaml` | Deployment `unc-heavy` | GPU shell: sweeps, `create_unigrad_io_data.py` |
| `deploy/nautilus/deployment-jupyter-lab.yaml` | `unc-jupyter` + Service | Jupyter Lab on `/files` |
| `deploy/nautilus/job-create-unigrad-io-data.yaml` | Job `unc-unigrad-io-data` | Batch full IO dataset |

| Script | Use |
|--------|-----|
| `deploy/nautilus/scripts/env.sh` | `HOME`, dataset paths, venv activate, `core.sshCommand` |
| `deploy/nautilus/scripts/setup_venv.sh` | One-time venv on PVC (`/files/venvs/unc`) |
| `deploy/nautilus/scripts/start-jupyter-lab.sh` | Called by Jupyter deployment entrypoint |
| `deploy/nautilus/scripts/ensure-system-deps.sh` | Manual `apt` for git/openssh if needed |
| `deploy/nautilus/scripts/check-git-ssh.sh` | Diagnose SSH paths on a pod |

GPU manifests use `nvidia.com/gpu.product` node affinity (V100 excluded). Widen the list in YAML if pods stay **Pending**.

---

## 1. PVC (once)

```bash
kubectl apply -f deploy/nautilus/pvc.yaml
kubectl get pvc unc-files
```

---

## 2. Clone repo & venv (once on PVC)

From **`unc-dev`** or **`unc-heavy`** after they are running:

```bash
kubectl apply -f deploy/nautilus/pod-pvc-admin.yaml
kubectl wait --for=condition=Ready pod/unc-dev --timeout=10m
kubectl exec -it unc-dev -- bash
```

Inside the pod:

```bash
mkdir -p /files/repo && cd /files/repo
git clone git@github.com:<org>/uncertainty-quantification.git
cd uncertainty-quantification
bash deploy/nautilus/scripts/setup_venv.sh
```

Or on GPU pod after `deployment-gpu.yaml` is up.

---

## 3. Git & SSH (once on PVC)

Stored on the volume, shared by all pods:

| What | Path |
|------|------|
| Git config | `/files/home/root/.gitconfig` |
| SSH key | `/files/home/root/.ssh/id_ed25519` |

**On `unc-dev`** (or any pod):

```bash
export HOME=/files/home/root
mkdir -p "${HOME}/.ssh" && chmod 700 "${HOME}/.ssh"
ssh-keygen -t ed25519 -f "${HOME}/.ssh/id_ed25519" -N ""
chmod 600 "${HOME}/.ssh/id_ed25519"
cat "${HOME}/.ssh/id_ed25519.pub"   # GitHub → Settings → SSH keys
git config --global user.name "Your Name"
git config --global user.email "YOUR_ID+you@users.noreply.github.com"
git config --global core.sshCommand "ssh -i ${HOME}/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new"
cd /files/repo/uncertainty-quantification && git pull
```

After that, `source deploy/nautilus/scripts/env.sh` keeps paths and `core.sshCommand` in sync.

**Note:** `git pull` uses `core.sshCommand`. Plain `ssh -T git@github.com` does not — use `ssh -i /files/home/root/.ssh/id_ed25519 -T git@github.com` or `export HOME=/files/home/root` first.

---

## 4. Start workloads

**PVC admin**

```bash
kubectl apply -f deploy/nautilus/pod-pvc-admin.yaml
kubectl exec -it unc-dev -- bash
```

**GPU interactive**

```bash
kubectl apply -f deploy/nautilus/deployment-gpu.yaml
kubectl rollout status deployment/unc-heavy --timeout=25m
kubectl exec -it deployment/unc-heavy -- bash
source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
```

**Jupyter Lab**

```bash
kubectl apply -f deploy/nautilus/deployment-jupyter-lab.yaml
kubectl rollout status deployment/unc-jupyter --timeout=25m
kubectl port-forward svc/unc-jupyter 8888:8888
```

Open http://127.0.0.1:8888 — kernel **unc**. In terminal: `bash`, then `source …/env.sh`.

**Batch IO job**

```bash
kubectl apply -f deploy/nautilus/job-create-unigrad-io-data.yaml
kubectl logs -f job/unc-unigrad-io-data
```

---

## 5. Copy data to PVC

From laptop (create dirs first):

```bash
kubectl exec unc-dev -- mkdir -p /files/repo/uncertainty-quantification/datasets/IXI
kubectl cp atlas.pkl unc-dev:/files/repo/uncertainty-quantification/datasets/IXI/atlas.pkl
# … Train/Val/Test *.pkl likewise
```

---

## 6. Experiments (3D IO)

Always:

```bash
source /files/repo/uncertainty-quantification/deploy/nautilus/scripts/env.sh
cd /files/repo/uncertainty-quantification
```

**IO iteration sweep** (pick IO budget; outputs under `assets/images/unigrad-io/3d/`):

```bash
python experiments/unigrad-io/sweep_io_iterations.py --mode 3d-pkl --split Train --num-subjects 5 --save-path ./assets/images/unigrad-io/3d/sweep_io.png --no-show
```

**Full 3D IO dataset** (resumable; one `.npz` per subject):

```bash
python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root ./datasets/IXI --atlas-pkl ./datasets/IXI/atlas.pkl --output-path ./datasets/IXI_unigrad_io --io-iterations 50
```

Smoke on a few subjects: add `--max-per-split 2 --splits Train`.

**GPU check:**

```bash
python experiments/resource_checks/diagnose_torch_gpu.py
```

Commit **`assets/`** from the cluster; pull on laptop. Large **`datasets/IXI_unigrad_io/`** stays on PVC only (gitignored).

---

## 7. Cleanup

```bash
kubectl delete pod unc-dev
kubectl delete deployment unc-heavy
kubectl delete deployment unc-jupyter
kubectl delete svc unc-jupyter
kubectl delete job unc-unigrad-io-data
```

PVC remains unless you delete `unc-files` separately.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| GPU pod **Pending** | `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{\t}{.metadata.labels.nvidia\.com/gpu\.product}{\n}{end}'` — add GPU model to YAML `values:` |
| `git pull` **publickey** | Key at `/files/home/root/.ssh/id_ed25519` + registered on GitHub; set `core.sshCommand` (see §3) |
| `git` / `ssh` missing | Pods install git/ssh on start; or `bash deploy/nautilus/scripts/ensure-system-deps.sh` |
| `set: pipefail` in YAML | Manifests must be **LF** (see `.gitattributes`) |
| Jupyter `source: not found` | Run `bash` first, or `. /files/venvs/unc/bin/activate` |
| Slow `git status` on PVC | `git status -uno` or narrow paths; many untracked/large files under `datasets/` |

---

## Quick reference

| Task | Command |
|------|---------|
| Env + venv | `source deploy/nautilus/scripts/env.sh` |
| 3D IO sweep | `python experiments/unigrad-io/sweep_io_iterations.py --mode 3d-pkl …` |
| 3D IO data | `python experiments/unigrad-io/create_unigrad_io_data.py --ixi-root ./datasets/IXI …` |
| Check SSH | `bash deploy/nautilus/scripts/check-git-ssh.sh` |

See also `commands.sh` on the repo root for common `kubectl` one-liners.
