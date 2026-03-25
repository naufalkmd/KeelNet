# KeelNet Local GPU Runtime Guide

This guide is for running KeelNet on your own GPU from WSL2 while still using
the Google Colab frontend through a local Jupyter runtime.

## What We Confirmed On This Machine

- WSL2 is active.
- `nvidia-smi` sees `NVIDIA GeForce RTX 5070`.
- Docker is not currently available in this WSL distro.
- System `pip` is not installed, but `python3 -m venv` works and bootstraps pip.

Because of that, the simplest path here is:

1. create a Python virtual environment in WSL
2. install Jupyter plus a CUDA-enabled PyTorch build
3. connect Colab to the local Jupyter server
4. run KeelNet from that local runtime

## Stage 1 Notebook Support

The Stage 1 notebook now supports both:

- hosted Google Colab
- Google Colab connected to a local Jupyter runtime

In local-runtime mode, the setup cell:

- reuses your local repo instead of cloning from GitHub
- uses a local project folder for artifacts
- reads `HF_TOKEN` from the environment if you set it
- installs the repo into the current kernel environment

Optional environment variables:

- `KEELNET_REPO_DIR` to point at a specific local checkout
- `KEELNET_PROJECT_DIR` to change the local artifact root

## 1. Create A Local Environment

From the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install notebook ipykernel
```

## 2. Install PyTorch With CUDA

Use the official PyTorch selector for:

- OS: `Linux`
- Package: `Pip`
- Language: `Python`
- Compute Platform: `CUDA 12.8`

For this machine, that should currently be:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Then install the project itself:

```bash
python -m pip install -e .
```

## 3. Verify GPU Access

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

If `torch.cuda.is_available()` is `False`, stop there and fix PyTorch or the
WSL GPU stack before trying Colab local runtime.

## 4. Prepare A `/content` Directory

Google's local-runtime guide notes that some Colab frontend features expect a
`/content` directory to exist. Create it once:

```bash
sudo mkdir -p /content
sudo chown "$USER":"$USER" /content
```

Optional but convenient:

```bash
ln -sfn "$(pwd)" /content/KeelNet
mkdir -p /content/KeelNet-local
```

## 5. Start A Local Jupyter Runtime

Run:

```bash
mkdir -p /tmp/jupyter-runtime /tmp/jupyter-data
JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime \
JUPYTER_DATA_DIR=/tmp/jupyter-data \
jupyter notebook \
  --ServerApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --ServerApp.port_retries=0 \
  --ServerApp.allow_credentials=True \
  --ServerApp.ip=127.0.0.1 \
  --ServerApp.root_dir="$(pwd)" \
  --no-browser
```

Copy the full URL with the token, which looks like:

```text
http://127.0.0.1:8888/?token=...
```

## 6. Connect From Colab

In Colab:

1. click `Connect`
2. choose `Connect to local runtime...`
3. paste the Jupyter URL from the previous step

After that, notebook code runs on your local WSL machine and should use your
local GPU if PyTorch sees CUDA.

## 7. Run The Stage 1 Notebook

Open:

- `stages/01-grounded-abstention-baseline/notebooks/google-colab.ipynb`

Then run the notebook setup cell. In local-runtime mode it should:

- detect `local-runtime`
- use `/content/KeelNet` if that symlink exists, otherwise your current repo
- save artifacts under `/content/KeelNet-local` by default

If you want different paths, set these before starting Jupyter:

```bash
export KEELNET_REPO_DIR="$(pwd)"
export KEELNET_PROJECT_DIR="/content/KeelNet-local"
```

## 8. Optional CLI Smoke Test

If you want to verify the GPU path before using the notebook, run a tiny local
training job:

```bash
python -m keelnet.train \
  --mode baseline \
  --output-dir /content/KeelNet-local/smoke-baseline \
  --max-train-samples 32 \
  --max-eval-samples 32 \
  --num-train-epochs 1 \
  --train-batch-size 2 \
  --eval-batch-size 2
```

If that works, the local GPU path is healthy.
