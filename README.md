# DSORNet-v3.1

**Dynamic Spatial Operator Routing with Cross-Scale Trajectory State Routing (CSTR)** is a convolution-free image model that learns continuous, content-dependent sampling coordinates and transports fine-stage spatial trajectories into a coarse routing stage.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/godsonj64/DSOR/blob/main/colab/DSORNet_v31_Colab.ipynb)

## Core idea

For fine-stage token position (p_i^{(1)}), DSOR predicts fractional offsets

$$
s_{im}^{(1)} = p_i^{(1)} + \Delta_{im}^{(1)}.
$$

The routing-weighted expected absolute sample location is

$$
\bar{s}_i^{(1)} = \sum_m \alpha_{im}^{(1)} s_{im}^{(1)}.
$$

For a fine-token group (G_j), the trajectory is transported into the coarse coordinate system:

$$
\hat{s}_j^{(2)} =
\mathcal{T}_{1\rightarrow 2}
\left(
\frac{1}{|G_j|}
\sum_{i\in G_j}
\bar{s}_i^{(1)}
\right),
\qquad
\pi_j^{(2)} = \hat{s}_j^{(2)} - p_j^{(2)}.
$$

Stage 2 combines inherited motion with a bounded residual:

$$
\Delta_{jm}^{(2)} =
g_j \odot \pi_j^{(2)} + r_{jm}^{(2)}.
$$

Later coarse blocks recurrently update the coordinate state from realized routed motion rather than predicting independent deformation fields from scratch.

## Repository

```text
DSOR/
├── dsorn_v31.py          # model + CIFAR-10/100 trainer
├── deployment.py         # frozen inference preparation
├── imaging.py            # image classification/restoration utilities
├── cifar_experiment.py   # research comparison driver
├── tests/                # executable core checks
├── colab/                # one-click CUDA notebook
├── requirements.txt
└── requirements-colab.txt
```

Datasets, checkpoints, caches, logs, and generated experiment outputs are excluded from version control.

## Local setup

```bash
git clone https://github.com/godsonj64/DSOR.git
cd DSOR
python -m pip install -r requirements.txt
pytest -q
```

## Train on CIFAR-10

```bash
python dsorn_v31.py \
  --dataset cifar10 \
  --train-per-class 100 \
  --test-per-class 50 \
  --epochs 12 \
  --batch-size 128 \
  --device cuda
```

For Apple Silicon, use `--device mps`. For CPU-only execution, use `--device cpu`.

## Train on CIFAR-100

```bash
python dsorn_v31.py \
  --dataset cifar100 \
  --train-per-class 25 \
  --test-per-class 20 \
  --epochs 15 \
  --batch-size 128 \
  --device cuda
```

The torchvision datasets are downloaded automatically when missing.

## Google Colab / GPU

Open the notebook with the badge above. In Colab, select **Runtime → Change runtime type → T4 GPU** or another NVIDIA accelerator. The notebook verifies CUDA, clones this repository, installs only the extra Colab dependencies, runs the test suite, and launches the same `dsorn_v31.py` training path with `--device cuda`.

This keeps Colab's preinstalled CUDA-enabled PyTorch build intact instead of replacing it with a generic wheel.

## Verification

The cleaned source archive was tested locally with:

```text
28 passed, 1 skipped
```

The skipped regression is Apple-MPS-specific when MPS hardware is unavailable. A real CIFAR-10 CPU smoke train also completed end-to-end.

## Research status

This repository is a research prototype. The present results support investigation of recurrent cross-scale coordinate inheritance, not a state-of-the-art or novelty claim. Stronger claims require a literature audit, matched-parameter baselines, multiple independent seeds, full-dataset training, and standardized evaluation.
