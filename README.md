# DSORNet-v3.1 — Cross-Scale Trajectory State Routing

This package implements the tested **sequential coordinate-inheritance** version
of DSORNet.

## Core idea

Stage 1 produces multiple continuous sample positions and routing weights:

$$
s_{im}^{(1)} = p_i^{(1)} + \Delta_{im}^{(1)}.
$$

Their attention-weighted expected absolute location is

$$
\bar s_i^{(1)} =
\sum_m \alpha_{im}^{(1)} s_{im}^{(1)}.
$$

For each 2×2 fine-token group \(G_j\), the coarse trajectory target is

$$
\hat s_j^{(2)} =
\mathcal T_{1\rightarrow2}
\left(
\frac1{|G_j|}
\sum_{i\in G_j}
\bar s_i^{(1)}
\right),
$$

and the inherited coarse displacement is

$$
\pi_j^{(2)} =
\hat s_j^{(2)} - p_j^{(2)}.
$$

Stage 2 does not predict a fresh unrelated offset. It predicts

$$
\Delta_{jm}^{(2)} =
g_j \odot \pi_j^{(2)} + r_{jm}^{(2)},
$$

where \(g_j\) is a learned inheritance gate and \(r_{jm}^{(2)}\) is a
bounded residual correction.

After the first coarse block, the coordinate state is recurrently updated:

$$
\tau_j^{(\ell+1)} =
\tau_j^{(\ell)} +
\eta_j^{(\ell)}
\odot
\left(
\mu_j^{(\ell)}-\tau_j^{(\ell)}
\right),
$$

where \(\mu_j^{(\ell)}\) is the attention-weighted realized displacement.

This creates an explicit **coarse-to-fine-to-coarse spatial search trajectory**.

## Executed tests

- Convolution modules: **0**
- Zero-motion transport error: **1.19e-7**
- Constant-motion transport error: **1.49e-7**
- Cross-scale transport gradient norm: **0.0420**
- Full forward/backward: **passed**
- Stage-2 coordinate bounds: **[-1, 1]**

## Small real-image ablation

Dataset: scikit-learn digits, transformed to 32×32 RGB with random
non-label-dependent appearance variation.

- Train: 1,347
- Test: 450
- Epochs: 10

| Model | Params | Clean accuracy | 2px-shift accuracy |
|---|---:|---:|---:|
| v2 independent offsets | 120,884 | 94.89% | 74.00% |
| v3 static inheritance | 142,536 | 94.44% | 74.00% |
| **v3.1 sequential trajectory** | 145,466 | **95.56%** | **74.44%** |

The most diagnostic change was trajectory alignment in the second coarse block:

- Stage-2 block 1 prior/realized cosine: **0.383**
- Stage-2 block 2 prior/realized cosine: **0.804**

The recurrent state update therefore made later routing substantially more
consistent with the inherited search trajectory.

## Real CIFAR-10 small-batch run

```bash
python -m pip install torch torchvision numpy pytest
python dsorn_v31.py   --dataset cifar10   --train-per-class 100   --test-per-class 50   --epochs 12   --device mps
```

## Real CIFAR-100 small-batch run

```bash
python dsorn_v31.py   --dataset cifar100   --train-per-class 25   --test-per-class 20   --epochs 15   --device mps
```

## Tests

```bash
pytest -q
```

## Research caution

This is a research prototype. The small pilot supports the *mechanistic*
hypothesis that recurrent coordinate-state inheritance can produce more coherent
routing and slightly better small-data accuracy. It does **not** establish
state-of-the-art performance or novelty. A literature audit, matched-parameter
baselines, multiple random seeds, and real CIFAR/ImageNet-scale experiments are
needed before making such claims.
