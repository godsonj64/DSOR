# DSOR: dynamic spatial routing

DSOR learns continuous, content-dependent spatial samples and carries routing
geometry across scales in a small image classifier. v3.1 is the default;
v3.2 distribution memory is experimental. There are no explicit convolution
modules, although patch embedding and token merging are mathematically strided
convolutions.

[Open v3.1 in Colab](https://colab.research.google.com/github/godsonj64/DSOR/blob/main/colab/DSORNet_v31_Colab.ipynb) ·
[Open v3.2 in Colab](https://colab.research.google.com/github/godsonj64/DSOR/blob/main/colab/DSORNet_v32_Colab.ipynb)

## Measured status

A controlled three-seed CPU pilot trained on 1,000 CIFAR-10 images, selected
checkpoints on 500 disjoint validation images, and tested on all 10,000 official
test images. With identical 30-epoch training settings, v3.1 achieved
40.22%±1.02 percentage points; v3.2 achieved 40.11%±0.30. This does not establish
an accuracy gain. Full-memory and same-parameter mean-only v3.2 validation
means were 35.53% and 35.47%, also inconclusive. These are subset pilots,
not state-of-the-art or full-training benchmarks.

See [mathematical audit](math_audit.md), [protocol and Colab checklist](docs/v32_experiment.md),
and [machine-readable evidence](docs/evidence/pilot_results.json).

## Models

- `dsorn_v31.py`: independent v2 baseline and sequential mean-state v3.1.
- `dsorn_v32.py`: experimental particle distribution, kernel-conditioned reads,
  covariance-aware features and a scalar probability-mixture state update.
- `deployment.py`: separate frozen copy with exact algebraic projection folding
  and fixed-grid position caches. Preserve canonical weights for training.
- `imaging.py`: classification/restoration utilities; identity initialization
  is a reconstruction sanity check, not restoration-quality evidence.
- `cifar_experiment.py`: validation-first multi-seed training and epoch recovery.

CIFAR-10 parameter counts: v2 120,884; v3.1 145,466; v3.2 144,121.
The same-parameter `v32-mean` ablation collapses carried geometry before each
coarse read. Learned sparse offsets are established prior art; the proposed
cross-scale distribution-memory combination requires a broader literature
review and accuracy/cost ablations before claiming novelty.

## Setup

Python 3.11/3.12 on Linux has separate hash locks with identical package versions:

```bash
git clone https://github.com/godsonj64/DSOR.git
cd DSOR
python -m pip install --require-hashes -r requirements-ci-3.12-lock.txt
python -m pip check
python -m pytest -q
```

Use `requirements-ci-3.11-lock.txt` for Python 3.11. These are CPU reference
stacks. For CUDA or Apple MPS, install a matched torch/torchvision build for the
platform, then `python -m pip install -r requirements.txt`. In Colab use
`requirements-colab.txt` to preserve its CUDA-enabled torch/vision/NumPy.
Local Python 3.11.16 and 3.12.14 checks passed **31 tests, one CUDA test skipped**;
actual Colab/T4 execution was not available during this audit.

## Train and resume

```bash
python cifar_experiment.py \
  --architecture v31 --dataset cifar10 --device cuda \
  --train-per-class 100 --val-per-class 50 --test-per-class 1000 \
  --epochs 40 --warmup-epochs 3 --batch-size 128 \
  --seeds 20260929 20260930 20260931 --output-dir runs/v31_pilot
```

Use `--architecture v32` for the experimental model. For exploratory ablations,
use `--architecture v32-mean --validation-only` with a new output directory.
`--device auto` selects CUDA, MPS, then CPU. CUDA float16 is opt-in with `--amp`;
compare architectures at identical precision. `python dsorn_v31.py ...`
delegates to this same checkpointed runner.

Each completed epoch atomically saves `seed_*/last.pt`, including model,
selected weights, optimizer, scheduler, scaler, RNG and history. `best.pt`
exports validation-selected weights. The manifest saves exact indices,
source/data hashes, configuration and environment. Completed seed results and
summary are written incrementally. One seed has null sample standard deviation.

To recover, repeat the original command with `--resume`. Source, data, stack,
seeds and schedule must match; recovery replays an incomplete epoch. A new run
refuses a nonempty output directory. The official test is evaluated after
validation selection, with completed results reused on resume. Keyed
augmentation, shuffle, MixUp and dropout make epoch recovery reproducible on a
fixed CPU stack; CUDA grid_sample backward may be nondeterministic.

## Colab

Choose a GPU runtime and run cells in order. The notebook pins runtime code,
fails on install/test errors, verifies forward/backward/optimizer behavior,
mounts Drive before training, runs the checkpointed experiment once, and checks
saved artifacts. Use a new run name for new settings. For recovery preserve the
old name, configuration, environment and pinned revision, and set RESUME=True.

[Full run checklist and recovery drill](docs/v32_experiment.md#final-colab-run-checklist)

## Mathematical and performance checks

```bash
python tools/audit_math.py --repo . --output runs/math-evidence.json
python tools/benchmark.py --device cuda --output runs/gpu-benchmark.json
```

The v3.1 optimized router costs O(BND²+BNMD), without an N×N attention matrix.
v3.2 adds O(BNMK) particle-kernel work. Fewer parameters or dense MACs do not
imply faster execution. Its current memory uses 16 then 21 particles; extending
the mixture to many blocks needs a bounded-memory design. See the audit for
exact MAC counts, finite-gradient and covariance proofs, counterexamples and
limits of the architectural interpretation.
