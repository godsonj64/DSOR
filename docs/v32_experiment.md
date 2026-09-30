# Reproducible DSOR experiments and Colab checklist

## What changed

v3.1 remains the default. Experimental v3.2 carries the full fine-stage sampling
mixture into coarse routing and retains particles with an exact scalar mixture
update. `v32-mean` provides the same-parameter mean-only ablation. All models
share the new training protocol: disjoint validation, keyed augmentation/order,
AdamW, warmup/cosine scheduling, label smoothing, MixUp, and residual-energy
regularization. These training changes differ from the historical v3.1 recipe.

The three-seed pilot found 40.22%±1.02 pp for v3.1 and 40.11%±0.30 pp for v3.2.
Full and mean-only v3.2 validation means were 35.53% and 35.47%. These results do
not support promoting v3.2 on accuracy. All numbers, split hashes, environment,
source hashes and per-seed selections are in `evidence/pilot_results.json`.
The exact original pilot source is commit `1fb94df122bcb9316d6316841dc5e2f6539affbc`.
Its default v3.2 graph is unchanged by adding the later mean-only ablation.

## Infrastructure findings and implemented fixes

| Inspected issue | Fix in this branch |
|---|---|
| Notebook deletes its current checkout, then rerun can fail from a deleted cwd | Change to `/content`, reuse the checkout, stop on source changes; no deletion |
| Clone follows moving main | Immutable runtime commit pin; verify HEAD |
| Notebook `!pip` / `!pytest` can fail without stopping subsequent cells | `subprocess.run` with `sys.executable` and `check=True` |
| Missing nbformat 4.5 cell IDs | Unique deterministic IDs; schema and code-cell compilation tests |
| Checkpoint cell is commented and depends on imports in earlier optional cells | One active, self-contained experiment launch |
| Basic run executes before the checkpointed run | Training runs through the checkpointed runner once |
| GPU check only prints forward finiteness | Assert loss/logits/gradients/parameters and perform an optimizer step |
| Drive mount/copy commented, hardcoded copy source | Configure Drive before training; runner writes to the chosen persistent path |
| Weights saved only at the end of each seed | Atomic `last.pt` after every completed epoch, with optimizer/scheduler/scaler/RNG |
| Best/last writes can be interrupted between files | Last checkpoint owns selected weights and metadata; rebuild best export on resume |
| Output files silently overwritten | Fresh run refuses nonempty output directory; resume checks run identity |
| No validation and test used every epoch | Balanced validation from training split; test after validation selection |
| Requested counts silently capped | Reject impossible class counts; save actual counts and split/data hashes |
| RNG-consuming internal tests perturb training initialization | CLI delegates to the runner; explicit seed immediately before construction |
| Model initialization changes augmentation RNG consumption | Keyed per-image/epoch augmentation and per-epoch shuffle; keyed per-batch dropout/MixUp |
| NaN LR/seeds/nonfinite logits/gradients can pass unnoticed | Finite/range checks, finite evaluation/training, error_if_nonfinite clipping |
| Single-seed sample SD is reported as zero | Null SD until two or more seeds |
| Open-ended dependencies drift across Python versions | Hash locks for Linux CPython 3.11/3.12, identical versions and version-specific wheels |
| Conditional PyTorch setuptools dependency missing on Python 3.12 | Explicit pinned setuptools in both locks |
| CI checks two tests and misses notebook/checkpoint/torchvision failures | Expanded geometry, distribution, resume, notebook, import and dependency checks |
| README/audit refer to unavailable archive checks and work paths | Current executable reproduction commands and measured evidence |

The locks freeze a reference CPU stack, not Colab's rolling CUDA stack. Colab
preserves its supplied torch/torchvision/NumPy, imports torchvision, runs tests,
and writes pip freeze beside the run. Exact compatibility with every future
Colab image is not assumed.

## Pilot reproduction

Create a clean Python 3.11 or 3.12 Linux environment. Install the matching lock:

```bash
python -m pip install --require-hashes -r requirements-ci-3.12-lock.txt
python -m pip check
python -m pytest -q
```

Use the pilot source commit above for exact original training hashes. The
current branch also supports the mean-only and validation-only options.
Run each architecture in a separate empty output directory with the same flags:

```bash
python cifar_experiment.py \
  --architecture v31 --dataset cifar10 --data-dir data \
  --train-per-class 100 --val-per-class 50 --test-per-class 1000 \
  --epochs 30 --warmup-epochs 3 --batch-size 128 \
  --seeds 20260929 20260930 20260931 --split-seed 20260929 \
  --device cpu --threads 2 --output-dir runs/pilot_v31
```

Repeat with `--architecture v32 --output-dir runs/pilot_v32`. The published
pilot used Python 3.12.14, torch 2.5.1+cpu, torchvision 0.20.1+cpu and NumPy
2.5.2; the reference CI lock uses NumPy 2.2.6. The recorded manifest preserves
this distinction. Cross-version numerical equality is not promised. The
runner fingerprints source, dataset and environment for recovery; an old run
must be resumed using its original stack/source, not this changed runner.

For development ablations, add `--validation-only`, use `--architecture
v32-mean` and a new output directory. No official test metrics are computed.
Do not rank additional variants on repeated test evaluations.

## Checkpoint recovery

Use the exact original command plus `--resume`. Preserve total epochs,
architecture, seeds, batch size, precision, split seed and all training flags.
Changing total epochs also changes the cosine schedule and is intentionally
rejected. `--stop-after-epoch N` pauses without testing and does not alter the
schedule; it is useful for a recovery drill.

```bash
# Configure the whole run but pause after epoch 1.
python cifar_experiment.py --architecture v32 --device cuda \
  --epochs 40 --stop-after-epoch 1 --output-dir runs/recovery_demo

# Continue the same 40-epoch configuration.
python cifar_experiment.py --architecture v32 --device cuda \
  --epochs 40 --resume --output-dir runs/recovery_demo
```

A run contains:

```text
manifest.json                 # source/data/split hashes, config, environment
seed_<seed>/last.pt            # epoch-boundary recovery and selected weights
seed_<seed>/best.pt            # canonical validation-selected model export
seed_<seed>/history.json       # training/validation per epoch
seed_<seed>/result.json        # final test, or null test for validation-only
summary.json                  # incrementally saved completed-seed results
status.json                   # paused or complete
```

`torch.load(path, map_location="cpu", weights_only=True)` reads these primitive
and tensor checkpoints. Reconstruct using `MODELS[checkpoint["architecture"]]`
and `num_classes`, then strictly load `model_state_dict`. Prepared inference
states have a different schema; prepare a canonical loaded model separately.
Only one writer may use a run directory. Epoch recovery replays an incomplete
epoch. Atomic replacement prevents partial checkpoint reads; Google Drive's
remote synchronization/durability is not controlled by Python. A completed
result is reused on resume, avoiding repeated final test computation.

## Final Colab run checklist

1. Open the notebook from this reviewed branch, choose an NVIDIA GPU runtime,
   and run the CUDA/torchvision check. Record Python, torch, torchvision, CUDA,
   GPU and the immutable CODE_REVISION.
2. Run checkout/dependency and test cells. Confirm each stops on failure. Both
   notebook schemas/code cells are checked in CI; the actual GPU checks must
   pass in Colab, including backward and the CUDA AMP test.
3. Mount Drive before training. Keep the generated run name for recovery or
   enter an explicit new name. Confirm the output points into MyDrive and
   RESUME=False for a new run. Never reuse a directory for changed settings.
4. Confirm dataset/counts: CIFAR-10 has at most 5,000 train+validation and 1,000
   test images/class; CIFAR-100 has 500 and 100. The default notebook is a
   1,000-training-image CIFAR-10 pilot. Use the same splits, precision and 3+
   seeds for architecture comparisons.
5. Run the single checkpointed training cell. Confirm the persistent manifest,
   pip-freeze record and first `seed_*/last.pt` appear. Training should log
   `val_acc`, not test accuracy each epoch.
6. Perform a separate recovery drill with `--stop-after-epoch 1`, then resume
   the identical configuration. Restore the original run name and pinned
   source after a runtime reset. Do not change total epochs on resume.
7. After completion, check `status=complete`, completed/requested seeds, all
   epoch histories, finite saved tensors and strict best-checkpoint reload.
   Confirm the official test was evaluated after validation selection.
8. Save the whole run directory and sibling pip-freeze file. For efficiency
   claims run `tools/benchmark.py --device cuda` on the same accelerator/batch
   settings, including peak memory. Label subset results as pilots.

A full CIFAR-10 protocol can reserve 500 validation images/class and train on
4,500/class, with all 1,000 test images/class. CIFAR-100 uses 450/50/100. Choose
and freeze the schedule on validation before final reporting. Include a
standard capacity-matched baseline, v2, v3.1, mean-only and full-memory
ablations; report accuracy, NLL, seed variation, latency and memory together.
