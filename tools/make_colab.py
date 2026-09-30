"""Generate validated notebooks; pin to a commit that contains the runtime code."""
import argparse
from pathlib import Path
import re

import nbformat as nbf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--code-revision', required=True)
args = parser.parse_args()
if not re.fullmatch('[0-9a-f]{40}', args.code_revision):
    parser.error('--code-revision must be an immutable full commit SHA')
root = Path(__file__).resolve().parents[1]

for architecture, filename in [('v31', 'DSORNet_v31_Colab.ipynb'), ('v32', 'DSORNet_v32_Colab.ipynb')]:
    title = 'v3.1 baseline' if architecture == 'v31' else 'v3.2 distribution-memory experiment'
    cells = [
        nbf.v4.new_markdown_cell(f'''# DSORNet {title} — Colab GPU

Select **Runtime → Change runtime type → GPU**. Run these cells in order.
The experiment selects checkpoints on disjoint validation data and tests once.
v3.2 is a research hypothesis; improved accuracy and literature novelty need evidence.
'''),
        nbf.v4.new_code_cell('''import platform
import torch
import torchvision

assert torch.cuda.is_available(), "Select a GPU runtime before continuing"
print({"python": platform.python_version(), "torch": str(torch.__version__),
       "torchvision": str(torchvision.__version__), "cuda": torch.version.cuda,
       "gpu": torch.cuda.get_device_name(0)})
# Importing torchvision here verifies the preinstalled torch/vision pair.
'''),
        nbf.v4.new_markdown_cell('''## Checkout and dependencies

The notebook pins runtime code to an immutable commit. Checkout can be rerun
from inside the repository. Changed tracked files stop the checkout. If changing
the revision after importing model modules, restart the runtime first.
Colab's CUDA-enabled torch/torchvision and NumPy are preserved.
'''),
        nbf.v4.new_code_cell(f'''from pathlib import Path
import os
import subprocess
import sys

CODE_REVISION = "{args.code_revision}"
REPO_DIR = Path("/content/DSOR")
os.chdir("/content")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone",
                    "https://github.com/godsonj64/DSOR.git", str(REPO_DIR)], check=True)
if not (REPO_DIR / ".git").is_dir():
    raise RuntimeError("/content/DSOR exists but is not the expected git checkout")
remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=REPO_DIR, text=True).strip()
if remote != "https://github.com/godsonj64/DSOR.git":
    raise RuntimeError("Checkout origin differs from the requested repository")
if subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_DIR, text=True).strip():
    raise RuntimeError("Tracked or untracked source changes exist; inspect them before checkout")
current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
if current != CODE_REVISION and any(name in sys.modules for name in ("dsorn_v31", "dsorn_v32")):
    raise RuntimeError("Restart the runtime before changing the imported model revision")
subprocess.run(["git", "fetch", "origin", CODE_REVISION], cwd=REPO_DIR, check=True)
subprocess.run(["git", "checkout", "--detach", CODE_REVISION], cwd=REPO_DIR, check=True)
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip() == CODE_REVISION
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-colab.txt"], check=True)
'''),
        nbf.v4.new_markdown_cell('''## Tests and GPU backward smoke

A failing test or pip command stops execution. The CUDA test includes float16
AMP; this explicit smoke also checks ordinary float32 backward and an optimizer step.
'''),
        nbf.v4.new_code_cell('''import subprocess
import sys

subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_DIR, check=True)
'''),
        nbf.v4.new_code_cell(f'''import torch
import torch.nn.functional as F
from dsorn_v31 import DSORNetV31Sequential
from dsorn_v32 import DSORNetV32Distribution, trajectory_energy

ARCHITECTURE = "{architecture}"
model_class = DSORNetV32Distribution if ARCHITECTURE == "v32" else DSORNetV31Sequential
smoke_model = model_class(num_classes=10).cuda()
optimizer = torch.optim.AdamW(smoke_model.parameters(), lr=1e-3)
x = torch.randn(4, 3, 32, 32, device="cuda")
y = torch.arange(4, device="cuda")
logits, aux = smoke_model(x, return_aux=True)
assert logits.shape == (4, 10) and torch.isfinite(logits).all()
loss = F.cross_entropy(logits, y) + .01 * trajectory_energy(aux)
assert torch.isfinite(loss)
loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in smoke_model.parameters())
torch.nn.utils.clip_grad_norm_(smoke_model.parameters(), 2., error_if_nonfinite=True)
optimizer.step()
assert all(torch.isfinite(p).all() for p in smoke_model.parameters())
print("CUDA forward, backward, and optimizer step passed")
del smoke_model, optimizer, logits, aux, loss, x, y
torch.cuda.empty_cache()
'''),
        nbf.v4.new_markdown_cell('''## Configure storage before training

Drive is enabled by default so every completed epoch survives runtime loss.
Authorize the mount when Colab asks. Local storage is suitable for a disposable smoke.
A rerun keeps RUN_NAME; for recovery set RESUME=True and restore the original
run name and all settings. Changing source, data, environment or configuration
is rejected on resume. A new experiment needs a new RUN_NAME.
'''),
        nbf.v4.new_code_cell(f'''from pathlib import Path
from datetime import datetime, timezone
import uuid

PERSIST_TO_DRIVE = True
RESUME = False
if PERSIST_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    OUTPUT_ROOT = Path("/content/drive/MyDrive/DSOR")
else:
    OUTPUT_ROOT = Path("/content/dsor_runs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
if "RUN_NAME" not in globals():
    RUN_NAME = "{architecture}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
EXPERIMENT_DIR = OUTPUT_ROOT / RUN_NAME

DATASET = "cifar10"
TRAIN_PER_CLASS = 100
VAL_PER_CLASS = 50
TEST_PER_CLASS = 1000  # CIFAR-100: at most 100/class
EPOCHS = 40
WARMUP_EPOCHS = 3
BATCH_SIZE = 128
SEEDS = [20260929]  # For a comparison use the same 3+ seeds for both architectures.
SPLIT_SEED = 20260929
AMP = False  # Optional CUDA float16; use the same precision for paired experiments.
print({{"architecture": ARCHITECTURE, "output": str(EXPERIMENT_DIR), "resume": RESUME,
       "dataset": DATASET, "epochs": EPOCHS, "seeds": SEEDS, "amp": AMP}})
'''),
        nbf.v4.new_markdown_cell('''## Run the checkpointed experiment

This is the training cell; there is no separate uncheckpointed run.
`last.pt` includes model, selected weights, optimizer, scheduler, scaler, RNG,
and epoch. `best.pt` exports validation-selected weights. The manifest records
exact splits, data/source hashes, config, and hardware. Recovery starts at the
last completed epoch; a partly completed epoch is replayed. Atomic rename
protects against partial files, but Drive synchronization is outside Python's control.
'''),
        nbf.v4.new_code_cell('''import subprocess
import sys

freeze_path = OUTPUT_ROOT / (RUN_NAME + ".pip-freeze.txt")
if not RESUME:
    if freeze_path.exists():
        raise FileExistsError("This run name already has an environment record")
    freeze_path.write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
cmd = [sys.executable, str(REPO_DIR / "cifar_experiment.py"),
       "--architecture", ARCHITECTURE, "--dataset", DATASET,
       "--data-dir", "/content/data", "--output-dir", str(EXPERIMENT_DIR),
       "--train-per-class", str(TRAIN_PER_CLASS), "--val-per-class", str(VAL_PER_CLASS),
       "--test-per-class", str(TEST_PER_CLASS), "--epochs", str(EPOCHS),
       "--warmup-epochs", str(WARMUP_EPOCHS), "--batch-size", str(BATCH_SIZE),
       "--seeds", *map(str, SEEDS), "--split-seed", str(SPLIT_SEED), "--device", "cuda"]
if AMP:
    cmd.append("--amp")
if RESUME:
    cmd.append("--resume")
print(" ".join(cmd))
subprocess.run(cmd, cwd=REPO_DIR, check=True)
'''),
        nbf.v4.new_markdown_cell('''## Verify saved artifacts

For each seed check that selection came from validation, all weights are finite,
and the saved classifier reloads. A single seed has no sample standard deviation.
'''),
        nbf.v4.new_code_cell('''import json
import torch
from cifar_experiment import MODELS

manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())
summary = json.loads((EXPERIMENT_DIR / "summary.json").read_text())
assert summary["completed_seeds"] == len(SEEDS)
assert json.loads((EXPERIMENT_DIR / "status.json").read_text())["state"] == "complete"
for record in summary["runs"]:
    directory = EXPERIMENT_DIR / f"seed_{record['seed']}"
    last = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    assert last["epoch"] == EPOCHS and last["run_id"] == manifest["run_id"] == best["run_id"]
    assert all(torch.isfinite(t).all() for t in best["model_state_dict"].values())
    restored = MODELS[best["architecture"]](num_classes=best["num_classes"]).cuda().eval()
    restored.load_state_dict(best["model_state_dict"], strict=True)
    with torch.no_grad():
        assert torch.isfinite(restored(torch.zeros(1, 3, 32, 32, device="cuda"))).all()
    del restored
print(json.dumps(summary, indent=2))
print("Saved run:", EXPERIMENT_DIR)
'''),
        nbf.v4.new_markdown_cell('''## Interpretation and next experiment

Small subsets are pilots. For a full CIFAR-10 run use 4,500 training and 500
validation images/class, the full 1,000 test images/class, and a predeclared
longer schedule. Compare v2, v3.1 and v3.2 with the same recipe and 3+ paired seeds.
Tune on validation, freeze the recipe, then report the final test result once.

There are no Conv modules, but patch embedding and merging are mathematically
strided convolutions. Sampling geometry is learned, not an uncertainty estimate
or proof of causality. Coordinate recurrence is across depth within one image.
CUDA grid_sample backward can be nondeterministic; bitwise GPU recovery is not promised.
See `math_audit.md` and `docs/v32_experiment.md` for the assumptions and ablation plan.
'''),
    ]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}, 'accelerator': 'GPU',
        'colab': {'name': filename, 'provenance': []}})
    for index, cell in enumerate(notebook.cells):
        cell.id = f'{architecture}-cell-{index:02d}'
    nbf.validate(notebook)
    destination = root/'colab'/filename
    nbf.write(notebook, destination)
    print(destination)
