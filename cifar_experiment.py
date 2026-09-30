"""CIFAR experiment with disjoint validation, atomic checkpoints and epoch resume.

Architecture comparisons share splits, augmentation, batch order and objectives.
The official test split is evaluated once using the validation-best checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from dsorn_v31 import DSORNetV2Independent, DSORNetV31Sequential
from dsorn_v32 import DSORNetV32Distribution, trajectory_energy

SCHEMA = 2
MODELS = {"v2": DSORNetV2Independent, "v31": DSORNetV31Sequential,
          "v32": DSORNetV32Distribution,
          "v32-mean": lambda **kwargs: DSORNetV32Distribution(memory_mode="mean", **kwargs)}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def keyed_seed(*parts):
    """Private RNG key independent of model initialization draw count."""
    payload = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def resolve_device(requested):
    if requested == "auto":
        requested = ("cuda" if torch.cuda.is_available() else
                     "mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda[:N], mps or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return device


def stratified_split(targets, train_per_class, val_per_class, seed):
    rng = np.random.default_rng(seed)
    train, validation = [], []
    targets = np.asarray(targets)
    for label in np.unique(targets):
        indices = rng.permutation(np.flatnonzero(targets == label))
        if train_per_class + val_per_class > len(indices):
            raise ValueError(f"class {label}: requested train+validation exceeds {len(indices)}")
        train.extend(indices[:train_per_class].tolist())
        validation.extend(indices[train_per_class:train_per_class + val_per_class].tolist())
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def stratified_test(targets, per_class, seed):
    indices, _ = stratified_split(targets, per_class, 0, seed)
    return indices


class KeyedImages(Dataset):
    """Crop/flip keyed by (run seed, epoch, original image index).

    num_workers=0 keeps the current epoch explicit. Epoch-boundary restarts
    reconstruct identical augmentation and shuffle order on a fixed stack.
    """
    def __init__(self, data, targets, indices, mean, std, seed, augment=False):
        self.data, self.targets, self.indices = data, targets, list(indices)
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.seed, self.augment, self.epoch = seed, augment, 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        image = torch.from_numpy(np.array(self.data[index], copy=True)).permute(2, 0, 1).float() / 255
        if self.augment:
            generator = torch.Generator().manual_seed(keyed_seed(self.seed, self.epoch, index, "image"))
            image = F.pad(image, (4, 4, 4, 4), mode="reflect")
            top, left = torch.randint(9, (2,), generator=generator).tolist()
            image = image[:, top:top+32, left:left+32]
            if torch.rand((), generator=generator).item() < 0.5:
                image = image.flip(-1)
        return (image - self.mean) / self.std, int(self.targets[index])


def indices_digest(indices):
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def load_data(args):
    from torchvision import datasets
    if args.dataset == "cifar10":
        cls, classes = datasets.CIFAR10, 10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    else:
        cls, classes = datasets.CIFAR100, 100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    train = cls(root=str(args.data_dir), train=True, download=True)
    test = cls(root=str(args.data_dir), train=False, download=True)
    train_idx, val_idx = stratified_split(train.targets, args.train_per_class,
                                        args.val_per_class, args.split_seed)
    test_idx = stratified_test(test.targets, args.test_per_class, args.split_seed + 1)
    splits = {"train": train_idx, "validation": val_idx, "test": test_idx}
    digest = hashlib.sha256()
    for images, labels in ((train.data, train.targets), (test.data, test.targets)):
        digest.update(np.asarray(images).tobytes())
        digest.update(np.asarray(labels, dtype="<i8").tobytes())
    metadata = {"num_classes": classes, "dataset_sha256": digest.hexdigest(),
                "normalization": {"mean": mean, "std": std},
                "counts": {k: len(v) for k, v in splits.items()},
                "split_sha256": {k: indices_digest(v) for k, v in splits.items()},
                "indices": splits}
    return train, test, mean, std, metadata


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rng_state():
    state_np = np.random.get_state()
    state = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "numpy": [state_np[0], state_np[1].tolist(), state_np[2], state_np[3], state_np[4]]}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
        state["mps"] = torch.mps.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    state_np = state["numpy"]
    np.random.set_state((state_np[0], np.asarray(state_np[1], dtype=np.uint32), *state_np[2:]))
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def source_identity():
    root = Path(__file__).parent
    names = ("dsorn_v31.py", "dsorn_v32.py", "cifar_experiment.py")
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                           stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "git_dirty": dirty, "sha256": hashes}


def environment(device):
    import torchvision
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "device": str(device), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads()}


def configuration(args):
    excluded = {"output_dir", "data_dir", "resume", "stop_after_epoch"}
    return {key: value for key, value in vars(args).items() if key not in excluded}


def run_identity(config, data, source, env):
    # A docs-only commit can change; actual training source hashes must match.
    payload = {"schema": SCHEMA, "config": config, "data": data,
               "source_sha256": source["sha256"], "environment": env}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        logits = model(x.to(device))
        y = y.to(device)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("nonfinite evaluation logits")
        batch_loss = F.cross_entropy(logits, y, reduction="sum")
        if not torch.isfinite(batch_loss):
            raise FloatingPointError("nonfinite evaluation loss")
        loss_sum += batch_loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
    if not total:
        raise ValueError("cannot evaluate an empty dataset")
    return {"accuracy": correct / total, "loss": loss_sum / total, "count": total}


def make_scheduler(optimizer, epochs, warmup):
    def scale(epoch):
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, (epoch - warmup) / max(1, epochs - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(model, loader, optimizer, scaler, device, args, seed, epoch):
    model.train()
    loader.dataset.epoch = epoch
    loss_sum, count = 0.0, 0
    for step, (x, y) in enumerate(loader):
        batch_seed = keyed_seed(seed, epoch, step, "batch")
        seed_all(batch_seed)
        x, y = x.to(device), y.to(device)
        if args.mixup > 0:
            rng = np.random.default_rng(batch_seed)
            lam = float(rng.beta(args.mixup, args.mixup))
            permutation = torch.randperm(len(y), generator=torch.Generator().manual_seed(batch_seed)).to(device)
            x, other = lam * x + (1-lam) * x[permutation], y[permutation]
        else:
            lam, other = 1.0, y
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp):
            logits, aux = model(x, return_aux=True)
            loss = (lam * F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
                    + (1-lam) * F.cross_entropy(logits, other, label_smoothing=args.label_smoothing)
                    + args.trajectory_weight * trajectory_energy(aux))
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite training value at epoch {epoch}, step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * len(y)
        count += len(y)
    return loss_sum / count


def run_seed(args, seed, device, loaded, run_id):
    train_data, test_data, mean, std, data = loaded
    indices = data["indices"]
    train_ds = KeyedImages(train_data.data, train_data.targets, indices["train"], mean, std, seed, True)
    val_ds = KeyedImages(train_data.data, train_data.targets, indices["validation"], mean, std, seed)
    test_ds = KeyedImages(test_data.data, test_data.targets, indices["test"], mean, std, seed)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 256), num_workers=0)
    seed_all(seed)
    model = MODELS[args.architecture](num_classes=data["num_classes"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    directory = args.output_dir / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    result_path = directory / "result.json"
    history, best, best_weights, start = [], None, None, 0
    if args.resume and last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["run_id"] != run_id or checkpoint["seed"] != seed:
            raise ValueError("checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        history, best, start = checkpoint["history"], checkpoint["best"], checkpoint["epoch"]
        best_weights = checkpoint["best_model_state_dict"]
        restore_rng(checkpoint["rng_state"])
        # last.pt owns both latest and selected weights: a crash between writes
        # cannot pair a future best.pt with an older optimizer/selection record.
        atomic_checkpoint(best_path, {"schema": SCHEMA, "run_id": run_id, "seed": seed,
                          "architecture": args.architecture, "num_classes": data["num_classes"],
                          "model_state_dict": best_weights, "validation": best})
    if args.resume and result_path.exists():
        result = json.loads(result_path.read_text())
        if result["run_id"] != run_id or start != args.epochs:
            raise ValueError("inconsistent completed seed")
        return result
    for epoch in range(start, args.epochs):
        began = time.monotonic()
        train_ds.epoch = epoch
        generator = torch.Generator().manual_seed(keyed_seed(seed, epoch, "shuffle"))
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=device.type == "cuda")
        learning_rate = optimizer.param_groups[0]["lr"]
        train_loss = train_epoch(model, loader, optimizer, scaler, device, args, seed, epoch)
        validation = evaluate(model, val_loader, device)
        scheduler.step()
        record = {"epoch": epoch + 1, "train_loss": train_loss, "lr": learning_rate,
                  "validation": validation, "seconds": time.monotonic() - began}
        history.append(record)
        improved = (best is None or validation["accuracy"] > best["accuracy"] or
                    (validation["accuracy"] == best["accuracy"] and validation["loss"] < best["loss"]))
        if improved:
            best = {**validation, "epoch": epoch + 1}
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        checkpoint = {"schema": SCHEMA, "run_id": run_id, "seed": seed,
                      "architecture": args.architecture, "num_classes": data["num_classes"],
                      "epoch": epoch + 1, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "rng_state": rng_state(),
                      "history": history, "best": best, "best_model_state_dict": best_weights,
                      "shuffle_policy": "keyed seed/epoch; reconstruct at epoch boundary"}
        atomic_checkpoint(last_path, checkpoint)
        atomic_checkpoint(best_path, {"schema": SCHEMA, "run_id": run_id, "seed": seed,
                          "architecture": args.architecture, "num_classes": data["num_classes"],
                          "model_state_dict": best_weights, "validation": best})
        atomic_json(directory / "history.json", history)
        print(f"seed={seed} epoch={epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_acc={validation['accuracy']:.4f} seconds={record['seconds']:.1f}", flush=True)
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch and epoch + 1 < args.epochs:
            return None
    chosen = torch.load(best_path, map_location="cpu", weights_only=True)
    if chosen["run_id"] != run_id:
        raise ValueError("best checkpoint identity mismatch")
    model.load_state_dict(chosen["model_state_dict"])
    # Test once after validation selection; paused training never tests.
    test_loader = DataLoader(test_ds, batch_size=max(args.batch_size, 256), num_workers=0)
    test_metrics = None if args.validation_only else evaluate(model, test_loader, device)
    result = {"run_id": run_id, "seed": seed, "architecture": args.architecture,
              "parameters": sum(p.numel() for p in model.parameters()),
              "best_validation": best, "test": test_metrics,
              "checkpoint": str(best_path.relative_to(args.output_dir))}
    atomic_json(result_path, result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=tuple(MODELS), default="v31")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("cifar_results"))
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.8e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument("--trajectory-weight", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260929])
    parser.add_argument("--split-seed", type=int, default=20260929)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="CUDA float16 autocast; default float32")
    parser.add_argument("--resume", action="store_true", help="resume the same configuration from last.pt")
    parser.add_argument("--validation-only", action="store_true", help="development ablation: never evaluate test data")
    parser.add_argument("--stop-after-epoch", type=int, help="pause at an epoch boundary without testing")
    args = parser.parse_args(argv)
    for name in ("train_per_class", "val_per_class", "test_per_class", "epochs", "batch_size", "threads"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "weight_decay", "mixup", "trajectory_weight", "label_smoothing"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name == "lr" and value == 0):
            parser.error(f"--{name.replace('_', '-')} must be finite and valid")
    if args.label_smoothing >= 1:
        parser.error("--label-smoothing must be less than 1")
    if not 0 <= args.warmup_epochs < args.epochs:
        parser.error("--warmup-epochs must be between 0 and epochs-1")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        parser.error("--stop-after-epoch must be between 1 and epochs")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be unique")
    if any(not 0 <= seed < 2**32 for seed in [*args.seeds, args.split_seed]):
        parser.error("seeds must be in [0, 2**32)")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp requires CUDA")
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if args.resume and not manifest_path.exists():
        raise FileNotFoundError("--resume requires an existing manifest.json")
    if not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError("output directory is not empty; choose a new path or use --resume")
    loaded = load_data(args)
    data = loaded[-1]
    config, source, env = configuration(args), source_identity(), environment(device)
    identity = run_identity(config, data, source, env)
    manifest = {"schema": SCHEMA, "run_id": identity, "config": config, "data": data,
                "source": source, "environment": env,
                "resume_policy": "epoch boundaries, identical source/data/environment/config",
                "cuda_reproducibility": "grid_sample backward may be nondeterministic"}
    if args.resume:
        previous = json.loads(manifest_path.read_text())
        if previous["run_id"] != identity:
            raise ValueError("resume configuration, source, dataset or environment changed")
    else:
        atomic_json(manifest_path, manifest)
    print(f"architecture={args.architecture} device={device} samples={data['counts']}", flush=True)
    results = []
    for seed in args.seeds:
        result = run_seed(args, seed, device, loaded, identity)
        if result is None:
            atomic_json(args.output_dir / "status.json", {"state": "paused", "seed": seed})
            return
        results.append(result)
        accuracies = [item["test"]["accuracy"] for item in results if item["test"] is not None]
        summary = {"run_id": identity, "architecture": args.architecture, "runs": results,
                   "completed_seeds": len(results), "requested_seeds": len(args.seeds),
                   "mean_validation_accuracy": float(np.mean([item["best_validation"]["accuracy"] for item in results])),
                   "mean_test_accuracy": float(np.mean(accuracies)) if accuracies else None,
                   "std_test_accuracy": float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else None}
        atomic_json(args.output_dir / "summary.json", summary)
    atomic_json(args.output_dir / "status.json", {"state": "complete"})
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
