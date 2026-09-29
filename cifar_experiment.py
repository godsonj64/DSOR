"""Reproducible DSORNet-v3.1 CIFAR experiment runner.

This module is intentionally self-contained: it trains the public DSORNet-v3.1
implementation on CIFAR-10 or CIFAR-100, repeats the run across explicit seeds,
and saves checkpoints plus a machine-readable JSON summary.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from dsorn_v31 import (
    DSORNetV31Sequential,
    evaluate,
    make_cifar,
    train,
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")
    return device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--output-dir", type=Path, default=Path("./cifar_results"))
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.8e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260929])
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:N, or mps",
    )
    args = parser.parse_args()

    for name in ("train_per_class", "test_per_class", "epochs", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain unique integers")
    return args


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    results = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===")
        seed_all(seed)

        train_ds, test_ds, num_classes = make_cifar(
            args.dataset,
            str(args.data_dir),
            args.train_per_class,
            args.test_per_class,
        )

        generator = torch.Generator().manual_seed(seed)
        pin_memory = device.type == "cuda"

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=max(args.batch_size, 256),
            shuffle=False,
            num_workers=0,
            pin_memory=pin_memory,
        )

        model = DSORNetV31Sequential(num_classes=num_classes)
        train(
            model,
            train_loader,
            test_loader,
            device,
            args.epochs,
            args.lr,
        )

        accuracy = evaluate(model, test_loader, device)
        checkpoint = args.output_dir / f"dsorn_v31_{args.dataset}_seed{seed}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "dataset": args.dataset,
                "num_classes": num_classes,
                "seed": seed,
                "epochs": args.epochs,
                "accuracy": accuracy,
            },
            checkpoint,
        )

        record = {
            "seed": seed,
            "test_accuracy": accuracy,
            "checkpoint": checkpoint.name,
        }
        results.append(record)
        print(json.dumps(record, indent=2))

    accuracies = np.asarray([r["test_accuracy"] for r in results], dtype=float)
    summary = {
        "dataset": args.dataset,
        "device": str(device),
        "train_per_class": args.train_per_class,
        "test_per_class": args.test_per_class,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "runs": results,
        "mean_test_accuracy": float(accuracies.mean()),
        "std_test_accuracy": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nsummary={summary_path.resolve()}")


if __name__ == "__main__":
    main()
