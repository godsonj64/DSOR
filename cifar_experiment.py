"""Paired CIFAR-10 pilot with a held-out validation split and a final test phase.

Examples (run from this directory)::
    python cifar_experiment.py --data-dir ../../work/data --phase train \
        --variants original fixed_optimized fixed_low_lr --epochs 8
    python cifar_experiment.py --phase test --output-dir cifar_results

The test phase uses the saved validation-selected checkpoints and refuses to
re-evaluate an already tested experiment. Small-subset accuracy is exploratory,
not an estimate of a fully trained CIFAR-10 model's final performance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10

MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(3, 1, 1)
STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(3, 1, 1)
HERE = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def keyed_seed(*values):
    data = ':'.join(str(v) for v in values).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little') % (2**63 - 1)


def digest_indices(indices):
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest_state(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def runtime_metadata(device):
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'numpy': np.__version__,
        'device': str(device),
        'torch_num_threads': torch.get_num_threads(),
        'mps_available': torch.backends.mps.is_available(),
        'cuda_available': torch.cuda.is_available(),
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'determinism_note': (
            'Initialization, data split, sample order, per-image augmentation and '
            'per-batch dropout RNG seeds are paired across arms. Deterministic '
            'algorithms use warn_only=True; accelerator reductions may still '
            'produce floating-point differences.'
        ),
    }


def make_splits(train_targets, test_targets, train_per_class, val_per_class,
                test_per_class, seed):
    rng = np.random.default_rng(seed)
    train_targets, test_targets = np.asarray(train_targets), np.asarray(test_targets)
    train_indices, val_indices, test_indices = [], [], []
    for label in range(10):
        available = rng.permutation(np.flatnonzero(train_targets == label))
        if train_per_class + val_per_class > len(available):
            raise ValueError('Requested training and validation sizes exceed available data')
        train_indices.extend(available[:train_per_class].tolist())
        val_indices.extend(available[train_per_class:train_per_class + val_per_class].tolist())
        available_test = rng.permutation(np.flatnonzero(test_targets == label))
        if test_per_class > len(available_test):
            raise ValueError('Requested test size exceeds available data')
        test_indices.extend(available_test[:test_per_class].tolist())
    assert not set(train_indices).intersection(val_indices)
    return {
        'split_seed': seed,
        'dataset': 'CIFAR-10',
        'train_indices': train_indices,
        'validation_indices': val_indices,
        'test_indices': test_indices,
        'index_source': {
            'train_indices': 'official training set',
            'validation_indices': 'official training set, disjoint from training indices',
            'test_indices': 'official test set',
        },
        'sha256': {
            'train': digest_indices(train_indices),
            'validation': digest_indices(val_indices),
            'test': digest_indices(test_indices),
        },
    }


class PairedImages(Dataset):
    """Augmentation uses a private CPU RNG, never the model's dropout RNG."""
    def __init__(self, full_dataset, indices, seed=0, augment=False):
        self.full = full_dataset
        self.indices = list(indices)
        self.seed = seed
        self.augment = augment
        self.epoch = 0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        raw_index = self.indices[position]
        x = torch.from_numpy(self.full.data[raw_index].copy()).permute(2, 0, 1).float().div_(255)
        if self.augment:
            generator = torch.Generator().manual_seed(keyed_seed('augment', self.seed, self.epoch, raw_index))
            top, left = torch.randint(0, 9, (2,), generator=generator).tolist()
            x = F.pad(x, (4, 4, 4, 4))[:, top:top + 32, left:left + 32]
            if torch.rand((), generator=generator).item() < 0.5:
                x = x.flip(-1)
        return (x - MEAN) / STD, int(self.full.targets[raw_index])


def loader_for(dataset, batch_size, seed, order=None):
    if order is not None:
        dataset = Subset(dataset, order)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                      drop_last=False, generator=torch.Generator().manual_seed(seed))


def trajectory_regularizer(aux):
    # Identical objective in all arms; matches the submitted implementation.
    regularizers = [
        (stage['weights'].unsqueeze(-1) * stage['residual']).sum(2).square().mean()
        for stage in aux['stage2']
    ]
    return torch.stack(regularizers).mean()


@torch.no_grad()
def evaluate(model, loader, device, save_predictions=False):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    predictions, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError('Non-finite evaluation logits')
        loss_sum += float(F.cross_entropy(logits, y, reduction='sum').item())
        pred = logits.argmax(1)
        correct += int((pred == y).sum().item())
        count += len(y)
        if save_predictions:
            predictions.extend(pred.cpu().tolist())
            targets.extend(y.cpu().tolist())
    result = {'loss': loss_sum / count, 'accuracy': correct / count,
              'correct': correct, 'count': count}
    if save_predictions:
        result.update(predictions=predictions, targets=targets)
    return result


def finite_parameters(model):
    return bool(torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item())


def train_arm(module, name, learning_rate, state, seed, train_full, split, args, device):
    arm_dir = args.output_dir / f'seed_{seed}' / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = module.DSORNetV31Sequential(num_classes=10)
    model.load_state_dict(state, strict=True)
    assert digest_state(model.state_dict()) == digest_state(state)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_data = PairedImages(train_full, split['train_indices'], seed=seed, augment=True)
    val_data = PairedImages(train_full, split['validation_indices'])
    val_loader = loader_for(val_data, args.batch_size, keyed_seed('val_loader', seed))
    history, best_key, best_epoch = [], None, None
    checkpoint_path = arm_dir / 'best.pt'
    synchronize(device)
    arm_started = time.perf_counter()
    try:
        for epoch in range(1, args.epochs + 1):
            train_data.epoch = epoch
            order_seed = keyed_seed('order', seed, epoch)
            order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(order_seed)).tolist()
            loader = loader_for(train_data, args.batch_size, keyed_seed('train_loader', seed, epoch), order)
            model.train()
            loss_sum, ce_sum, regularizer_sum, correct, count = 0.0, 0.0, 0.0, 0, 0
            grad_norms = []
            epoch_lr = optimizer.param_groups[0]['lr']
            synchronize(device)
            epoch_started = time.perf_counter()
            for batch, (x, y) in enumerate(loader):
                seed_all(keyed_seed('dropout', seed, epoch, batch))
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, aux = model(x, return_aux=True)
                ce = F.cross_entropy(logits, y)
                regularizer = trajectory_regularizer(aux)
                loss = ce + args.regularizer_weight * regularizer
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {batch}')
                loss.backward()
                # error_if_nonfinite checks all gradients via the aggregate norm.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad,
                                                     error_if_nonfinite=True)
                grad_norms.append(float(norm.item()))
                optimizer.step()
                n = len(y)
                count += n
                loss_sum += float(loss.item()) * n
                ce_sum += float(ce.item()) * n
                regularizer_sum += float(regularizer.item()) * n
                correct += int((logits.detach().argmax(1) == y).sum().item())
            if not finite_parameters(model):
                raise FloatingPointError(f'Non-finite parameter after epoch {epoch}')
            synchronize(device)
            train_seconds = time.perf_counter() - epoch_started
            validation = evaluate(model, val_loader, device)
            synchronize(device)
            epoch_seconds = time.perf_counter() - epoch_started
            scheduler.step()
            row = {
                'epoch': epoch, 'learning_rate': epoch_lr,
                'train_loss': loss_sum / count, 'train_cross_entropy': ce_sum / count,
                'train_regularizer': regularizer_sum / count, 'train_accuracy': correct / count,
                'train_count': count, 'validation': validation,
                'all_gradients_finite': True, 'all_parameters_finite': True,
                'gradient_norm_mean_before_clip': statistics.mean(grad_norms),
                'gradient_norm_max_before_clip': max(grad_norms),
                'train_seconds': train_seconds, 'epoch_seconds': epoch_seconds,
                'train_examples_per_second': count / train_seconds,
                'sample_order_sha256': digest_indices([train_data.indices[i] for i in order]),
            }
            history.append(row)
            current_key = (validation['accuracy'], -validation['loss'])
            if best_key is None or current_key > best_key:
                best_key, best_epoch = current_key, epoch
                torch.save({
                    'model_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    'epoch': epoch, 'variant': name, 'seed': seed,
                    'validation': validation, 'learning_rate': learning_rate,
                }, checkpoint_path)
            write_json(arm_dir / 'history.json', history)
            print(f'{name} seed={seed} epoch={epoch:02d}/{args.epochs} '
                  f'train_loss={row["train_loss"]:.4f} train_acc={row["train_accuracy"]:.3f} '
                  f'val_loss={validation["loss"]:.4f} val_acc={validation["accuracy"]:.3f} '
                  f'seconds={epoch_seconds:.2f}', flush=True)
    except Exception as exc:
        write_json(arm_dir / 'failure.json', {'variant': name, 'seed': seed,
                   'error': repr(exc), 'completed_epochs': len(history)})
        raise
    synchronize(device)
    return {
        'variant': name, 'seed': seed, 'learning_rate': learning_rate,
        'initial_state_sha256': digest_state(state), 'best_epoch': best_epoch,
        'best_validation': history[best_epoch - 1]['validation'],
        'checkpoint': str(checkpoint_path.relative_to(args.output_dir)),
        'checkpoint_sha256': digest_file(checkpoint_path),
        'history': str((arm_dir / 'history.json').relative_to(args.output_dir)),
        'elapsed_seconds': time.perf_counter() - arm_started,
    }


def mean_and_sd(values):
    return {'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values) > 1 else None,
            'n': len(values)}


def run_train(args, device):
    manifest_path = args.output_dir / 'manifest.json'
    saved_manifest = None
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError('Output already contains an experiment; use --resume or a new --output-dir')
        saved_manifest = json.loads(manifest_path.read_text())
        summary_path = args.output_dir / 'summary.json'
        if summary_path.exists() and json.loads(summary_path.read_text())['status'] != 'trained_test_not_evaluated':
            raise RuntimeError('Cannot resume training after final test has started')
    elif args.resume:
        raise FileNotFoundError('--resume requires an existing manifest.json')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    original = load_module(args.original, 'dsorn_original_experiment')
    improved = load_module(args.improved, 'dsorn_improved_experiment')
    # Loading metadata/data is allowed here; test predictions are not computed until run_test.
    train_full = CIFAR10(root=str(args.data_dir), train=True, download=args.download)
    test_full = CIFAR10(root=str(args.data_dir), train=False, download=args.download)
    split = make_splits(train_full.targets, test_full.targets, args.train_per_class,
                        args.val_per_class, args.test_per_class, args.split_seed)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        'configuration': config, 'runtime': runtime_metadata(device),
        'sources': {
            'original': {'path': str(args.original), 'sha256': digest_file(args.original)},
            'improved': {'path': str(args.improved), 'sha256': digest_file(args.improved)},
            'driver': {'path': str(Path(__file__).resolve()), 'sha256': digest_file(__file__)},
        },
        'split_sha256': split['sha256'],
        'training_objective': 'cross_entropy + regularizer_weight * mean_stage2_squared_expected_residual',
        'checkpoint_selection': 'maximum validation accuracy, then minimum validation cross-entropy',
        'recipe_selection': 'mean best-validation accuracy across seeds, then mean validation cross-entropy',
        'test_policy': 'official test evaluated only after recipe selection; one pass per selected arm and seed',
    }
    records = []
    if saved_manifest is not None:
        ignored_keys = {'phase', 'resume', 'download'}
        changed = [k for k in config if k not in ignored_keys and
                   saved_manifest['configuration'].get(k) != config[k]]
        if changed:
            raise RuntimeError(f'Resume configuration differs: {changed}; repeat the original training arguments')
        for kind in ('original', 'improved'):
            if saved_manifest['sources'][kind]['sha256'] != manifest['sources'][kind]['sha256']:
                raise RuntimeError(f'{kind} source changed; resume requires the original training source')
        if saved_manifest['split_sha256'] != split['sha256']:
            raise RuntimeError('Data splits changed since the original run')
        records_path = args.output_dir / 'training_records.json'
        if records_path.exists():
            records = json.loads(records_path.read_text())
        for record in records:
            checkpoint_path = args.output_dir / record['checkpoint']
            if digest_file(checkpoint_path) != record['checkpoint_sha256']:
                raise RuntimeError('A completed checkpoint changed; refusing to resume')
            history = json.loads((args.output_dir / record['history']).read_text())
            if len(history) != args.epochs:
                raise RuntimeError('A completed arm has incomplete history')
    else:
        write_json(args.output_dir / 'splits.json', split)
        write_json(manifest_path, manifest)
    for seed in args.seeds:
        seed_all(seed)
        initial = original.DSORNetV31Sequential(num_classes=10)
        state = {k: v.detach().cpu().clone() for k, v in initial.state_dict().items()}
        seed_dir = args.output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        initial_path = seed_dir / 'initial_state.pt'
        if initial_path.exists():
            if digest_state(torch.load(initial_path, map_location='cpu', weights_only=True)) != digest_state(state):
                raise RuntimeError('Saved initialization differs from the paired seeded initialization')
        else:
            torch.save(state, initial_path)
        del initial
        for name in args.variants:
            if any(r['seed'] == seed and r['variant'] == name for r in records):
                print(f'Reusing completed arm {name} seed={seed}', flush=True)
                continue
            module = original if name == 'original' else improved
            lr = args.candidate_lr if name == 'fixed_low_lr' else args.lr
            records.append(train_arm(module, name, lr, state, seed, train_full, split, args, device))
            write_json(args.output_dir / 'training_records.json', records)
    candidates = [v for v in args.variants if v != 'original']
    aggregated = {}
    for name in args.variants:
        rows = [r for r in records if r['variant'] == name]
        aggregated[name] = {
            'best_validation_accuracy': mean_and_sd([r['best_validation']['accuracy'] for r in rows]),
            'best_validation_loss': mean_and_sd([r['best_validation']['loss'] for r in rows]),
            'training_seconds': mean_and_sd([r['elapsed_seconds'] for r in rows]),
        }
    winner = max(candidates, key=lambda name: (
        aggregated[name]['best_validation_accuracy']['mean'],
        -aggregated[name]['best_validation_loss']['mean']))
    summary = {
        'status': 'trained_test_not_evaluated', 'records': records, 'aggregate_validation': aggregated,
        'selected_improved_variant': winner, 'final_test_variants': ['original', winner],
        'selection_frozen_before_test': True,
    }
    write_json(args.output_dir / 'summary.json', summary)
    print(f'Validation-selected improved recipe: {winner}. Test has not been evaluated.', flush=True)


def wilson_interval(correct, count):
    z = 1.959963984540054
    p, z2 = correct / count, z * z
    center = (p + z2 / (2 * count)) / (1 + z2 / count)
    radius = z * math.sqrt(p * (1 - p) / count + z2 / (4 * count * count)) / (1 + z2 / count)
    return [center - radius, center + radius]


def run_test(args, device):
    summary_path = args.output_dir / 'summary.json'
    summary = json.loads(summary_path.read_text())
    test_path = args.output_dir / 'test_results.json'
    if test_path.exists() or summary['status'] != 'trained_test_not_evaluated':
        raise RuntimeError('Final test was already started/evaluated; no repeated test peeking')
    manifest = json.loads((args.output_dir / 'manifest.json').read_text())
    config = manifest['configuration']
    split = json.loads((args.output_dir / 'splits.json').read_text())
    modules = {}
    for kind in ('original', 'improved'):
        saved_source = manifest['sources'][kind]
        # CLI paths can relocate an unchanged package; source digest must still match.
        path = args.original if kind == 'original' else args.improved
        if digest_file(path) != saved_source['sha256']:
            raise RuntimeError(f'{kind} source changed after training; test would be an inconsistent comparison')
        modules[kind] = load_module(path, 'dsorn_final_' + kind)
    data_dir = Path(config['data_dir'])
    test_full = CIFAR10(root=str(data_dir), train=False, download=args.download)
    test_data = PairedImages(test_full, split['test_indices'])
    test_loader = loader_for(test_data, config['batch_size'], keyed_seed('test_loader', config['split_seed']))
    # Freeze the decision on disk before accessing model predictions. A failed
    # final test remains marked started rather than silently allowing test reuse.
    summary['status'] = 'final_test_started'
    write_json(summary_path, summary)
    test_records = []
    for record in summary['records']:
        if record['variant'] not in summary['final_test_variants']:
            continue
        checkpoint_path = args.output_dir / record['checkpoint']
        if digest_file(checkpoint_path) != record['checkpoint_sha256']:
            raise RuntimeError('Selected checkpoint changed after validation selection')
        seed_all(record['seed'])
        kind = 'original' if record['variant'] == 'original' else 'improved'
        model = modules[kind].DSORNetV31Sequential(num_classes=10).to(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state'], strict=True)
        synchronize(device)
        started = time.perf_counter()
        metrics = evaluate(model, test_loader, device, save_predictions=True)
        synchronize(device)
        metrics['seconds'] = time.perf_counter() - started
        metrics['accuracy_95pct_wilson_interval'] = wilson_interval(metrics['correct'], metrics['count'])
        result = {'variant': record['variant'], 'seed': record['seed'],
                  'selected_epoch': record['best_epoch'], **metrics}
        test_records.append(result)
        write_json(test_path, {'status': 'in_progress', 'records': test_records})
        print(f'FINAL TEST {record["variant"]} seed={record["seed"]}: '
              f'{metrics["correct"]}/{metrics["count"]} accuracy={metrics["accuracy"]:.3f}', flush=True)
    aggregate = {
        name: mean_and_sd([r['accuracy'] for r in test_records if r['variant'] == name])
        for name in summary['final_test_variants']
    }
    paired = []
    for seed in config['seeds']:
        rows = {r['variant']: r for r in test_records if r['seed'] == seed}
        original, improved = rows['original'], rows[summary['selected_improved_variant']]
        orig_ok = np.asarray(original['predictions']) == np.asarray(original['targets'])
        improved_ok = np.asarray(improved['predictions']) == np.asarray(improved['targets'])
        paired.append({
            'seed': seed, 'accuracy_difference': improved['accuracy'] - original['accuracy'],
            'improved_only_correct': int((improved_ok & ~orig_ok).sum()),
            'original_only_correct': int((orig_ok & ~improved_ok).sum()),
            'both_correct': int((orig_ok & improved_ok).sum()),
            'neither_correct': int((~orig_ok & ~improved_ok).sum()),
        })
    write_json(test_path, {
        'status': 'complete', 'records': test_records, 'aggregate_test_accuracy': aggregate,
        'paired_comparison': paired, 'runtime': runtime_metadata(device),
        'uncertainty_note': 'Wilson intervals quantify finite test-sample uncertainty only; '
                            'seed SD is separate. Seeds reuse the same test images and are not independent test samples.',
    })
    summary['status'] = 'complete'
    summary['test_results'] = 'test_results.json'
    write_json(summary_path, summary)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--phase', choices=['train', 'test', 'all'], default='all')
    parser.add_argument('--original', type=Path, default=HERE / 'reference' / 'dsorn_v31_original.py')
    parser.add_argument('--improved', type=Path, default=HERE / 'dsorn_v31.py')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'cifar_results')
    parser.add_argument('--download', action='store_true', help='Download CIFAR-10 if missing')
    parser.add_argument('--resume', action='store_true', help='Reuse completed arms; restart interrupted arms deterministically')
    parser.add_argument('--variants', nargs='+', choices=['original', 'fixed_optimized', 'fixed_low_lr'],
                        default=['original', 'fixed_optimized'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260929])
    parser.add_argument('--split-seed', type=int, default=20260929)
    parser.add_argument('--train-per-class', type=int, default=100)
    parser.add_argument('--val-per-class', type=int, default=25)
    parser.add_argument('--test-per-class', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2.8e-3)
    parser.add_argument('--candidate-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--regularizer-weight', type=float, default=0.01)
    parser.add_argument('--clip-grad', type=float, default=2.0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', default='cpu',
                        help='CPU is the portable fair baseline; accelerator use must support both arms')
    args = parser.parse_args(argv)
    for key in ('original', 'improved', 'data_dir', 'output_dir'):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for key in ('train_per_class', 'val_per_class', 'test_per_class', 'epochs', 'batch_size', 'threads'):
        if getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    for key in ('lr', 'candidate_lr', 'clip_grad'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be finite and positive')
    for key in ('weight_decay', 'regularizer_weight'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) < 0:
            parser.error(f'--{key.replace("_", "-")} must be finite and nonnegative')
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 or s >= 2**63 for s in args.seeds):
        parser.error('--seeds must be unique integers in [0, 2**63)')
    if args.split_seed < 0:
        parser.error('--split-seed must be nonnegative')
    if len(args.variants) != len(set(args.variants)) or 'original' not in args.variants or len(args.variants) < 2:
        parser.error('--variants must contain original and at least one improved variant, with no duplicates')
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    if args.phase in ('train', 'all'):
        run_train(args, device)
    if args.phase in ('test', 'all'):
        run_test(args, device)


if __name__ == '__main__':
    main()
