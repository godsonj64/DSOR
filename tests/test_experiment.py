import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import cifar_experiment as experiment


def fixture_data(args):
    rng = np.random.default_rng(41)
    images = rng.integers(0, 256, (16, 32, 32, 3), dtype=np.uint8)
    raw = SimpleNamespace(data=images, targets=[0, 1]*8)
    metadata = {'num_classes': 2, 'counts': {'train': 8, 'validation': 4, 'test': 4},
                'indices': {'train': list(range(8)), 'validation': list(range(8, 12)),
                            'test': list(range(12, 16))}, 'dataset_sha256': 'fixture-only'}
    return raw, raw, (0.5,)*3, (0.25,)*3, metadata


def test_split_disjoint_balanced_and_rejects_silent_truncation():
    labels = [0]*10 + [1]*10
    train, val = experiment.stratified_split(labels, 6, 3, 7)
    assert not set(train) & set(val)
    assert len(train) == 12 and len(val) == 6
    assert [sum(labels[i] == c for i in train) for c in (0, 1)] == [6, 6]
    assert (train, val) == experiment.stratified_split(labels, 6, 3, 7)
    with pytest.raises(ValueError, match='exceeds'):
        experiment.stratified_split(labels, 8, 3, 7)


def test_augmentation_independent_of_global_rng_draws():
    raw, _, mean, std, _ = fixture_data(None)
    dataset = experiment.KeyedImages(raw.data, raw.targets, [0], mean, std, 17, True)
    a = dataset[0][0]
    torch.rand(1000)
    np.random.rand(1000)
    assert torch.equal(a, dataset[0][0])
    dataset.epoch = 1
    assert not torch.equal(a, dataset[0][0])


@pytest.mark.parametrize('option,value', [('--lr', 'nan'), ('--lr', 'inf'), ('--lr', '0'),
                                        ('--seeds', '-1'), ('--seeds', str(2**32)),
                                        ('--mixup', 'nan'), ('--label-smoothing', '1')])
def test_argument_finite_and_seed_guards(option, value):
    with pytest.raises(SystemExit):
        experiment.parse_args([option, value])


def test_nonfinite_evaluation_cannot_return_plausible_accuracy():
    class Broken(torch.nn.Module):
        def forward(self, x):
            return torch.full((len(x), 2), float('nan'))
    loader = DataLoader(TensorDataset(torch.zeros(2, 3, 32, 32), torch.zeros(2, dtype=torch.long)))
    with pytest.raises(FloatingPointError):
        experiment.evaluate(Broken(), loader, torch.device('cpu'))


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            equal_tree(av, bv)
    else:
        assert a == b


def test_epoch_resume_matches_uninterrupted_and_restores_best(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'load_data', fixture_data)
    full, paused = tmp_path/'full', tmp_path/'paused'
    base = ['--architecture', 'v32', '--device', 'cpu', '--epochs', '3', '--warmup-epochs', '1',
            '--batch-size', '4', '--seeds', '17']
    experiment.main([*base, '--output-dir', str(full)])
    experiment.main([*base, '--output-dir', str(paused), '--stop-after-epoch', '1'])
    assert not (paused/'seed_17/result.json').exists()  # No test evaluation when paused.
    (paused/'seed_17/best.pt').unlink()  # Recover selected weights from authoritative last.pt.
    experiment.main([*base, '--output-dir', str(paused), '--resume'])
    a = torch.load(full/'seed_17/last.pt', weights_only=True)
    b = torch.load(paused/'seed_17/last.pt', weights_only=True)
    for key in ('model_state_dict', 'best_model_state_dict', 'optimizer_state_dict',
                'scheduler_state_dict', 'scaler_state_dict', 'rng_state', 'best'):
        equal_tree(a[key], b[key])
    results = [json.loads((p/'seed_17/result.json').read_text()) for p in (full, paused)]
    assert results[0] == results[1]
    summary = json.loads((full/'summary.json').read_text())
    assert summary['std_test_accuracy'] is None
    with pytest.raises(FileExistsError):
        experiment.main([*base, '--output-dir', str(full)])
    with pytest.raises(ValueError, match='changed'):
        experiment.main([*base, '--lr', '.001', '--output-dir', str(paused), '--resume'])
    # A completed resume uses the cached test result.
    monkeypatch.setattr(experiment, 'evaluate', lambda *a: pytest.fail('unexpected repeat evaluation'))
    experiment.main([*base, '--output-dir', str(paused), '--resume'])


def test_validation_only_does_not_consult_test_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'load_data', fixture_data)
    original = experiment.evaluate
    calls = []
    def spy(model, loader, device):
        calls.append(loader.dataset.indices)
        return original(model, loader, device)
    monkeypatch.setattr(experiment, 'evaluate', spy)
    experiment.main(['--architecture', 'v32-mean', '--device', 'cpu', '--epochs', '1',
                     '--warmup-epochs', '0', '--seeds', '19', '--validation-only',
                     '--output-dir', str(tmp_path/'ablation')])
    assert calls == [list(range(8, 12))]
    result = json.loads((tmp_path/'ablation/summary.json').read_text())
    assert result['mean_test_accuracy'] is None
