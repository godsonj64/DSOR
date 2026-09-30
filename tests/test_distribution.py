import copy

import pytest
import torch
import torch.nn.functional as F

from deployment import prepare_for_inference
from dsorn_v31 import (DSORNetV31Sequential, base_grid, bilinear_sample_tokens,
                       normalized_entropy, TrajectoryStateUpdate)
from dsorn_v32 import (DSORNetV32Distribution, TrajectoryDistribution,
                       distribution_moments, mixture_update, transport_fine_distribution,
                       trajectory_energy)


def test_exact_distribution_transport_and_covariance():
    coords = base_grid(8, 8, dtype=torch.float64).unsqueeze(2).expand(1, 64, 4, 2).clone()
    weights = torch.rand(1, 64, 4, dtype=torch.float64)
    weights /= weights.sum(-1, keepdim=True)
    state = transport_fine_distribution({'coords': coords, 'weights': weights})
    mean, covariance = distribution_moments(state)
    torch.testing.assert_close(mean, base_grid(4, 4, dtype=torch.float64), atol=1e-14, rtol=0)
    expected = torch.eye(2, dtype=torch.float64).expand_as(covariance) / 36
    torch.testing.assert_close(covariance, expected, atol=1e-14, rtol=0)
    assert state.positions.max() > 1  # Do not clip frame coordinates before aggregation.
    assert torch.allclose(state.weights.sum(-1), torch.ones(1, 16, dtype=torch.float64))


def test_full_covariance_law_and_vector_convex_mean():
    old = TrajectoryDistribution(torch.tensor([[[[-.4, .2], [.3, -.1]]]], dtype=torch.float64),
                                 torch.tensor([[[.7, .3]]], dtype=torch.float64))
    new = TrajectoryDistribution(torch.tensor([[[[.8, -.6], [.2, .4]]]], dtype=torch.float64),
                                 torch.tensor([[[.4, .6]]], dtype=torch.float64))
    eta = torch.tensor([[[.3]]], dtype=torch.float64, requires_grad=True)
    mixed = mixture_update(old, new.positions, new.weights, eta)
    m0, c0 = distribution_moments(old)
    m1, c1 = distribution_moments(new)
    mean, covariance = distribution_moments(mixed)
    torch.testing.assert_close(mean, (1-eta)*m0 + eta*m1)
    delta = m1-m0
    expected = ((1-eta).unsqueeze(-1)*c0 + eta.unsqueeze(-1)*c1
                + (eta*(1-eta)).unsqueeze(-1)*delta.unsqueeze(-1)*delta.unsqueeze(-2))
    torch.testing.assert_close(covariance, expected)
    assert torch.linalg.eigvalsh(covariance).min() >= -1e-14
    (mean.square().sum()+covariance.sum()).backward()
    assert torch.isfinite(eta.grad).all()
    for endpoint, state in ((0., old), (1., new)):
        result = mixture_update(old, new.positions, new.weights, torch.full_like(eta, endpoint))
        for actual, wanted in zip(distribution_moments(result), distribution_moments(state)):
            torch.testing.assert_close(actual, wanted)


def test_coincident_particles_have_finite_gradients_and_preserve_small_variance():
    positions = torch.zeros(1, 2, 4, 2, requires_grad=True)
    weights = torch.full((1, 2, 4), .25, requires_grad=True)
    mean, covariance = distribution_moments(TrajectoryDistribution(positions, weights))
    (mean.sum()+covariance.sum()).backward()
    assert torch.isfinite(positions.grad).all() and torch.isfinite(weights.grad).all()
    small = torch.tensor([[[[-1e-4, 0.], [1e-4, 0.]]]], dtype=torch.float16)
    _, covariance = distribution_moments(TrajectoryDistribution(small, torch.full((1, 1, 2), .5).half()))
    assert covariance.dtype == torch.float32
    assert covariance[..., 0, 0].item() > 9e-9


def test_retained_state_distinguishes_equal_mean_distributions():
    separated = TrajectoryDistribution(torch.tensor([[[[-.3, 0.], [.3, 0.]]]]), torch.full((1, 1, 2), .5))
    coincident = TrajectoryDistribution(torch.zeros(1, 1, 2, 2), separated.weights)
    ma, ca = distribution_moments(separated)
    mb, cb = distribution_moments(coincident)
    assert torch.equal(ma, mb)
    assert ca[..., 0, 0].item() > .08 and torch.equal(cb, torch.zeros_like(cb))


def test_energy_penalty_cannot_cancel_opposite_samples():
    residual = torch.tensor([[[[.2, 0.], [-.2, 0.]]]], requires_grad=True)
    aux = {'stage2': [{'weights': torch.full((1, 1, 2), .5), 'residual': residual}]}
    loss = trajectory_energy(aux)
    torch.testing.assert_close(loss, torch.tensor(.04))
    loss.backward()
    assert residual.grad.abs().sum() > 0


@pytest.mark.parametrize('architecture', [DSORNetV31Sequential, DSORNetV32Distribution])
def test_prepared_logits_ownership_and_training_guard(architecture):
    torch.manual_seed(17)
    model = architecture().eval().double()
    x = torch.randn(2, 3, 32, 32, dtype=torch.float64)
    before = copy.deepcopy(model.state_dict())
    frozen = prepare_for_inference(model)
    with torch.no_grad():
        torch.testing.assert_close(model(x), frozen(x), atol=2e-12, rtol=2e-12)
    assert not any(p.requires_grad for p in frozen.parameters())
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    expected = frozen(x).clone()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert torch.equal(frozen(x), expected)
    with pytest.raises(RuntimeError, match='frozen'):
        frozen.train()


def test_v32_forward_backward_and_probability_invariants():
    model = DSORNetV32Distribution()
    x = torch.randn(3, 3, 32, 32, requires_grad=True)
    logits, aux = model(x, True)
    loss = F.cross_entropy(logits, torch.tensor([1, 2, 3])) + .01*trajectory_energy(aux)
    loss.backward()
    assert logits.shape == (3, 10) and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert aux['state1'].positions.shape == (3, 16, 16, 2)
    assert aux['state2'].positions.shape == (3, 16, 21, 2)
    for state in (aux['state1'], aux['state2']):
        assert (state.weights >= 0).all()
        torch.testing.assert_close(state.weights.sum(-1), torch.ones(3, 16))
        assert torch.linalg.eigvalsh(distribution_moments(state)[1]).min() >= -1e-6
    for stage in aux['stage2']:
        assert stage['coords'].abs().max() <= 1
        torch.testing.assert_close(stage['responsibilities'].sum(-1), torch.ones(3, 16, 5))


@pytest.mark.parametrize('h,w', [(4, 4), (1, 4), (4, 1), (1, 1)])
def test_sampler_native_and_grid_forward_backward(h, w):
    tokens = torch.randn(1, h*w, 3, dtype=torch.float64, requires_grad=True)
    coords = torch.rand(1, h*w, 3, 2, dtype=torch.float64, requires_grad=True)*2.8-1.4
    # Include both border equality and out-of-domain behavior.
    coords = coords.detach().requires_grad_()
    coords.data[0, 0, 0] = torch.tensor([-1., 1.])
    grid = bilinear_sample_tokens(tokens, coords, h, w, 'grid')
    gather = bilinear_sample_tokens(tokens, coords, h, w, 'gather')
    torch.testing.assert_close(grid, gather, atol=1e-12, rtol=1e-12)
    gradients = [torch.autograd.grad(value.square().sum(), (tokens, coords), retain_graph=True)
                 for value in (grid, gather)]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)


def test_legacy_half_entropy_and_variance_accumulate_in_float32():
    entropy = normalized_entropy(torch.tensor([[1., 0.], [.5, .5]], dtype=torch.float16))
    torch.testing.assert_close(entropy, torch.tensor([0., 1.]))
    update = TrajectoryStateUpdate(8)
    ref = base_grid(4, 4).unsqueeze(2)
    residual = torch.tensor([[-1e-4, 0.], [1e-4, 0.]]).expand(1, 16, 2, 2).half()
    aux = {'coords': ref+residual.float(), 'reference': ref.squeeze(2),
           'weights': torch.full((1, 16, 2), .5).half(), 'residual': residual}
    _, stats, _ = update(torch.zeros(1, 16, 8), torch.zeros(1, 16, 2), torch.zeros(1, 16, 4), aux)
    assert stats[..., 1].min() > 9e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_cuda_amp_optimizer_step():
    model = DSORNetV32Distribution().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        logits, aux = model(torch.randn(4, 3, 32, 32, device='cuda'), True)
        loss = F.cross_entropy(logits, torch.arange(4, device='cuda')) + .01*trajectory_energy(aux)
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_mean_ablation_has_identical_initial_weights_and_discards_covariance():
    torch.manual_seed(91)
    full = DSORNetV32Distribution()
    torch.manual_seed(91)
    mean_only = DSORNetV32Distribution(memory_mode='mean')
    assert full.state_dict().keys() == mean_only.state_dict().keys()
    assert all(torch.equal(value, mean_only.state_dict()[key]) for key, value in full.state_dict().items())
    _, aux = mean_only(torch.randn(2, 3, 32, 32), True)
    for state in (aux['state1'], aux['state2']):
        assert state.positions.shape[-2] == 1
        assert torch.equal(distribution_moments(state)[1], torch.zeros(2, 16, 2, 2))
