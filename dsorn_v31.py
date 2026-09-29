
"""
DSORNet-v3.1
Hierarchical convolution-free image network with Cross-Scale Trajectory State Routing (CSTR).

Key idea:
  1) Stage 1 predicts dynamic fractional sample coordinates.
  2) Attention-weighted expected ABSOLUTE sample positions are aggregated across 2x2 fine cells.
  3) Those trajectories are transported into the coarse stage's local coordinate system.
  4) Stage 2 uses a gated inherited prior + a learned residual deformation.
  5) A recurrent coordinate state update propagates the realized stage-2 trajectory to the next block.

No convolution layers are used.
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20260929


# ------------------------------------------------------------
# Geometry
# ------------------------------------------------------------

def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def base_grid(h, w, device=None, dtype=None):
    if h < 1 or w < 1:
        raise ValueError("grid dimensions must be positive")
    ys = (torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
          if h > 1 else torch.zeros(1, device=device, dtype=dtype))
    xs = (torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
          if w > 1 else torch.zeros(1, device=device, dtype=dtype))
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(1, h*w, 2)


def bilinear_sample_tokens(tokens, coords, h, w, backend="auto"):
    """Border-padded, align_corners=True bilinear interpolation.

    Native gather interpolation supplies MPS coordinate/token backward where
    grid_sample backward is unavailable. Both paths implement the same map.
    """
    B, N, D = tokens.shape
    if N != h*w or coords.ndim != 4 or coords.shape[:2] != (B, N) or coords.shape[-1] != 2:
        raise ValueError("incompatible token/grid shapes")
    if backend not in ("auto", "grid", "gather"):
        raise ValueError("backend must be auto, grid, or gather")
    M = coords.shape[2]
    if backend == "gather" or (backend == "auto" and tokens.device.type == "mps"):
        # grid_sample's border clamp has zero coordinate derivative AT the
        # boundary as well as outside it; plain torch.clamp differs at equality.
        bounded = torch.where(
            (coords > -1) & (coords < 1), coords, coords.clamp(-1, 1).detach()
        )
        x = (bounded[..., 0] + 1) * ((w - 1) / 2)
        y = (bounded[..., 1] + 1) * ((h - 1) / 2)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0 + 1).clamp_max(w - 1), (y0 + 1).clamp_max(h - 1)
        dx, dy = (x - x0.to(x.dtype)).unsqueeze(-1), (y - y0.to(y.dtype)).unsqueeze(-1)

        def corner(ix, iy):
            index = (iy * w + ix).reshape(B, N*M, 1).expand(-1, -1, D)
            return tokens.gather(1, index).reshape(B, N, M, D)

        return (corner(x0, y0) * ((1 - dx) * (1 - dy))
                + corner(x1, y0) * (dx * (1 - dy))
                + corner(x0, y1) * ((1 - dx) * dy)
                + corner(x1, y1) * (dx * dy))
    field = tokens.transpose(1, 2).reshape(B, D, h, w)
    grid = coords.reshape(B, N*M, 1, 2)
    y = F.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return y.squeeze(-1).transpose(1, 2).reshape(B, N, M, D)


def patchify_4x4(x):
    if x.ndim != 4 or x.shape[1:] != (3, 32, 32):
        raise ValueError("expected RGB input of shape (batch, 3, 32, 32)")
    B, C, H, W = x.shape
    assert (H, W) == (32, 32)
    return (
        x.reshape(B, C, 8, 4, 8, 4)
         .permute(0, 2, 4, 1, 3, 5)
         .reshape(B, 64, 48)
    )


def merge_2x2_tokens(z, h, w):
    B, N, D = z.shape
    assert N == h*w
    x = z.reshape(B, h, w, D)
    x = x.reshape(B, h//2, 2, w//2, 2, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(B, (h//2)*(w//2), 4*D)


def group_2x2(v, h, w):
    B, N = v.shape[:2]
    tail = v.shape[2:]
    x = v.reshape(B, h, w, *tail)
    x = x.reshape(B, h//2, 2, w//2, 2, *tail)
    perm = [0, 1, 3, 2, 4] + list(range(5, 5+len(tail)))
    x = x.permute(*perm).contiguous()
    return x.reshape(B, (h//2)*(w//2), 4, *tail)


def fine_abs_to_coarse_local(x_abs, fine_size=8):
    if fine_size < 4 or fine_size % 2:
        raise ValueError("fine_size must be even and at least 4")
    # For 8->4 grouping, the coarse logical centers occupy +/-6/7.
    span = (fine_size - 2) / (fine_size - 1)
    return x_abs / span


# ------------------------------------------------------------
# Cross-scale trajectory aggregation
# ------------------------------------------------------------

def zero_safe_sqrt(value):
    """Exact sqrt for positive variance, finite zero subgradient at collapse.

    Mask BEFORE sqrt: masking sqrt(0) afterwards alone still yields 0 * inf
    during backward. This leaves all positive forward values unchanged.
    """
    positive = value > 0
    safe = torch.where(positive, value, torch.ones_like(value))
    return torch.where(positive, safe.sqrt(), torch.zeros_like(value))


def normalized_entropy(weights):
    """Normalized categorical entropy with finite gradients at zero weights."""
    samples = weights.shape[-1]
    if samples < 1:
        raise ValueError("at least one routing sample is required")
    if samples == 1:
        return weights.sum(-1) * 0.0
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return -(weights * safe.log()).sum(-1) / math.log(samples)


def aggregate_stage1_trajectory(stage1_aux, correct_geometry=True):
    offsets = stage1_aux["offsets"]
    coords = stage1_aux["coords"]
    weights = stage1_aux["weights"]

    B, N, M, _ = offsets.shape
    assert N == 64

    expected_abs = (weights.unsqueeze(-1) * coords).sum(2)
    # The sampler visits clamped coordinates, not the proposed raw offset.
    reference = stage1_aux.get("reference")
    if reference is None:
        reference = base_grid(8, 8, coords.device, coords.dtype)
    effective = coords - reference.unsqueeze(2)
    motion = effective if correct_geometry else offsets
    expected_off = (weights.unsqueeze(-1) * motion).sum(2)

    grouped_abs = group_2x2(expected_abs, 8, 8)
    grouped_off = group_2x2(expected_off, 8, 8)

    coarse_abs_target = grouped_abs.mean(2)
    coarse_local_target = fine_abs_to_coarse_local(coarse_abs_target, 8)

    p2 = base_grid(
        4, 4,
        device=offsets.device,
        dtype=offsets.dtype
    ).expand(B, -1, -1)

    prior = coarse_local_target - p2

    entropy = normalized_entropy(weights)
    grouped_entropy = (
        group_2x2(entropy.unsqueeze(-1), 8, 8)
        .squeeze(-1)
        .mean(2)
    )

    spread = zero_safe_sqrt(
        (grouped_abs - coarse_abs_target.unsqueeze(2)).square()
        .sum(-1)
        .mean(2)
    )
    mean_off_mag = grouped_off.norm(dim=-1).mean(2)
    prior_mag = prior.norm(dim=-1)

    stats = torch.stack(
        [grouped_entropy, spread, mean_off_mag, prior_mag],
        dim=-1
    )
    return prior, stats


# ------------------------------------------------------------
# Dynamic routing primitives
# ------------------------------------------------------------

class IndependentRouter(nn.Module):
    def __init__(self, dim, h, w, samples, max_offset):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.samples = samples
        if samples < 1:
            raise ValueError("samples must be positive")
        self.efficient = True
        self.max_offset = max_offset

        self.norm = nn.LayerNorm(dim)
        self.coord = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.offset = nn.Sequential(
            nn.Linear(dim*3, dim),
            nn.GELU(),
            nn.Linear(dim, samples*2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim*3, max(8, dim//2)),
            nn.GELU(),
            nn.Linear(max(8, dim//2), samples)
        )

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.register_buffer("ref", base_grid(h, w), persistent=False)

        nn.init.normal_(self.offset[-1].weight, 0.0, 0.003)
        nn.init.uniform_(self.offset[-1].bias, -0.03, 0.03)

    def forward(self, z, return_aux=False):
        B, N, D = z.shape
        zn = self.norm(z)
        p = self.ref.to(z).expand(B, -1, -1)
        g = zn.mean(1, keepdim=True).expand(-1, N, -1)
        positional = self.coord(self.ref.to(z)).expand(B, -1, -1)
        ctx = torch.cat([zn, g, positional], -1)

        residual = self.max_offset * torch.tanh(
            self.offset(ctx).reshape(B, N, self.samples, 2)
        )
        coords = (p.unsqueeze(2) + residual).clamp(-1, 1)

        sampled = bilinear_sample_tokens(zn, coords, self.h, self.w)
        if self.efficient:
            # q^T K s = (K^T q)^T s. Project each query, not every sample.
            query = F.linear(self.q(zn), self.k.weight.t())
            logits = (query.unsqueeze(2) * sampled).sum(-1) / math.sqrt(D)
        else:
            logits = (self.q(zn).unsqueeze(2) * self.k(sampled)).sum(-1) / math.sqrt(D)
        logits = logits + self.score(ctx)
        weights = logits.softmax(-1)

        if self.efficient:
            # V sum_m(alpha_m s_m) = sum_m(alpha_m V s_m).
            routed = self.v((weights.unsqueeze(-1) * sampled).sum(2))
        else:
            routed = (weights.unsqueeze(-1) * self.v(sampled)).sum(2)
        out = z + self.proj(routed)

        if return_aux:
            return out, {
                "offsets": residual,
                "residual": residual,
                "coords": coords,
                "reference": p,
                "weights": weights,
            }
        return out


class InheritedRouter(nn.Module):
    def __init__(
        self,
        dim,
        h=4,
        w=4,
        samples=5,
        residual_radius=0.42
    ):
        super().__init__()
        self.dim = dim
        self.h = h
        self.w = w
        self.samples = samples
        if samples < 1:
            raise ValueError("samples must be positive")
        self.efficient = True
        self.residual_radius = residual_radius

        self.norm = nn.LayerNorm(dim)
        self.coord = nn.Sequential(
            nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.inherit_embed = nn.Sequential(
            nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        self.residual = nn.Sequential(
            nn.Linear(dim*4, dim),
            nn.GELU(),
            nn.Linear(dim, samples*2)
        )
        self.gate = nn.Sequential(
            nn.Linear(dim*4, dim//2),
            nn.GELU(),
            nn.Linear(dim//2, 2)
        )
        self.score = nn.Sequential(
            nn.Linear(dim*4, max(8, dim//2)),
            nn.GELU(),
            nn.Linear(max(8, dim//2), samples)
        )

        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

        self.register_buffer("ref", base_grid(h, w), persistent=False)

        nn.init.normal_(self.residual[-1].weight, 0.0, 0.003)
        nn.init.uniform_(self.residual[-1].bias, -0.02, 0.02)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, z, prior, stats, return_aux=False):
        B, N, D = z.shape
        zn = self.norm(z)
        p = self.ref.to(z).expand(B, -1, -1)
        g = zn.mean(1, keepdim=True).expand(-1, N, -1)
        inherit = self.inherit_embed(torch.cat([prior, stats], -1))
        positional = self.coord(self.ref.to(z)).expand(B, -1, -1)
        ctx = torch.cat([zn, g, positional, inherit], -1)

        gate = torch.sigmoid(self.gate(ctx))
        residual = self.residual_radius * torch.tanh(
            self.residual(ctx).reshape(B, N, self.samples, 2)
        )
        inherited = gate.unsqueeze(2) * prior.unsqueeze(2)
        total_offset = inherited + residual
        coords = (p.unsqueeze(2) + total_offset).clamp(-1, 1)

        sampled = bilinear_sample_tokens(zn, coords, self.h, self.w)
        if self.efficient:
            query = F.linear(self.q(zn), self.k.weight.t())
            logits = (query.unsqueeze(2) * sampled).sum(-1) / math.sqrt(D)
        else:
            logits = (self.q(zn).unsqueeze(2) * self.k(sampled)).sum(-1) / math.sqrt(D)
        logits = logits + self.score(ctx)
        weights = logits.softmax(-1)

        if self.efficient:
            routed = self.v((weights.unsqueeze(-1) * sampled).sum(2))
        else:
            routed = (weights.unsqueeze(-1) * self.v(sampled)).sum(2)
        out = z + self.proj(routed)

        if return_aux:
            return out, {
                "offsets": total_offset,
                "residual": residual,
                "inherited": inherited,
                "prior": prior,
                "gate": gate,
                "coords": coords,
                "reference": p,
                "weights": weights,
                "stats": stats,
            }
        return out


class FFBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
        )

    def forward(self, z):
        return z + self.ff(self.norm(z))


class Stage1Block(nn.Module):
    def __init__(self, dim=32, samples=4):
        super().__init__()
        self.router = IndependentRouter(dim, 8, 8, samples, 0.38)
        self.ff = FFBlock(dim)

    def forward(self, z, return_aux=False):
        if return_aux:
            z, aux = self.router(z, True)
            return self.ff(z), aux
        return self.ff(self.router(z))


class Stage2IndependentBlock(nn.Module):
    def __init__(self, dim=48, samples=5):
        super().__init__()
        self.router = IndependentRouter(dim, 4, 4, samples, 0.68)
        self.ff = FFBlock(dim)

    def forward(self, z, return_aux=False):
        if return_aux:
            z, aux = self.router(z, True)
            return self.ff(z), aux
        return self.ff(self.router(z))


class Stage2InheritedBlock(nn.Module):
    def __init__(self, dim=48, samples=5):
        super().__init__()
        self.router = InheritedRouter(dim, 4, 4, samples, 0.42)
        self.ff = FFBlock(dim)

    def forward(self, z, prior, stats, return_aux=False):
        if return_aux:
            z, aux = self.router(z, prior, stats, True)
            return self.ff(z), aux
        return self.ff(self.router(z, prior, stats))


# ------------------------------------------------------------
# Recurrent trajectory state
# ------------------------------------------------------------

class TrajectoryStateUpdate(nn.Module):
    """
    tau_{l+1} = tau_l + eta_l * (mu_l - tau_l)
    """
    def __init__(self, dim):
        super().__init__()
        self.correct_geometry = True
        self.register_buffer("ref", base_grid(4, 4), persistent=False)
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim + 8, dim),
            nn.GELU(),
            nn.Linear(dim, 2)
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, z, old_prior, old_stats, router_aux):
        weights = router_aux["weights"]
        reference = router_aux.get("reference", self.ref.to(z))
        total = (router_aux["coords"] - reference.unsqueeze(2)
                 if self.correct_geometry else router_aux["offsets"])
        residual = router_aux["residual"]

        realized = (weights.unsqueeze(-1) * total).sum(2)
        expected_residual = (
            weights.unsqueeze(-1) * residual
        ).sum(2)

        entropy = normalized_entropy(weights)

        spread = zero_safe_sqrt((
            weights.unsqueeze(-1)
            * (total - realized.unsqueeze(2)).square()
        ).sum((2, 3)))

        resmag = expected_residual.norm(dim=-1)
        realized_mag = realized.norm(dim=-1)
        diagnostics = torch.stack(
            [entropy, spread, resmag, realized_mag],
            dim=-1
        )

        inp = torch.cat([
            self.norm(z),
            old_prior,
            realized,
            diagnostics
        ], -1)

        eta = torch.sigmoid(self.gate(inp))
        next_prior = old_prior + eta * (realized - old_prior)

        next_stats = torch.stack([
            entropy,
            spread,
            realized_mag,
            next_prior.norm(dim=-1)
        ], -1)

        return next_prior, next_stats, {
            "eta": eta,
            "realized": realized,
            "expected_residual": expected_residual,
        }


# ------------------------------------------------------------
# Networks
# ------------------------------------------------------------

class BackboneBase(nn.Module):
    def __init__(self, num_classes, d1=32, d2=48, d3=64):
        super().__init__()

        self.patch_embed = nn.Linear(48, d1)
        self.pos1 = nn.Sequential(
            nn.Linear(2, d1), nn.GELU(), nn.Linear(d1, d1)
        )
        self.register_buffer("grid1", base_grid(8, 8), persistent=False)

        self.stage1 = Stage1Block(d1, 4)
        self.merge1 = nn.Linear(4*d1, d2)

        self.pos2 = nn.Sequential(
            nn.Linear(2, d2), nn.GELU(), nn.Linear(d2, d2)
        )
        self.register_buffer("grid2", base_grid(4, 4), persistent=False)

        self.merge2 = nn.Linear(4*d2, d3)
        self.norm = nn.LayerNorm(d3)
        self.head = nn.Sequential(
            nn.Linear(2*d3, 2*d3),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(2*d3, num_classes)
        )

    def stem(self, x):
        B = x.shape[0]
        z = self.patch_embed(patchify_4x4(x))
        return z + self.pos1(self.grid1.to(z))

    def coarse(self, z):
        B = z.shape[0]
        z = self.merge1(merge_2x2_tokens(z, 8, 8))
        return z + self.pos2(self.grid2.to(z))

    def finish(self, z):
        z = self.merge2(merge_2x2_tokens(z, 4, 4))
        z = self.norm(z)
        pooled = torch.cat([z.mean(1), z.amax(1)], -1)
        return self.head(pooled)


class DSORNetV2Independent(BackboneBase):
    def __init__(self, num_classes=10, d1=32, d2=48, d3=64):
        super().__init__(num_classes, d1, d2, d3)
        self.stage2a = Stage2IndependentBlock(d2, 5)
        self.stage2b = Stage2IndependentBlock(d2, 5)

    def forward(self, x, return_aux=False):
        z = self.stem(x)
        if return_aux:
            z, a1 = self.stage1(z, True)
        else:
            z = self.stage1(z)

        z = self.coarse(z)

        if return_aux:
            z, a2 = self.stage2a(z, True)
            z, a3 = self.stage2b(z, True)
            return self.finish(z), {
                "stage1": a1,
                "stage2": [a2, a3]
            }

        z = self.stage2a(z)
        z = self.stage2b(z)
        return self.finish(z)


class DSORNetV31Sequential(BackboneBase):
    def __init__(self, num_classes=10, d1=32, d2=48, d3=64):
        super().__init__(num_classes, d1, d2, d3)
        self.correct_geometry = True
        self.stage2a = Stage2InheritedBlock(d2, 5)
        self.state_update = TrajectoryStateUpdate(d2)
        self.stage2b = Stage2InheritedBlock(d2, 5)

    def forward(self, x, return_aux=False):
        z = self.stem(x)

        z, a1 = self.stage1(z, True)
        prior1, stats1 = aggregate_stage1_trajectory(a1, self.correct_geometry)

        z = self.coarse(z)

        z, a2 = self.stage2a(z, prior1, stats1, True)
        prior2, stats2, update = self.state_update(
            z, prior1, stats1, a2
        )
        if return_aux:
            z, a3 = self.stage2b(z, prior2, stats2, True)
        else:
            z = self.stage2b(z, prior2, stats2)

        logits = self.finish(z)

        if return_aux:
            return logits, {
                "stage1": a1,
                "prior1": prior1,
                "stats1": stats1,
                "stage2": [a2, a3],
                "update": update,
                "prior2": prior2,
                "stats2": stats2,
            }
        return logits


# ------------------------------------------------------------
# Tests
# ------------------------------------------------------------

def run_unit_tests():
    results = {}

    model = DSORNetV31Sequential(10)
    conv_types = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    )
    results["conv_module_count"] = sum(
        isinstance(m, conv_types) for m in model.modules()
    )
    assert results["conv_module_count"] == 0

    B, M = 2, 4
    p1 = base_grid(8, 8).expand(B, -1, -1)
    coords = p1.unsqueeze(2).expand(B, 64, M, 2).clone()
    offsets = torch.zeros_like(coords)
    weights = torch.rand(B, 64, M)
    weights = weights / weights.sum(-1, keepdim=True)

    prior, _ = aggregate_stage1_trajectory({
        "coords": coords,
        "offsets": offsets,
        "weights": weights
    })

    zero_err = float(prior.abs().max())
    results["zero_motion_transport_error"] = zero_err
    assert zero_err < 2e-6

    d = torch.tensor([0.03, -0.02])
    coords2 = coords + d
    offsets2 = offsets + d
    prior2, _ = aggregate_stage1_trajectory({
        "coords": coords2,
        "offsets": offsets2,
        "weights": weights
    })
    expected = d * (7.0/6.0)
    const_err = float((prior2 - expected).abs().max())
    results["constant_motion_transport_error"] = const_err
    assert const_err < 2e-6

    raw = torch.randn(1, 64, M, 2, requires_grad=True) * 0.01
    raw.retain_grad()
    coords3 = base_grid(8, 8).expand(1, -1, -1).unsqueeze(2) + raw
    w3 = torch.softmax(torch.randn(1, 64, M), -1)
    p3, s3 = aggregate_stage1_trajectory({
        "coords": coords3,
        "offsets": raw,
        "weights": w3
    })
    (p3.square().mean() + s3.mean()).backward()
    grad_norm = float(raw.grad.norm())
    results["transport_gradient_norm"] = grad_norm
    assert grad_norm > 0.0 and math.isfinite(grad_norm)

    x = torch.randn(2, 3, 32, 32, requires_grad=True)
    y, aux = model(x, True)
    assert y.shape == (2, 10)
    y.mean().backward()
    assert torch.isfinite(x.grad).all()
    results["full_forward_backward_finite"] = True

    return results


# ------------------------------------------------------------
# Optional real CIFAR small-batch experiment
# ------------------------------------------------------------

def stratified_indices(targets, per_class, seed):
    if not isinstance(per_class, (int, np.integer)) or per_class < 1:
        raise ValueError("per_class must be a positive integer")
    rng = np.random.default_rng(seed)
    targets = np.asarray(targets)
    idx = []
    for c in np.unique(targets):
        choices = np.flatnonzero(targets == c)
        rng.shuffle(choices)
        idx.extend(choices[:min(per_class, len(choices))].tolist())
    rng.shuffle(idx)
    return idx


def make_cifar(dataset_name, root, train_per_class, test_per_class):
    from torchvision import datasets, transforms

    if dataset_name == "cifar10":
        cls = datasets.CIFAR10
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
    elif dataset_name == "cifar100":
        cls = datasets.CIFAR100
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError(dataset_name)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_full = cls(root=root, train=True, download=True, transform=train_tf)
    test_full = cls(root=root, train=False, download=True, transform=test_tf)

    train_idx = stratified_indices(
        train_full.targets, train_per_class, SEED
    )
    test_idx = stratified_indices(
        test_full.targets, test_per_class, SEED+1
    )

    return (
        torch.utils.data.Subset(train_full, train_idx),
        torch.utils.data.Subset(test_full, test_idx),
        num_classes,
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        correct += int((logits.argmax(1) == y).sum())
        total += len(y)
    if total == 0:
        raise ValueError("cannot evaluate an empty loader")
    return correct / total


def trajectory_regularizer(aux):
    reg = 0.0
    for a in aux["stage2"]:
        expected_res = (
            a["weights"].unsqueeze(-1) * a["residual"]
        ).sum(2)
        reg = reg + expected_res.square().mean()
    return reg / len(aux["stage2"])


def train(model, train_loader, test_loader, device, epochs, lr):
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs
    )

    for epoch in range(1, epochs+1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits, aux = model(x, True)
            loss = (
                F.cross_entropy(logits, y)
                + 0.01 * trajectory_regularizer(aux)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

        sched.step()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d}: test_acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100"],
        default="cifar10"
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--test-per-class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2.8e-3)
    parser.add_argument(
        "--device",
        default=(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    )
    args = parser.parse_args()

    seed_all()
    print(run_unit_tests())

    train_ds, test_ds, num_classes = make_cifar(
        args.dataset,
        args.data_dir,
        args.train_per_class,
        args.test_per_class
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        num_workers=0
    )

    model = DSORNetV31Sequential(num_classes)
    device = torch.device(args.device)

    train(
        model,
        train_loader,
        test_loader,
        device,
        args.epochs,
        args.lr
    )


if __name__ == "__main__":
    main()
