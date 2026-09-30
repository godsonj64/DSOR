"""Experimental DSOR v3.2: cross-scale sampling-distribution memory.

Fine routing particles and their probability mass are transported into the
coarse coordinate frame. Each coarse router reads several kernel-conditioned
centres from that distribution, and a scalar mixture gate retains earlier
particles alongside new realized samples. This is a testable architectural
hypothesis, not a claim of established novelty or superior accuracy.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as F

try:
    from .dsorn_v31 import (BackboneBase, FFBlock, InheritedRouter,
                           base_grid, bilinear_sample_tokens, group_2x2)
except ImportError:
    from dsorn_v31 import (BackboneBase, FFBlock, InheritedRouter,
                          base_grid, bilinear_sample_tokens, group_2x2)


class TrajectoryDistribution(NamedTuple):
    positions: torch.Tensor  # [B, N, K, 2], absolute coarse-frame coordinates
    weights: torch.Tensor    # [B, N, K], normalized nonnegative probability mass


def geometry_dtype(tensor):
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


def distribution_moments(state):
    """Probability mean and covariance, accumulated in at least float32.

    The centered form is positive semidefinite by construction and avoids the
    cancellation in E[xx^T] - E[x]E[x]^T. No matrix square root/eigendecomposition
    is needed, so coincident particles have finite analytical gradients.
    """
    dtype = geometry_dtype(state.positions)
    positions, weights = state.positions.to(dtype), state.weights.to(dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(dtype).tiny)
    mean = (weights.unsqueeze(-1) * positions).sum(-2)
    delta = positions - mean.unsqueeze(-2)
    covariance = (weights[..., None, None] * delta.unsqueeze(-1) * delta.unsqueeze(-2)).sum(-3)
    return mean, covariance


def distribution_features(state, reference, grid_size=4):
    """Six dimensionless geometry features; no entropy-as-uncertainty claim."""
    mean, covariance = distribution_moments(state)
    reference = reference.to(mean)
    step = 2.0 / (grid_size - 1)
    variance_scale = step * step
    weights = state.weights.to(mean.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(mean.dtype).tiny)
    excess = state.positions.to(mean.dtype) - state.positions.to(mean.dtype).clamp(-1, 1)
    boundary_excess = (weights * excess.square().sum(-1)).sum(-1) / variance_scale
    return torch.cat([
        (mean - reference) / step,
        torch.log1p(covariance[..., 0, 0].clamp_min(0) / variance_scale).unsqueeze(-1),
        torch.log1p(covariance[..., 1, 1].clamp_min(0) / variance_scale).unsqueeze(-1),
        (covariance[..., 0, 1] / variance_scale).unsqueeze(-1),
        torch.log1p(boundary_excess).unsqueeze(-1),
    ], -1)


def transport_fine_distribution(aux):
    """Exact 8-to-4 coordinate pushforward of the 4-query routing mixture.

    Particle positions remain unclipped during frame conversion. Clipping them
    here would shift boundary group means even under zero fine-stage motion.
    Actual coarse sampling is clamped later, at the sampler interface.
    """
    coords = aux['coords'].to(geometry_dtype(aux['coords']))
    weights = aux['weights'].to(coords.dtype)
    if coords.shape[1] != 64 or coords.shape[:-1] != weights.shape:
        raise ValueError('Expected fine coordinates [B,64,M,2] and weights [B,64,M]')
    positions = group_2x2(coords, 8, 8).flatten(2, 3) * (7.0 / 6.0)
    mass = group_2x2(weights, 8, 8).flatten(2, 3) * 0.25
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
    return TrajectoryDistribution(positions, mass)


def mixture_update(old, realized_positions, realized_weights, eta):
    """Exact probability mixture with scalar eta for each token.

    Its mean is a vector convex combination. Its covariance includes the
    between-component mean term; neither old modes nor their mass are dropped.
    Particle count grows by the router sample budget per depth transition.
    """
    positions = realized_positions.to(old.positions.dtype)
    weights = realized_weights.to(old.weights.dtype)
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    eta = eta.to(weights.dtype)
    mass = torch.cat([(1-eta) * old.weights, eta * weights], -1)
    mass = mass / mass.sum(-1, keepdim=True).clamp_min(torch.finfo(mass.dtype).tiny)
    return TrajectoryDistribution(torch.cat([old.positions, positions], -2), mass)


class DistributionRouter(nn.Module):
    """Read a carried spatial distribution with five distinct query anchors."""
    def __init__(self, dim=48):
        super().__init__()
        # Reuse the canonical projection/MLP layout and parameter budget.
        template = InheritedRouter(dim, samples=5, residual_radius=0.42)
        self.dim, self.samples = dim, 5
        self.h = self.w = 4
        self.residual_radius = 0.42
        for name in ('norm', 'coord', 'inherit_embed', 'residual', 'gate', 'score', 'q', 'k', 'v', 'proj'):
            setattr(self, name, getattr(template, name))
        self.register_buffer('ref', base_grid(4, 4), persistent=False)
        self.register_buffer('anchors', torch.tensor([[0.,0.],[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]])/6,
                             persistent=False)
        self._folded = False
        with torch.no_grad():
            self.gate[-1].bias[0] = math.log(0.75 / 0.25)
            self.gate[-1].bias[1] = math.log((0.15-0.05)/(0.60-0.15))

    def forward(self, z, state, return_aux=False):
        batch, tokens, dim = z.shape
        zn = self.norm(z)
        reference = self.ref.to(z).expand(batch, -1, -1)
        inherited = self.inherit_embed(distribution_features(state, reference).to(z.dtype))
        global_context = zn.mean(1, keepdim=True).expand(-1, tokens, -1)
        positional = self.coord(self.ref.to(z)).expand(batch, -1, -1)
        ctx = torch.cat([zn, global_context, positional, inherited], -1)
        controls = self.gate(ctx)
        gate = controls[..., :1].sigmoid()
        bandwidth = 0.05 + 0.55 * controls[..., 1:].sigmoid()
        residual = self.residual_radius * self.residual(ctx).reshape(batch,tokens,5,2).tanh()
        dtype = geometry_dtype(state.positions)
        anchors = reference.to(dtype).unsqueeze(-2) + self.anchors.to(dtype)
        queries = anchors + residual.to(dtype)
        particles = state.positions.to(dtype)
        distance = (queries.unsqueeze(-2)-particles.unsqueeze(-3)).square().sum(-1)
        mass = state.weights.to(dtype)
        safe_mass = mass.clamp_min(torch.finfo(dtype).tiny)
        log_mass = safe_mass.log().masked_fill(mass == 0, -torch.inf)
        log_kernel = -distance / (2*bandwidth.to(dtype).square().unsqueeze(-1))
        responsibilities = (log_kernel + log_mass.unsqueeze(-2)).softmax(-1)
        centres = (responsibilities.unsqueeze(-1)*particles.unsqueeze(-3)).sum(-2)
        proposed = (1-gate.to(dtype).unsqueeze(-1))*anchors + gate.to(dtype).unsqueeze(-1)*centres + residual.to(dtype)
        coords = proposed.clamp(-1,1)
        sampled = bilinear_sample_tokens(zn, coords.to(zn.dtype), 4, 4)
        if self._folded:
            query = F.linear(zn, self.qk_weight)
        else:
            query = F.linear(self.q(zn), self.k.weight.t())
        logits = (query.unsqueeze(2)*sampled).sum(-1)/math.sqrt(dim)+self.score(ctx)
        weights = logits.softmax(-1)
        pooled = (weights.unsqueeze(-1)*sampled).sum(2)
        routed = F.linear(pooled,self.vp_weight) if self._folded else self.proj(self.v(pooled))
        out = z + routed
        if return_aux:
            return out, {'coords':coords,'weights':weights,'residual':residual,'reference':reference,
                         'gate':gate,'bandwidth':bandwidth,'responsibilities':responsibilities,
                         'prototype_centres':centres,'proposed_coords':proposed}
        return out

    def fold_for_inference(self, position_cache):
        if self._folded:
            return self
        self.register_buffer('qk_weight', (self.k.weight.T @ self.q.weight).detach().clone())
        self.register_buffer('vp_weight', (self.proj.weight @ self.v.weight).detach().clone())
        self.coord = position_cache(self.coord(self.ref))
        del self.q, self.k, self.v, self.proj
        self._folded = True
        return self


class DistributionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.router = DistributionRouter(dim)
        self.ff = FFBlock(dim)

    def forward(self, z, state, return_aux=False):
        if return_aux:
            out, aux = self.router(z,state,True)
            return self.ff(out), aux
        return self.ff(self.router(z,state))


class DistributionStateUpdate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim+12,dim//2),nn.GELU(),nn.Linear(dim//2,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, z, old, aux):
        realized = TrajectoryDistribution(aux['coords'].to(old.positions.dtype), aux['weights'].to(old.weights.dtype))
        inp = torch.cat([self.norm(z),distribution_features(old,aux['reference']).to(z.dtype),
                         distribution_features(realized,aux['reference']).to(z.dtype)],-1)
        eta = self.gate(inp).sigmoid()
        return mixture_update(old,realized.positions,realized.weights,eta), eta


class DSORNetV32Distribution(BackboneBase):
    """Same-width classifier with explicit sampling-distribution transport."""
    def __init__(self,num_classes=10,d1=32,d2=48,d3=64):
        super().__init__(num_classes,d1,d2,d3)
        # Distinct initial fine sampling points within the existing L-infinity bound.
        pattern = torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]]) * 0.08
        with torch.no_grad():
            self.stage1.router.offset[-1].bias.copy_(torch.atanh(pattern/0.38).flatten())
        self.stage2a = DistributionBlock(d2)
        self.stage2b = DistributionBlock(d2)
        self.state_update = DistributionStateUpdate(d2)

    def forward(self,x,return_aux=False):
        fine, a1 = self.stage1(self.stem(x),True)
        state1 = transport_fine_distribution(a1)
        coarse, a2 = self.stage2a(self.coarse(fine),state1,True)
        state2, eta = self.state_update(coarse,state1,a2)
        if return_aux:
            coarse,a3 = self.stage2b(coarse,state2,True)
        else:
            coarse = self.stage2b(coarse,state2)
        logits = self.finish(coarse)
        if return_aux:
            return logits, {'stage1':a1,'stage2':[a2,a3],'state1':state1,'state2':state2,'eta':eta}
        return logits


def trajectory_energy(aux):
    """Penalize sampled residual energy, including opposite residual directions."""
    return torch.stack([(a['weights'].float()*a['residual'].float().square().sum(-1)).sum(-1).mean()
                        for a in aux['stage2']]).mean()


__all__ = ['DSORNetV32Distribution','TrajectoryDistribution','distribution_moments',
           'distribution_features','transport_fine_distribution','mixture_update','trajectory_energy']
