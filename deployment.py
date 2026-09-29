"""Explicit, frozen inference preparation for DSORNet and its subclasses.

``prepared = prepare_for_inference(trained_model)`` copies the model, folds the
bias-free query/key and value/output projections, and evaluates its fixed-grid
position MLPs once. The source model, its parameters, and its canonical
``state_dict`` remain untouched. Save that canonical state for further training;
the prepared copy has a deliberately different deployment-only state schema.

There is no low-rank approximation, quantization, or reduction in routing samples.
Matrix reassociation has the usual floating-point rounding differences. Prepare
after selecting the final device and dtype for the closest numerical agreement.
The copy owns its weights, so later source training cannot leave stale caches.
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dsorn_v31 import (
        BackboneBase, IndependentRouter, InheritedRouter, bilinear_sample_tokens,
    )
except ImportError:
    from dsorn_v31 import (
        BackboneBase, IndependentRouter, InheritedRouter, bilinear_sample_tokens,
    )


class CachedPosition(nn.Module):
    """A fixed-grid MLP's precomputed output, broadcast across the input batch."""

    def __init__(self, values):
        super().__init__()
        self.register_buffer("values", values.detach().clone())

    def forward(self, coords):
        return self.values.to(coords).expand(coords.shape[0], -1, -1)


class _FrozenRouter(nn.Module):
    def __init__(self, router):
        super().__init__()
        self.dim = router.dim
        self.h, self.w = router.h, router.w
        self.samples = router.samples
        self.norm = router.norm
        self.score = router.score
        self.register_buffer("ref", router.ref.detach().clone(), persistent=False)
        self.coord = CachedPosition(router.coord(router.ref))

        layers = (router.q, router.k, router.v, router.proj)
        if any(layer.bias is not None for layer in layers):
            raise ValueError("frozen routing currently requires bias-free q/k/v/proj")
        # q=(z Wq^T), so q Wk = z (Wk^T Wq)^T.
        self.register_buffer("qk_weight", (router.k.weight.T @ router.q.weight).detach().clone())
        # P(V(s)) = (s Wv^T) Wp^T = s (Wp Wv)^T.
        self.register_buffer("vp_weight", (router.proj.weight @ router.v.weight).detach().clone())

    def _context(self, z):
        batch, tokens, _ = z.shape
        zn = self.norm(z)
        p = self.ref.to(z).expand(batch, -1, -1)
        global_context = zn.mean(1, keepdim=True).expand(-1, tokens, -1)
        positional = self.coord(self.ref.to(z)).expand(batch, -1, -1)
        return zn, p, global_context, positional

    def _route(self, z, zn, coords, ctx):
        sampled = bilinear_sample_tokens(zn, coords, self.h, self.w)
        query = F.linear(zn, self.qk_weight)
        logits = (query.unsqueeze(2) * sampled).sum(-1) / math.sqrt(self.dim)
        weights = (logits + self.score(ctx)).softmax(-1)
        pooled = (weights.unsqueeze(-1) * sampled).sum(2)
        return z + F.linear(pooled, self.vp_weight), weights


class FrozenIndependentRouter(_FrozenRouter):
    def __init__(self, router):
        super().__init__(router)
        self.max_offset = router.max_offset
        self.offset = router.offset

    def forward(self, z, return_aux=False):
        batch, tokens, _ = z.shape
        zn, p, global_context, positional = self._context(z)
        ctx = torch.cat([zn, global_context, positional], -1)
        residual = self.max_offset * torch.tanh(
            self.offset(ctx).reshape(batch, tokens, self.samples, 2)
        )
        coords = (p.unsqueeze(2) + residual).clamp(-1, 1)
        out, weights = self._route(z, zn, coords, ctx)
        if return_aux:
            return out, {
                "offsets": residual,
                "residual": residual,
                "reference": p,
                "coords": coords,
                "weights": weights,
            }
        return out


class FrozenInheritedRouter(_FrozenRouter):
    def __init__(self, router):
        super().__init__(router)
        self.residual_radius = router.residual_radius
        self.inherit_embed = router.inherit_embed
        self.residual = router.residual
        self.gate = router.gate

    def forward(self, z, prior, stats, return_aux=False):
        batch, tokens, _ = z.shape
        zn, p, global_context, positional = self._context(z)
        inherit = self.inherit_embed(torch.cat([prior, stats], -1))
        ctx = torch.cat([zn, global_context, positional, inherit], -1)
        gate = torch.sigmoid(self.gate(ctx))
        residual = self.residual_radius * torch.tanh(
            self.residual(ctx).reshape(batch, tokens, self.samples, 2)
        )
        inherited = gate.unsqueeze(2) * prior.unsqueeze(2)
        total_offset = inherited + residual
        coords = (p.unsqueeze(2) + total_offset).clamp(-1, 1)
        out, weights = self._route(z, zn, coords, ctx)
        if return_aux:
            return out, {
                "offsets": total_offset,
                "residual": residual,
                "inherited": inherited,
                "prior": prior,
                "gate": gate,
                "reference": p,
                "coords": coords,
                "weights": weights,
                "stats": stats,
            }
        return out


class PreparedInference(nn.Module):
    """An independently owned inference snapshot; training is intentionally barred.

    Additional subclass heads and restoration modules stay in ``model`` and are
    deep-copied unchanged. Call the wrapper as the original model, including
    ``return_aux=True``. Its forward always runs with gradients disabled.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.requires_grad_(False)
        self.train(False)

    def train(self, mode=True):
        if mode:
            raise RuntimeError(
                "PreparedInference is frozen; train the canonical source model "
                "and call prepare_for_inference again."
            )
        return super().train(False)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@torch.no_grad()
def prepare_for_inference(model):
    """Return a separate frozen model with exact algebraic inference folding.

    The supplied model's training flag, parameter identities, and state are not
    modified. DSORNet backbone subclasses retain their own forward and heads.
    Fixed tensors use buffers, so ``.to(device)`` and ``.to(dtype)`` still work.
    Preparing directly at the final dtype avoids casting a previously rounded
    fused matrix if strict equivalence tolerances are important.
    """
    if isinstance(model, PreparedInference):
        raise TypeError("model is already prepared for inference")
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")

    frozen = copy.deepcopy(model).eval().requires_grad_(False)
    backbone_count = 0
    for module in list(frozen.modules()):
        if isinstance(module, BackboneBase):
            backbone_count += 1
            module.pos1 = CachedPosition(module.pos1(module.grid1))
            module.pos2 = CachedPosition(module.pos2(module.grid2))
        for name, child in list(module.named_children()):
            if isinstance(child, IndependentRouter):
                setattr(module, name, FrozenIndependentRouter(child))
            elif isinstance(child, InheritedRouter):
                setattr(module, name, FrozenInheritedRouter(child))

    if not backbone_count:
        raise TypeError("model must contain a DSORNet BackboneBase")
    return PreparedInference(frozen)


__all__ = ["CachedPosition", "PreparedInference", "prepare_for_inference"]
