"""Shared DSORNet classification and residual RGB image reconstruction.

Inputs are floating-point RGB batches in [0, 1], shaped [B, 3, 32, 32].
Restoration is initialized to the identity. The returned image is deliberately
unclipped so training retains gradients; callers may clamp it for display.
"""
from __future__ import annotations

import torch
from torch import nn

from dsorn_v31 import DSORNetV31Sequential, aggregate_stage1_trajectory


def unpatchify_4x4(tokens):
    """Invert dsorn_v31.patchify_4x4 without changing RGB or pixel order."""
    if tokens.ndim != 3 or tuple(tokens.shape[1:]) != (64, 48):
        raise ValueError('Expected patch tokens shaped [B, 64, 48]')
    batch = tokens.shape[0]
    return (tokens.reshape(batch, 8, 8, 3, 4, 4)
            .permute(0, 3, 1, 4, 2, 5).reshape(batch, 3, 32, 32))


class NanoImagingModel(nn.Module):
    """One shared encoder with classification and a small residual RGB head."""
    def __init__(self, num_classes=10, d1=32, d2=48, d3=64):
        super().__init__()
        self.backbone = DSORNetV31Sequential(num_classes, d1, d2, d3)
        self.register_buffer('mean', torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1))
        self.coarse_project = nn.Linear(d2, d1)
        self.restore_norm = nn.LayerNorm(d1)
        self.restore_head = nn.Sequential(nn.Linear(d1, 2*d1), nn.GELU(), nn.Linear(2*d1, 48))
        nn.init.zeros_(self.restore_head[-1].weight)
        nn.init.zeros_(self.restore_head[-1].bias)
        self._backbone_frozen = False

    def freeze_backbone(self, frozen=True):
        """Keep classifier weights and dropout fixed while fitting the RGB head."""
        self._backbone_frozen = bool(frozen)
        self.backbone.requires_grad_(not self._backbone_frozen)
        self.backbone.train(False if self._backbone_frozen else self.training)
        return self

    def train(self, mode=True):
        super().train(mode)
        if self._backbone_frozen:
            self.backbone.eval()
        return self

    def _encode(self, normalized, return_aux=False):
        backbone = self.backbone
        fine, a1 = backbone.stage1(backbone.stem(normalized), True)
        prior1, stats1 = aggregate_stage1_trajectory(a1, backbone.correct_geometry)
        coarse = backbone.coarse(fine)
        coarse, a2 = backbone.stage2a(coarse, prior1, stats1, True)
        prior2, stats2, update = backbone.state_update(coarse, prior1, stats1, a2)
        if return_aux:
            coarse, a3 = backbone.stage2b(coarse, prior2, stats2, True)
            aux = {'stage1': a1, 'prior1': prior1, 'stats1': stats1,
                   'stage2': [a2, a3], 'update': update, 'prior2': prior2, 'stats2': stats2}
        else:
            coarse = backbone.stage2b(coarse, prior2, stats2)
            aux = None
        return fine, coarse, aux

    def forward(self, raw_rgb, task='both', return_aux=False):
        if task not in ('both', 'classify', 'restore'):
            raise ValueError("task must be 'both', 'classify', or 'restore'")
        if raw_rgb.ndim != 4 or tuple(raw_rgb.shape[1:]) != (3, 32, 32):
            raise ValueError('Expected RGB images shaped [B, 3, 32, 32]')
        if not raw_rgb.is_floating_point():
            raise TypeError('Input must be floating-point RGB in [0, 1]')
        normalized = (raw_rgb - self.mean) / self.std
        if task == 'classify':
            return self.backbone(normalized, return_aux=return_aux)
        fine, coarse, aux = self._encode(normalized, return_aux)
        batch, _, dim = fine.shape
        context = self.coarse_project(coarse).reshape(batch, 4, 4, dim)
        context = context.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).reshape(batch, 64, dim)
        residual = unpatchify_4x4(self.restore_head(self.restore_norm(fine + context)))
        restored = raw_rgb + residual
        result = {'logits': self.backbone.finish(coarse), 'restored': restored} if task == 'both' else restored
        if return_aux:
            return result, aux
        return result
