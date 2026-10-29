# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
The Pavan Krovvidi FingerWave Neural Network.

    input ─► U-Net encoder ─► seed s, confidence c
          ─► coarse Wavefront PDE (H/4, 40 steps, long-range fill)
          ─► upsample, blend with seed
          ─► fine Wavefront PDE (H, 20 steps, local detail) ─► Ψ = A·e^{i·2θ}

Inspired by fingerprint development: ridges start at a few reliable initiation sites
and travel inward as waves; cores and deltas form where the waves meet.
"""

from importlib import resources

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pde import WavefrontPDELayer

__all__ = ["UNet", "PavanKrovvidiFingerWaveNetwork"]

PRETRAINED = "fingerwave_synthetic_v1.pt"


def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GELU(), nn.Conv2d(o, o, 3, padding=1), nn.GELU())


class UNet(nn.Module):
    """Small three-level U-Net (used as the FingerWave encoder and as a baseline)."""

    def __init__(self, in_channels=4, out_channels=2, width=16):
        super().__init__()
        w = width
        self.e1, self.e2, self.e3 = _block(in_channels, w), _block(w, 2 * w), _block(2 * w, 4 * w)
        self.d2, self.d1 = _block(6 * w, 2 * w), _block(3 * w, w)
        self.out = nn.Conv2d(w, out_channels, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2))
        c = self.e3(F.avg_pool2d(b, 2))
        b = self.d2(torch.cat([F.interpolate(c, scale_factor=2, mode="bilinear"), b], 1))
        a = self.d1(torch.cat([F.interpolate(b, scale_factor=2, mode="bilinear"), a], 1))
        return self.out(a)


class PavanKrovvidiFingerWaveNetwork(nn.Module):
    """
    Pavan Krovvidi FingerWave Neural Network.

    Args:
        in_channels: input channels (default 4: image, u, v, coherence).
        width: encoder width.
        coarse_steps, fine_steps: PDE steps at H/4 and H resolution.
        seeded: use boundary seeding (confidence forcing). False = ablation.
        isotropic: tie d⊥ to d∥. True = ablation.

    Input:  (B, in_channels, H, W) with H, W divisible by 4.
    Output: (B, 2, H, W) doubled-angle orientation field; θ = ½·atan2(out[:,1], out[:,0]).
    """

    def __init__(self, in_channels=4, width=16, coarse_steps=40, fine_steps=20,
                 seeded=True, isotropic=False):
        super().__init__()
        self.encoder = UNet(in_channels, 3, width)
        self.coarse = WavefrontPDELayer(steps=coarse_steps, d_par=0.9, d_perp=0.2, isotropic=isotropic)
        self.fine = WavefrontPDELayer(steps=fine_steps, d_par=0.9, d_perp=0.2, isotropic=isotropic)
        self.seeded = seeded
        self.raw_lam = nn.Parameter(torch.zeros(2))   # λ = 4·sigmoid(raw) -> λ·Δt ≤ 0.8

    def forward(self, x, return_confidence=False):
        e = self.encoder(x)
        seed, c = e[:, :2], torch.sigmoid(e[:, 2:3])
        lam = 4 * torch.sigmoid(self.raw_lam)
        sc, cc = F.avg_pool2d(seed, 4), F.avg_pool2d(c, 4)
        if self.seeded:
            psi = self.coarse(sc, seed=sc, confidence=cc, lam=lam[0])
            psi = F.interpolate(psi, scale_factor=4, mode="bilinear")
            psi = c * seed + (1 - c) * psi
            psi = self.fine(psi, seed=seed, confidence=c, lam=lam[1])
        else:
            psi = self.coarse(sc)
            psi = self.fine(F.interpolate(psi, scale_factor=4, mode="bilinear"))
        return (psi, c) if return_confidence else psi

    @classmethod
    def from_pretrained(cls, map_location="cpu"):
        """Load weights trained on the synthetic fingerprint benchmark (64×64 fields, 4 inputs)."""
        model = cls()
        ref = resources.files("pavan_krovvidi_fingerwave").joinpath("weights", PRETRAINED)
        with resources.as_file(ref) as path:
            state = torch.load(path, map_location=map_location, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
