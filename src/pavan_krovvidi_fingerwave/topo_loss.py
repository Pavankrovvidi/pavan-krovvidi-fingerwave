# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
Differentiable topological loss for director (orientation) fields.

A ridge orientation θ has no head or tail (θ ≡ θ + π). FingerWave stores it as the
doubled-angle field  w = u + i·v = A·e^{i·2θ}  with two real channels, so fingerprint
cores (+½) and deltas (−½) become ordinary ±1 vortices of w.

Two training terms, both smooth:
  * Ginzburg–Landau energy  |∇w|² + (1 − |w|²)² / (2ε²)   — every defect costs energy.
  * Soft topological charge  ρ = (∂x u ∂y v − ∂y u ∂x v) / π — the net index can be
    constrained to a target (0 for arches, loops and whorls).

`count_defects` is an exact, integer-valued counter. It has zero gradient and is meant
for evaluation only.
"""

import math

import torch
import torch.nn.functional as F

TWO_PI = 2 * math.pi

__all__ = [
    "wrap_angle", "central_grad", "plaquette_winding", "ginzburg_landau_energy",
    "soft_charge_density", "topological_loss", "count_defects",
]


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (−π, π]."""
    return torch.remainder(a + math.pi, TWO_PI) - math.pi


def central_grad(f: torch.Tensor):
    """Central differences of f (B, H, W) with replicate padding. Returns (∂x f, ∂y f)."""
    fp = F.pad(f.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    dx = (fp[:, 1:-1, 2:] - fp[:, 1:-1, :-2]) / 2.0
    dy = (fp[:, 2:, 1:-1] - fp[:, :-2, 1:-1]) / 2.0
    return dx, dy


def plaquette_winding(phi: torch.Tensor) -> torch.Tensor:
    """Winding number of angle field phi (B, H, W) around every 2×2 plaquette -> (B, H-1, W-1)."""
    a, b = phi[:, :-1, :-1], phi[:, :-1, 1:]
    c, d = phi[:, 1:, 1:], phi[:, 1:, :-1]
    total = wrap_angle(b - a) + wrap_angle(c - b) + wrap_angle(d - c) + wrap_angle(a - d)
    return total / TWO_PI


def ginzburg_landau_energy(field: torch.Tensor, eps: float = 1.5) -> torch.Tensor:
    """Ginzburg–Landau energy of a doubled-angle field (B, 2, H, W). eps = core size in pixels."""
    u, v = field[:, 0], field[:, 1]
    ux, uy = central_grad(u)
    vx, vy = central_grad(v)
    smooth = (ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2).mean()
    amp = ((1.0 - (u ** 2 + v ** 2)) ** 2).mean() / (2 * eps ** 2)
    return smooth + amp


def soft_charge_density(field: torch.Tensor) -> torch.Tensor:
    """Differentiable vortex-charge density ρ of w (B, H, W). Director index density = ρ / 2."""
    u, v = field[:, 0], field[:, 1]
    ux, uy = central_grad(u)
    vx, vy = central_grad(v)
    return (ux * vy - uy * vx) / math.pi


def topological_loss(field: torch.Tensor, target_index: float = 0.0, eps: float = 1.5,
                     w_gl: float = 1.0, w_charge: float = 0.1) -> torch.Tensor:
    """
    FingerWave topological loss = w_gl · GL energy + w_charge · (net director index − target)².

    Args:
        field: (B, 2, H, W) doubled-angle orientation field.
        target_index: desired net index (0 for arch, loop and whorl fingerprints).
    """
    gl = ginzburg_landau_energy(field, eps)
    net_index = soft_charge_density(field).sum(dim=(1, 2)) / 2.0
    charge = (net_index - target_index).pow(2).mean()
    return w_gl * gl + w_charge * charge


@torch.no_grad()
def count_defects(field: torch.Tensor):
    """
    Exact singular-point counter (evaluation only).

    Returns:
        stats: dict with "plus_half" (cores), "minus_half" (deltas) and "net_index".
        index: (B, H-1, W-1) map with values in {−½, 0, +½, ...} at plaquette centres.
    """
    phi2 = torch.atan2(field[:, 1], field[:, 0])
    index = torch.round(plaquette_winding(phi2)) / 2.0
    stats = {
        "plus_half": int((index > 0).sum()),
        "minus_half": int((index < 0).sum()),
        "net_index": float(index.sum()),
    }
    return stats, index
