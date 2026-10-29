# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
Wavefront PDE layer: the "travelling ridge waves" of FingerWave.

    ∂Ψ/∂t = ∇·(D∇Ψ) + (α + iβ)Ψ − (1 + iγ)|Ψ|²Ψ

* Ψ is a complex field stored as two real channels (B, 2, H, W).
* D = d⊥·I + (d∥ − d⊥)·c·t tᵀ, t = (cos θ, sin θ): information moves faster along ridges.
* Diffusion uses a conservative flux form (mass is conserved with no-flux boundaries).
* The cubic reaction is integrated semi-implicitly, so large |Ψ| cannot blow up.
* d∥, d⊥ ≤ d_max (sigmoid) and Δt = 0.2 / d_max keep the explicit scheme stable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["diffusion_tensor", "diffusion_tensor_from_psi", "anisotropic_divergence",
           "complex_mul", "WavefrontPDELayer"]


def diffusion_tensor(theta, d_par, d_perp, coherence=None):
    """D components (Dxx, Dxy, Dyy) from an explicit ridge angle theta (B, H, W)."""
    c = torch.ones_like(theta) if coherence is None else coherence
    ct, st = torch.cos(theta), torch.sin(theta)
    k = (d_par - d_perp) * c
    return d_perp + k * ct * ct, k * ct * st, d_perp + k * st * st


def diffusion_tensor_from_psi(psi, d_par, d_perp):
    """
    D directly from Ψ = c·e^{i·2θ} without atan2 (finite gradients where Ψ = 0):
    t tᵀ = ½(I + [[cos2θ, sin2θ], [sin2θ, −cos2θ]]), with c·cos2θ = u and c·sin2θ = v.
    """
    u, v = psi[:, 0], psi[:, 1]
    amp = torch.sqrt(u * u + v * v + 1e-8)
    s = 1.0 / torch.clamp(amp, min=1.0)      # coherence ≤ 1 keeps eigenvalues ≤ d_max
    u, v, c = u * s, v * s, amp * s
    k = 0.5 * (d_par - d_perp)
    return d_perp + k * (c + u), k * v, d_perp + k * (c - u)


def anisotropic_divergence(psi, Dxx, Dxy, Dyy):
    """∇·(D∇ψ) for psi (B, C, H, W) using face fluxes and zero flux at the domain boundary."""
    Dxx, Dxy, Dyy = (t.unsqueeze(1) for t in (Dxx, Dxy, Dyy))
    p = F.pad(psi, (1, 1, 1, 1), mode="replicate")
    gx_c = (p[..., 1:-1, 2:] - p[..., 1:-1, :-2]) / 2
    gy_c = (p[..., 2:, 1:-1] - p[..., :-2, 1:-1]) / 2

    dx_f = psi[..., :, 1:] - psi[..., :, :-1]
    gy_xf = (gy_c[..., :, 1:] + gy_c[..., :, :-1]) / 2
    Jx = 0.5 * (Dxx[..., 1:] + Dxx[..., :-1]) * dx_f + 0.5 * (Dxy[..., 1:] + Dxy[..., :-1]) * gy_xf

    dy_f = psi[..., 1:, :] - psi[..., :-1, :]
    gx_yf = (gx_c[..., 1:, :] + gx_c[..., :-1, :]) / 2
    Jy = 0.5 * (Dxy[..., 1:, :] + Dxy[..., :-1, :]) * gx_yf + 0.5 * (Dyy[..., 1:, :] + Dyy[..., :-1, :]) * dy_f

    Jx = F.pad(Jx, (1, 1, 0, 0))
    Jy = F.pad(Jy, (0, 0, 1, 1))
    return (Jx[..., :, 1:] - Jx[..., :, :-1]) + (Jy[..., 1:, :] - Jy[..., :-1, :])


def complex_mul(a_re, a_im, x):
    """(a_re + i·a_im) × Ψ for Ψ of shape (B, 2, H, W)."""
    xr, xi = x[:, 0:1], x[:, 1:2]
    return torch.cat([a_re * xr - a_im * xi, a_re * xi + a_im * xr], dim=1)


def _inv_sigmoid(v):
    v = min(max(v, 1e-4), 1 - 1e-4)
    return math.log(v / (1 - v))


class WavefrontPDELayer(nn.Module):
    """
    Learnable, stable reaction–diffusion layer acting on a doubled-angle field.

    Args:
        steps: number of explicit time steps per forward pass.
        d_max: upper bound for the diffusion coefficients (sets Δt = 0.2 / d_max).
        d_par, d_perp: initial diffusion along / across ridges.
        alpha, beta, gamma: initial complex Ginzburg–Landau coefficients.
        reaction: include the Ginzburg–Landau reaction term.
        isotropic: tie d⊥ to d∥ (ablation).
    """

    def __init__(self, steps=20, d_max=1.0, d_par=0.9, d_perp=0.2, alpha=1.0, beta=0.0,
                 gamma=0.0, reaction=True, isotropic=False):
        super().__init__()
        self.steps, self.d_max, self.reaction, self.isotropic = steps, d_max, reaction, isotropic
        self.dt = 0.2 / d_max
        self.raw_par = nn.Parameter(torch.tensor(_inv_sigmoid(d_par / d_max)))
        self.raw_perp = nn.Parameter(torch.tensor(_inv_sigmoid(d_perp / d_max)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        self.beta = nn.Parameter(torch.tensor(float(beta)))
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))

    def d_values(self):
        d_par = self.d_max * torch.sigmoid(self.raw_par)
        d_perp = d_par if self.isotropic else self.d_max * torch.sigmoid(self.raw_perp)
        return d_par, d_perp

    def step(self, psi, theta=None):
        """One explicit time step. If theta is None, D is derived from Ψ itself."""
        d_par, d_perp = self.d_values()
        if theta is None:
            Dxx, Dxy, Dyy = diffusion_tensor_from_psi(psi, d_par, d_perp)
        else:
            Dxx, Dxy, Dyy = diffusion_tensor(theta, d_par, d_perp)
        dpsi = anisotropic_divergence(psi, Dxx, Dxy, Dyy)
        if not self.reaction:
            return psi + self.dt * dpsi
        amp2 = psi.pow(2).sum(1, keepdim=True)
        star = psi + self.dt * (dpsi + complex_mul(self.alpha, self.beta, psi))
        rot = -self.gamma * self.dt * amp2
        star = complex_mul(torch.cos(rot), torch.sin(rot), star)
        return star / (1.0 + self.dt * amp2)

    def forward(self, psi, seed=None, confidence=None, lam=None, theta=None, steps=None):
        """
        Run the PDE. With seed and confidence, confident pixels are pulled towards the seed:
            Ψ ← Ψ + Δt · λ · c · (seed − Ψ)          (boundary seeding)
        """
        for _ in range(steps or self.steps):
            psi = self.step(psi, theta)
            if seed is not None and confidence is not None:
                psi = psi + self.dt * lam * confidence * (seed - psi)
        return psi
