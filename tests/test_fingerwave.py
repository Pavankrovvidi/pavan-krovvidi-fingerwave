# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
"""Tests that reproduce the verification section of the research report."""

import math

import pytest
import torch

import pavan_krovvidi_fingerwave as pk
from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork, WavefrontPDELayer, count_defects, topological_loss
from pavan_krovvidi_fingerwave.topo_loss import plaquette_winding, soft_charge_density


def synthetic_field(H, W, defects, core=2.0):
    yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing="ij")
    phi2, amp = torch.zeros(H, W), torch.ones(H, W)
    for x0, y0, k in defects:
        phi2 = phi2 + 2 * k * torch.atan2(yy - y0, xx - x0)
        amp = amp * torch.tanh(torch.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) / core)
    return torch.stack([amp * torch.cos(phi2), amp * torch.sin(phi2)]).unsqueeze(0)


def legacy_winding_loss(field):
    """The original (zero-gradient) loss, kept only to document the bug."""
    idx = plaquette_winding(torch.atan2(field[:, 1], field[:, 0]))
    return idx.abs().mean() + (idx.sum(dim=(1, 2)) - 1.0).pow(2).mean()


# ---------------------------------------------------------------- topology
@pytest.mark.parametrize("defects,plus,minus", [
    ([(32.3, 22.4, 0.5), (32.3, 44.4, -0.5)], 1, 1),                              # loop
    ([(32.3, 26.4, 1.0), (16.3, 48.4, -0.5), (48.3, 48.4, -0.5)], 2, 2),          # whorl
    ([], 0, 0),                                                                    # arch
])
def test_count_defects_known_patterns(defects, plus, minus):
    s, _ = count_defects(synthetic_field(64, 64, defects))
    assert (s["plus_half"], s["minus_half"], s["net_index"]) == (plus, minus, 0.0)


def test_new_loss_has_gradient_old_does_not():
    f = torch.randn(1, 2, 32, 32, requires_grad=True)
    g_old, = torch.autograd.grad(legacy_winding_loss(f), f)
    g_new, = torch.autograd.grad(topological_loss(f), f)
    assert g_old.norm() < 1e-5
    assert g_new.norm() > 1e-3


def test_soft_charge_net_zero_for_loop():
    f = synthetic_field(64, 64, [(32.3, 22.4, 0.5), (32.3, 44.4, -0.5)])
    assert abs(float(soft_charge_density(f).sum() / 2)) < 0.05


# ---------------------------------------------------------------- PDE physics
def _spread(theta, d_par, d_perp, T=10.0, N=101):
    layer = WavefrontPDELayer(d_max=1.0, d_par=d_par, d_perp=d_perp, reaction=False)
    with torch.no_grad():
        layer.raw_par.fill_(math.log(d_par / (1 - d_par)) if d_par < 1 else 20.0)
        layer.raw_perp.fill_(math.log(d_perp / (1 - d_perp)) if d_perp < 1 else 20.0)
        psi = torch.zeros(1, 1, N, N); psi[..., N // 2, N // 2] = 1.0
        out = layer(psi, theta=torch.full((1, N, N), theta), steps=int(round(T / layer.dt)))
    img = out[0, 0].double(); img = img / img.sum()
    yy, xx = torch.meshgrid(torch.arange(N, dtype=torch.float64), torch.arange(N, dtype=torch.float64), indexing="ij")
    mx, my = (img * xx).sum(), (img * yy).sum()
    C = torch.tensor([[(img * (xx - mx) ** 2).sum(), (img * (xx - mx) * (yy - my)).sum()],
                      [(img * (xx - mx) * (yy - my)).sum(), (img * (yy - my) ** 2).sum()]])
    return C


def test_isotropic_heat_equation():
    C = _spread(0.0, 1.0, 1.0)
    assert abs(C[0, 0] - 20.0) / 20.0 < 0.02 and abs(C[1, 1] - 20.0) / 20.0 < 0.02


def test_anisotropy_ratio_and_direction():
    C = _spread(math.radians(30), 1.0, 0.1)
    ev, evec = torch.linalg.eigh(C)
    assert abs(ev[1] / ev[0] - 10) < 0.5
    assert abs(math.degrees(math.atan2(evec[1, 1], evec[0, 1])) % 180 - 30) < 1.0


def test_mass_conservation():
    layer = WavefrontPDELayer(d_par=0.99, d_perp=0.05, reaction=False)
    psi, theta = torch.rand(1, 2, 64, 64), torch.rand(1, 64, 64) * math.pi
    with torch.no_grad():
        out = layer(psi, theta=theta, steps=300)
    assert abs(out.double().sum() - psi.double().sum()) / psi.double().sum() < 1e-5


def test_stability_with_large_inputs():
    layer = WavefrontPDELayer(d_par=0.99, d_perp=0.99)
    with torch.no_grad():
        layer.raw_par.fill_(50.0); layer.raw_perp.fill_(50.0)
        out = layer(torch.randn(1, 2, 64, 64) * 3, steps=60)
    assert torch.isfinite(out).all() and out.abs().max() < 2.0


# ---------------------------------------------------------------- network
@pytest.mark.parametrize("kw", [{}, {"seeded": False}, {"isotropic": True}])
def test_network_forward_backward(kw):
    model = PavanKrovvidiFingerWaveNetwork(coarse_steps=5, fine_steps=3, **kw)
    x = torch.randn(2, 4, 32, 32)
    out, c = model(x, return_confidence=True)
    assert out.shape == (2, 2, 32, 32) and c.shape == (2, 1, 32, 32)
    (out.pow(2).mean() + topological_loss(out)).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_pretrained_weights_load():
    model = PavanKrovvidiFingerWaveNetwork.from_pretrained()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 64, 64))
    assert out.shape == (1, 2, 64, 64) and torch.isfinite(out).all()


def test_metadata_and_attribution():
    assert pk.__author__ == "Pavan Kumar Krovvidi"
    assert "Pavan Kumar Krovvidi" in pk.about()
    assert "Krovvidi" in pk.cite()
