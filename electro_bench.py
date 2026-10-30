# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
"""FingerWave (seeded physics diffusion) vs same-size U-Net as a fast electrostatics surrogate."""
import json, math, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
# pip install pavan-krovvidi-fingerwave scipy   (run from this folder)
from pavan_krovvidi_fingerwave.network import UNet
from pavan_krovvidi_fingerwave.pde import anisotropic_divergence
from electro_data import make_dataset, EPS_OUT, KAPPA2

PHI_SCALE = 5.0
torch.set_num_threads(2)


def to_tensors(d):
    x = np.stack([d["eps"] / EPS_OUT, d["kap"] / KAPPA2, d["rho"] / 40.0, d["inside"]], 1)
    return torch.tensor(x), torch.tensor(d["phi"][:, None] * PHI_SCALE), torch.tensor(d["surface"][:, None] > 0)


def isp(v):  # inverse softplus
    return math.log(math.expm1(v))


class PhysicsStage(nn.Module):
    """phi <- phi + dt*[a*div(eps grad phi) - b*kap*phi + g*rho] + dt*lam*c*(seed - phi)."""
    def __init__(self, steps, a, b, g):
        super().__init__()
        self.steps, self.dt = steps, 0.2
        self.ra, self.rb, self.rg = (nn.Parameter(torch.tensor(isp(v))) for v in (a, b, g))

    def forward(self, phi, eps, kap, rho, seed, conf, lam):
        a = F.softplus(self.ra).clamp(max=1.0)   # a*eps <= 1 keeps the step stable
        b, g = F.softplus(self.rb), F.softplus(self.rg)
        D = a * eps[:, 0]
        for _ in range(self.steps):
            phi = phi + self.dt * (anisotropic_divergence(phi, D, torch.zeros_like(D), D) - b * kap * phi + g * rho)
            phi = phi + self.dt * lam * conf * (seed - phi)
        return phi


class FingerWaveElectro(nn.Module):
    """Pavan Krovvidi FingerWave for scalar fields: encoder seed + confidence, then coarse and fine physics waves."""
    def __init__(self, width=16):
        super().__init__()
        self.encoder = UNet(4, 2, width)
        self.coarse = PhysicsStage(40, 0.9, 0.5, 0.5)
        self.fine = PhysicsStage(20, 0.9, 0.5, 0.5)
        self.raw_lam = nn.Parameter(torch.zeros(2))

    def forward(self, x, return_confidence=False):
        e = self.encoder(x)
        seed, c = e[:, :1], torch.sigmoid(e[:, 1:2])
        lam = 4 * torch.sigmoid(self.raw_lam)
        eps, kap, rho = x[:, :1], x[:, 1:2], x[:, 2:3]
        p4 = lambda t: F.avg_pool2d(t, 4)
        phi = self.coarse(p4(seed), p4(eps), p4(kap), p4(rho), p4(seed), p4(c), lam[0])
        phi = F.interpolate(phi, scale_factor=4, mode="bilinear")
        phi = c * seed + (1 - c) * phi
        phi = self.fine(phi, eps, kap, rho, seed, c, lam[1])
        return (phi, c) if return_confidence else phi


def metrics(pred, y, surf):
    err = (pred - y) / PHI_SCALE
    rel = (err.flatten(1).norm(dim=1) / (y / PHI_SCALE).flatten(1).norm(dim=1)).mean().item()
    s = surf.float()
    surf_mae = ((err.abs() * s).flatten(1).sum(1) / s.flatten(1).sum(1)).mean().item()
    # sign of the surface potential decides attraction vs repulsion for a charged ligand
    sign_ok = (((torch.sign(pred) == torch.sign(y)) | (y.abs() < 0.05 * PHI_SCALE)).float() * s).flatten(1).sum(1) / s.flatten(1).sum(1)
    return {"rel_l2": rel, "surface_mae": surf_mae, "surface_sign_acc": sign_ok.mean().item()}


def train(model, X, Y, seed, epochs=15, bs=16):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    steps = epochs * math.ceil(len(X) / bs); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            loss = F.mse_loss(model(X[j]), Y[j])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
    return time.time() - t


if __name__ == "__main__":
    t = time.time(); tr = make_dataset(2000, 0); te = make_dataset(300, 1)
    solver_ms = (time.time() - t) / 2300 * 1000
    Xtr, Ytr, _ = to_tensors(tr); Xte, Yte, Ste = to_tensors(te)
    print(f"data ready, solver {solver_ms:.1f} ms/sample", flush=True)
    res = {"solver_ms": solver_ms, "runs": {}}
    for name, mk in [("U-Net", lambda: UNet(4, 1, 16)), ("FingerWave", FingerWaveElectro)]:
        res["runs"][name] = []
        for seed in (0, 1, 2):
            torch.manual_seed(seed); m = mk()
            tt = train(m, Xtr, Ytr, seed)
            m.eval()
            with torch.no_grad():
                pred = m(Xte)
                t0 = time.time(); m(Xte[:100]); ms = (time.time() - t0) / 100 * 1000
            r = metrics(pred, Yte, Ste); r.update(train_s=tt, infer_ms=ms, params=sum(p.numel() for p in m.parameters()))
            res["runs"][name].append(r); print(name, seed, {k: round(v, 4) for k, v in r.items()}, flush=True)
            if seed == 0:
                torch.save(m.state_dict(), f"{name.replace('-', '').lower()}_electro.pt")
    json.dump(res, open("electro_results.json", "w"), indent=1)
