# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
"""
Synthetic 2D electrostatics benchmark for drug discovery.

Each sample is a small "molecule": 6-14 atoms with partial charges inside a low-dielectric
interior (eps_in = 2), surrounded by water (eps_out = 80) with salt screening outside.
The exact potential phi solves the linearized Poisson-Boltzmann equation

    -div(eps grad phi) + kappa^2 (x) phi = rho

on a 64x64 grid with phi = 0 on the boundary, using a sparse direct solver.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

N = 64
EPS_IN, EPS_OUT, KAPPA2 = 2.0, 80.0, 0.02


def molecule(rng):
    n = rng.integers(6, 15)
    # chain-like layout: random walk of atoms around the centre
    pos = [np.array([N / 2, N / 2]) + rng.normal(0, 3, 2)]
    for _ in range(n - 1):
        step = rng.normal(0, 1, 2); step = step / np.linalg.norm(step) * rng.uniform(3.0, 4.5)
        pos.append(np.clip(pos[-1] + step, 12, N - 12))
    pos = np.array(pos)
    rad = rng.uniform(2.4, 3.6, n)
    q = rng.choice([-1.0, -0.5, 0.5, 1.0], n) * rng.uniform(0.6, 1.0, n)
    return pos, rad, q


def fields(pos, rad, q):
    yy, xx = np.mgrid[0:N, 0:N] + 0.5
    d = np.sqrt((xx[None] - pos[:, 0, None, None]) ** 2 + (yy[None] - pos[:, 1, None, None]) ** 2)
    sdist = (d - rad[:, None, None]).min(0)                         # signed distance to the surface
    inside = 1 / (1 + np.exp(sdist / 0.7))                          # smooth molecular interior
    eps = EPS_IN * inside + EPS_OUT * (1 - inside)
    kap = KAPPA2 * (1 - inside)
    rho = np.zeros((N, N))
    for (x, y), qq in zip(pos, q):
        rho += qq * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 0.8 ** 2)) / (2 * np.pi * 0.8 ** 2)
    surface = np.abs(sdist) < 1.2
    return eps, kap, rho * 40.0, inside, surface


def solve(eps, kap, rho):
    """Exact solution of -div(eps grad phi) + kap phi = rho, phi=0 outside (5-point flux scheme)."""
    idx = np.arange(N * N).reshape(N, N)
    rows, cols, vals = [], [], []
    diag = kap.ravel().copy()
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        en = np.zeros((N, N))
        ys, xs = np.mgrid[0:N, 0:N]
        yn, xn = ys + dy, xs + dx
        ok = (yn >= 0) & (yn < N) & (xn >= 0) & (xn < N)
        e_face = np.where(ok, 0.5 * (eps + eps[np.clip(yn, 0, N - 1), np.clip(xn, 0, N - 1)]), eps)
        diag += e_face.ravel()
        rows.append(idx[ok]); cols.append(idx[yn[ok], xn[ok]]); vals.append(-e_face[ok])
    A = sp.csr_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [idx.ravel()]), np.concatenate(cols + [idx.ravel()]))), shape=(N * N, N * N))
    return spla.spsolve(A, rho.ravel()).reshape(N, N)


def make_dataset(n, seed):
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("eps", "kap", "rho", "inside", "surface", "phi")}
    for _ in range(n):
        eps, kap, rho, inside, surface = fields(*molecule(rng))
        for k, v in zip(out, (eps, kap, rho, inside, surface, solve(eps, kap, rho))):
            out[k].append(v)
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


if __name__ == "__main__":
    import time
    t = time.time(); d = make_dataset(20, 0); dt = (time.time() - t) / 20
    print("solver ms/sample", round(dt * 1000, 1), "phi range", d["phi"].min(), d["phi"].max(), "std", d["phi"].std())
