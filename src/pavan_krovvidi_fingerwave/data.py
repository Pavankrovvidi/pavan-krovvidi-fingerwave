# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
Synthetic fingerprint benchmark generator (in the spirit of SFinGe).

1. Orientation fields: Sherlock–Monro zero-pole model, 5 classes
   (arch, tented arch, left loop, right loop, whorl), all with net index 0.
2. Ridges: iterative orientation-adaptive Gabor filtering.
3. Degradations: contrast loss, blur, noise, 1–3 smudges, optional partial print.
4. Model input: structure-tensor orientation estimate + coherence + image (64×64).
5. Ground truth: clean doubled-angle field, core/delta positions, class, occlusion mask.

    data = make_dataset(500, seed=1)
    x = make_model_input(data)          # (N, 4, 64, 64)
"""

import math

import torch
import torch.nn.functional as F

__all__ = ["CLASSES", "make_dataset", "make_model_input", "structure_tensor_field", "gaussian_blur"]

IMG = 128          # ridge image size (pixels)
ORI = 64           # orientation grid (2×2 blocks)
PERIOD = 7.0       # ridge period (pixels)
CLASSES = ["arch", "tented_arch", "left_loop", "right_loop", "whorl"]
MAX_DEF = 4        # maximum singular points per image


# ---------------------------------------------------------------------------
# 1. Orientation field (Sherlock–Monro)
# ---------------------------------------------------------------------------
def grid(n):
    s = IMG / n
    yy, xx = torch.meshgrid((torch.arange(n) + 0.5) * s, (torch.arange(n) + 0.5) * s, indexing="ij")
    return xx, yy


def smooth_noise(n, scale, amp, g):
    """Low-frequency random scalar field (single-valued, so it cannot change topology)."""
    z = torch.randn(1, 1, 6, 6, generator=g) * amp
    return F.interpolate(z, size=(n, n), mode="bicubic", align_corners=True)[0, 0] * scale


def sample_defects(cls, g):
    """Core (+½) / delta (−½) positions for a class, in image pixels."""
    U = lambda a, b: a + (b - a) * torch.rand(1, generator=g).item()
    cx, cy = U(52, 76), U(42, 60)
    if cls == "arch":
        return []
    if cls == "tented_arch":
        return [(cx, cy, 0.5), (cx + U(-3, 3), cy + U(12, 20), -0.5)]
    if cls == "left_loop":
        return [(cx, cy, 0.5), (cx + U(18, 32), cy + U(22, 34), -0.5)]
    if cls == "right_loop":
        return [(cx, cy, 0.5), (cx - U(18, 32), cy + U(22, 34), -0.5)]
    if cls == "whorl":
        dy = U(5, 10)
        return [(cx, cy - dy, 0.5), (cx + U(-3, 3), cy + dy, 0.5),
                (cx - U(24, 34), cy + U(26, 36), -0.5), (cx + U(24, 34), cy + U(26, 36), -0.5)]
    raise ValueError(cls)


def orientation_field(cls, defects, n, g, rot, pert):
    """Returns the ridge angle θ on an (n, n) grid."""
    xx, yy = grid(n)
    two_theta = torch.zeros(n, n)
    for x0, y0, k in defects:
        two_theta = two_theta + 2 * k * torch.atan2(yy - y0, xx - x0)
    if cls == "arch":
        cx, cy = IMG / 2, IMG * 0.45
        s = IMG * 0.22
        slope = -1.6 * (xx - cx) / s * torch.exp(-((xx - cx) ** 2) / (2 * s * s)) \
            * torch.exp(-((yy - cy) / (IMG * 0.6)) ** 2)
        two_theta = two_theta + 2 * torch.atan(slope)
    return 0.5 * (two_theta + pert) + rot


# ---------------------------------------------------------------------------
# 2. Gabor ridge rendering
# ---------------------------------------------------------------------------
N_BINS = 16


def gabor_bank(size=13, sigma=3.0):
    r = torch.arange(size) - size // 2
    yy, xx = torch.meshgrid(r.float(), r.float(), indexing="ij")
    ks = []
    for b in range(N_BINS):
        th = math.pi * b / N_BINS
        nrm = -xx * math.sin(th) + yy * math.cos(th)      # ridges run along θ; oscillation is along the normal
        k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * torch.cos(2 * math.pi * nrm / PERIOD)
        ks.append(k - k.mean())
    return torch.stack(ks).unsqueeze(1)                   # (16, 1, s, s)


def render_ridges(theta, g, iters=8):
    """Render ridges for theta (B, IMG, IMG) by iterative orientation-adaptive Gabor filtering of noise."""
    B = theta.shape[0]
    bank = gabor_bank()
    img = torch.randn(B, 1, IMG, IMG, generator=g)
    pos = torch.remainder(theta, math.pi) / math.pi * N_BINS
    lo = torch.floor(pos).long() % N_BINS
    hi = (lo + 1) % N_BINS
    w = (pos - torch.floor(pos)).unsqueeze(1)
    for _ in range(iters):
        resp = F.conv2d(img, bank, padding=bank.shape[-1] // 2)        # (B, 16, H, W)
        r = (1 - w) * resp.gather(1, lo.unsqueeze(1)) + w * resp.gather(1, hi.unsqueeze(1))
        img = torch.tanh(3 * r / (r.std(dim=(2, 3), keepdim=True) + 1e-6))
    return img[:, 0]


# ---------------------------------------------------------------------------
# 3. Degradations
# ---------------------------------------------------------------------------
def degrade(img, g):
    """img: (IMG, IMG) in [-1, 1]. Returns degraded image, occlusion mask (1 = occluded)."""
    U = lambda a, b: a + (b - a) * torch.rand(1, generator=g).item()
    xx, yy = grid(IMG)
    occ = torch.zeros(IMG, IMG)
    for _ in range(int(U(1, 3.99))):                        # 1-3 smudge blobs
        x0, y0 = U(20, 108), U(20, 108)
        a, b, phi = U(10, 22), U(8, 18), U(0, math.pi)
        dx, dy = xx - x0, yy - y0
        u = dx * math.cos(phi) + dy * math.sin(phi)
        v = -dx * math.sin(phi) + dy * math.cos(phi)
        occ = torch.maximum(occ, ((u / a) ** 2 + (v / b) ** 2 <= 1).float())
    if U(0, 1) < 0.5:                                      # partial print
        side, width = int(U(0, 3.99)), int(U(12, 30))
        band = torch.zeros(IMG, IMG)
        if side == 0: band[:, :width] = 1
        elif side == 1: band[:, -width:] = 1
        elif side == 2: band[:width, :] = 1
        else: band[-width:, :] = 1
        occ = torch.maximum(occ, band)

    out = img * U(0.5, 1.0)
    out = out * (1 - occ) + occ * U(-0.2, 0.4)
    k = int(U(0, 2.99)) * 2 + 1                            # blur 1/3/5
    if k > 1:
        out = F.avg_pool2d(out[None, None], k, 1, k // 2, count_include_pad=False)[0, 0]
    out = out + torch.randn(IMG, IMG, generator=g) * U(0.3, 0.9)
    return out, occ


# ---------------------------------------------------------------------------
# 4. Structure tensor orientation estimate
# ---------------------------------------------------------------------------
def gaussian_blur(x, sigma):
    r = int(3 * sigma)
    t = torch.arange(-r, r + 1).float()
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1),
                 groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1),
                    groups=x.shape[1])


def structure_tensor_field(img, sigma=2.0):
    """
    img: (B, IMG, IMG) -> director field w = c·e^{i·2θ} on ORI grid, (B, 2, ORI, ORI).
    Ridges are perpendicular to the intensity gradient, i.e. a factor −1 in doubled-angle form.
    """
    x = img.unsqueeze(1)
    sx = torch.tensor([[-1., 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3) / 8
    gx = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), sx)
    gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), sx.transpose(2, 3))
    T = torch.cat([gx * gx - gy * gy, 2 * gx * gy, gx * gx + gy * gy], 1)
    T = gaussian_blur(T, sigma)
    T = F.avg_pool2d(T, IMG // ORI)
    w = -T[:, :2] / (T[:, 2:3] + 1e-6)
    return w


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def make_dataset(n, seed, batch=50):
    g = torch.Generator().manual_seed(seed)
    out = {k: [] for k in ["img", "w_obs", "w_true", "occ", "defects", "n_def", "cls"]}
    for start in range(0, n, batch):
        m = min(batch, n - start)
        thetas_img, thetas_ori, defs, clss = [], [], [], []
        for _ in range(m):
            ci = int(torch.randint(0, len(CLASSES), (1,), generator=g))
            cls = CLASSES[ci]
            d = sample_defects(cls, g)
            rot = (torch.rand(1, generator=g).item() - 0.5) * math.radians(30)
            pert_seed = int(torch.randint(0, 2**31, (1,), generator=g))
            pg = torch.Generator().manual_seed(pert_seed)
            p_img = smooth_noise(IMG, 1.0, 0.35, pg)
            p_ori = F.interpolate(p_img[None, None], size=(ORI, ORI), mode="area")[0, 0]
            thetas_img.append(orientation_field(cls, d, IMG, g, rot, p_img))
            thetas_ori.append(orientation_field(cls, d, ORI, g, rot, p_ori))
            defs.append(d); clss.append(ci)
        ridges = render_ridges(torch.stack(thetas_img), g)
        degr, occs = zip(*[degrade(r, g) for r in ridges])
        degr = torch.stack(degr)
        out["img"].append(F.avg_pool2d(degr.unsqueeze(1), IMG // ORI)[:, 0].half())
        out["w_obs"].append(structure_tensor_field(degr).float())
        th = torch.stack(thetas_ori)
        out["w_true"].append(torch.stack([torch.cos(2 * th), torch.sin(2 * th)], 1))
        out["occ"].append(F.avg_pool2d(torch.stack(occs).unsqueeze(1), IMG // ORI)[:, 0] > 0.5)
        dt = torch.full((m, MAX_DEF, 3), float("nan"))
        for i, d in enumerate(defs):
            for j, (x0, y0, k) in enumerate(d):
                dt[i, j] = torch.tensor([x0 * ORI / IMG, y0 * ORI / IMG, k])   # ORI grid coords
        out["defects"].append(dt)
        out["n_def"].append(torch.tensor([len(d) for d in defs]))
        out["cls"].append(torch.tensor(clss))
    return {k: torch.cat(v) for k, v in out.items()}


def make_model_input(data):
    """Stack [normalised image, u, v, coherence] -> (N, 4, ORI, ORI), as used in the benchmark."""
    img = data["img"].float()
    img = (img - img.mean(dim=(1, 2), keepdim=True)) / (img.std(dim=(1, 2), keepdim=True) + 1e-6)
    coh = data["w_obs"].norm(dim=1, keepdim=True)
    return torch.cat([img.unsqueeze(1), data["w_obs"], coh], 1)
