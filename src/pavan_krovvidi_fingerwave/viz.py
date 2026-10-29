# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""Visualisation helpers (requires matplotlib: pip install "pavan-krovvidi-fingerwave[viz]")."""

import math

import torch

from .topo_loss import count_defects

__all__ = ["orientation_image", "plot_field"]


def orientation_image(field):
    """RGB image of a (1, 2, H, W) field: hue = ridge direction, brightness = |Ψ|."""
    from matplotlib.colors import hsv_to_rgb
    phi2 = torch.atan2(field[0, 1], field[0, 0])
    amp = torch.sqrt(field[0, 0] ** 2 + field[0, 1] ** 2).clamp(max=1.0)
    hsv = torch.stack([(phi2 + math.pi) / (2 * math.pi), torch.full_like(phi2, 0.75), 0.25 + 0.75 * amp], -1)
    return hsv_to_rgb(hsv.detach().cpu().numpy())


def plot_field(field, ax=None, title=None, show_defects=True):
    """Plot a (1, 2, H, W) field with cores (○) and deltas (△)."""
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    ax.imshow(orientation_image(field), origin="upper")
    if show_defects:
        s, idx = count_defects(field)
        ip, im = (idx[0] > 0).nonzero(), (idx[0] < 0).nonzero()
        ax.scatter(ip[:, 1] + 1, ip[:, 0] + 1, s=30, c="white", edgecolors="black")
        ax.scatter(im[:, 1] + 1, im[:, 0] + 1, s=30, c="black", edgecolors="white", marker="^")
        title = (title + "\n" if title else "") + f"cores {s['plus_half']}  deltas {s['minus_half']}"
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return ax
