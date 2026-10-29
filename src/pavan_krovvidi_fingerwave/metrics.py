# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""Evaluation metrics: orientation error, singular-point detection, TER and class accuracy."""

import torch

from .topo_loss import count_defects, wrap_angle

__all__ = ["angle_error_deg", "evaluate"]

SUPER_CLASS = {0: 0, 1: 1, 2: 1, 3: 1, 4: 2}   # arch / loop-family / whorl


def angle_error_deg(pred, true):
    """Per-pixel ridge-orientation error in degrees, (B, H, W)."""
    d = wrap_angle(torch.atan2(pred[:, 1], pred[:, 0]) - torch.atan2(true[:, 1], true[:, 0])) / 2
    return torch.rad2deg(d.abs())


def _match(pred_pts, gt_pts, radius):
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return 0
    D = torch.cdist(pred_pts, gt_pts)
    tp = 0
    while D.numel() and D.min() <= radius:
        i, j = divmod(int(D.argmin()), D.shape[1])
        tp += 1
        D[i, :] = float("inf")
        D[:, j] = float("inf")
    return tp


@torch.no_grad()
def evaluate(pred, data, match_radius=4.0):
    """
    Args:
        pred: (N, 2, H, W) predicted doubled-angle fields.
        data: dict from `data.make_dataset` (needs w_true, occ, defects, cls).
        match_radius: detection tolerance in grid pixels (4 = 8 image pixels).
    """
    ae = angle_error_deg(pred, data["w_true"])
    occ = data["occ"]
    _, idx = count_defects(pred)
    tp = fp = fn = wrong = right = 0
    N = pred.shape[0]
    for i in range(N):
        gt = data["defects"][i]
        gt = gt[~torch.isnan(gt[:, 0])]
        n_pred, bad = {}, False
        for sgn in (1, -1):
            loc = (idx[i] * sgn > 0).nonzero().float()
            pts = torch.stack([loc[:, 1] + 1.0, loc[:, 0] + 1.0], 1) if len(loc) else torch.zeros(0, 2)
            g = gt[gt[:, 2] * sgn > 0][:, :2]
            t = _match(pts, g, match_radius)
            tp, fp, fn = tp + t, fp + len(pts) - t, fn + len(g) - t
            n_pred[sgn] = len(pts)
            bad |= len(pts) != len(g)
        wrong += bad
        pc = {(0, 0): 0, (1, 1): 1, (2, 2): 2}.get((n_pred[1], n_pred[-1]), -1)
        right += pc == SUPER_CLASS[int(data["cls"][i])]
    return {
        "angle_all": ae.mean().item(),
        "angle_occ": ae[occ].mean().item(),
        "angle_vis": ae[~occ].mean().item(),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "TER": wrong / N,
        "class_acc": right / N,
        "spurious_per_img": fp / N,
    }
