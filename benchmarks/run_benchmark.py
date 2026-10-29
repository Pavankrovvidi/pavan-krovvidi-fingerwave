# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
Reproduce the synthetic fingerprint benchmark from the research report.

    python benchmarks/run_benchmark.py                       # all 5 learned methods, 3 seeds
    python benchmarks/run_benchmark.py --methods FingerWave --seeds 0 --save-weights w.pt

Outputs results/benchmark_results.json. On a 2-core CPU the full run takes about 1.5 hours.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork, UNet, topological_loss
from pavan_krovvidi_fingerwave.data import gaussian_blur, make_dataset, make_model_input
from pavan_krovvidi_fingerwave.metrics import angle_error_deg, evaluate

METHODS = {
    "U-Net": (lambda: UNet(4, 2, 16), 0.0),
    "U-Net + topo": (lambda: UNet(4, 2, 16), 0.05),
    "FingerWave": (lambda: PavanKrovvidiFingerWaveNetwork(), 0.05),
    "FingerWave - seeding": (lambda: PavanKrovvidiFingerWaveNetwork(seeded=False), 0.05),
    "FingerWave - anisotropy": (lambda: PavanKrovvidiFingerWaveNetwork(isotropic=True), 0.05),
}


def load_or_make(path, n, seed):
    if os.path.exists(path):
        return torch.load(path)
    print(f"Generating {n} synthetic fingerprints (seed {seed}) ...", flush=True)
    d = make_dataset(n, seed=seed)
    torch.save(d, path)
    return d


def train(model, x, y, topo_w, seed, epochs, bs=16, lr=2e-3):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(x) / bs))
    for ep in range(epochs):
        perm, tot = torch.randperm(len(x)), 0.0
        for s in range(0, len(x), bs):
            b = perm[s:s + bs]
            pred = model(x[b])
            loss = F.mse_loss(pred, y[b]) + (topo_w * topological_loss(pred) if topo_w else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * len(b)
        print(f"    epoch {ep + 1:2d}/{epochs}  loss {tot / len(x):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, x, bs=100):
    model.eval()
    out = torch.cat([model(x[s:s + bs]) for s in range(0, len(x), bs)])
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/benchmark_results.json")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+", default=list(METHODS) + ["Classic"])
    ap.add_argument("--save-weights", default=None, help="save the first FingerWave model here")
    a = ap.parse_args()
    os.makedirs(a.data_dir, exist_ok=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tr = load_or_make(os.path.join(a.data_dir, "fp_train.pt"), 2000, 0)
    te = load_or_make(os.path.join(a.data_dir, "fp_test.pt"), 500, 1)
    xtr, xte = make_model_input(tr), make_model_input(te)
    results = {}

    if "Classic" in a.methods:
        best = min([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                   key=lambda s: angle_error_deg(gaussian_blur(tr["w_obs"][:400], s) if s else tr["w_obs"][:400],
                                                 tr["w_true"][:400]).mean().item())
        results["Classic"] = [evaluate(gaussian_blur(te["w_obs"], best) if best else te["w_obs"], te)]
        print(f"[Classic] sigma={best}: {results['Classic'][0]}")

    for name in [m for m in a.methods if m in METHODS]:
        make, tw = METHODS[name]
        results[name] = []
        for sd in a.seeds:
            torch.manual_seed(sd)
            model = make()
            print(f"\n[{name}] seed {sd}", flush=True)
            t0 = time.time()
            train(model, xtr, tr["w_true"], tw, sd, a.epochs)
            r = evaluate(predict(model, xte), te)
            r["train_seconds"] = time.time() - t0
            print(f"    -> angle {r['angle_all']:.2f} deg (occluded {r['angle_occ']:.2f})  TER {r['TER']:.3f}", flush=True)
            results[name].append(r)
            if a.save_weights and name == "FingerWave" and not os.path.exists(a.save_weights):
                torch.save(model.state_dict(), a.save_weights)
                print(f"    saved weights -> {a.save_weights}")
        json.dump(results, open(a.out, "w"), indent=1)

    print("\nMethod                      angle(all)  angle(occ)   TER     class acc")
    for name, runs in results.items():
        m = {k: sum(r[k] for r in runs) / len(runs) for k in ("angle_all", "angle_occ", "TER", "class_acc")}
        print(f"{name:26s} {m['angle_all']:9.2f}  {m['angle_occ']:9.2f}  {m['TER']:7.3f}  {m['class_acc']:8.3f}")


if __name__ == "__main__":
    main()
