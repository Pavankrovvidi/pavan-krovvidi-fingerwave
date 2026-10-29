# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
"""
Quickstart: clean a noisy, partly hidden synthetic fingerprint with the pretrained model.

    pip install "pavan-krovvidi-fingerwave[viz]"
    python examples/quickstart.py          # writes quickstart.png
"""

import matplotlib.pyplot as plt
import torch

import pavan_krovvidi_fingerwave as pk
from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork
from pavan_krovvidi_fingerwave.data import make_dataset, make_model_input
from pavan_krovvidi_fingerwave.metrics import evaluate
from pavan_krovvidi_fingerwave.viz import plot_field

pk.about()

data = make_dataset(8, seed=123)            # 8 new synthetic fingerprints with ground truth
x = make_model_input(data)                  # (8, 4, 64, 64)

model = PavanKrovvidiFingerWaveNetwork.from_pretrained()
with torch.no_grad():
    pred = model(x)

print({k: round(v, 3) for k, v in evaluate(pred, data).items()})

fig, ax = plt.subplots(3, 4, figsize=(10, 7.8))
for i in range(4):
    plot_field(data["w_obs"][i:i + 1], ax[0, i], "noisy input")
    plot_field(pred[i:i + 1], ax[1, i], "FingerWave")
    plot_field(data["w_true"][i:i + 1], ax[2, i], "ground truth")
fig.suptitle("Pavan Krovvidi FingerWave Neural Network")
fig.tight_layout()
fig.savefig("quickstart.png", dpi=120)
print("saved quickstart.png")
