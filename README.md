# Pavan Krovvidi FingerWave Neural Network

> *"Every life has a unique print. So does every prediction."*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Pavankrovvidi/pavan-krovvidi-fingerwave/blob/main/examples/FingerWave_Quickstart.ipynb)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Author](https://img.shields.io/badge/author-Pavan%20Kumar%20Krovvidi-1f6f8b)](https://www.linkedin.com/in/pavan-kumar-krovvidi-927290250)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23766123.svg)](https://doi.org/10.5281/zenodo.23766123)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--3286--7173-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0001-3286-7173)

**A biologically inspired, boundary-seeded neural PDE, conceived and created by [Pavan Kumar Krovvidi](https://www.linkedin.com/in/pavan-kumar-krovvidi-927290250).**

Your fingerprints were drawn in the womb by waves. Ridges begin at a few reliable initiation sites on the fingertip and travel inward. Where the waves meet, they form the cores and deltas that make every print unique. The FingerWave Neural Network brings that process into AI. It **starts from what it can trust, propagates information along the structure, and conserves topology**.

**▶️ Try it in your browser in 2 minutes, no install needed:** [Open the FingerWave notebook in Google Colab](https://colab.research.google.com/github/Pavankrovvidi/pavan-krovvidi-fingerwave/blob/main/examples/FingerWave_Quickstart.ipynb). Repair damaged fingerprints, cut your own holes into them, and measure the results.

```bash
pip install pavan-krovvidi-fingerwave
```

```python
from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork

model = PavanKrovvidiFingerWaveNetwork.from_pretrained()
clean_field = model(x)          # x: (B, 4, H, W) -> (B, 2, H, W) orientation field
```

---

## How it works

| Biology (fingerprint development) | FingerWave component |
|---|---|
| Ridges start at reliable initiation sites | **Boundary seeding**: a learned confidence map pulls trusted pixels toward an encoder seed |
| Ridges travel across the pad as waves | **Wavefront PDE layer**: stable reaction–diffusion, `∂Ψ/∂t = ∇·(D∇Ψ) + (α+iβ)Ψ − (1+iγ)|Ψ|²Ψ` |
| Pad geometry steers the waves | **Anisotropic diffusion tensor** `D` (faster along ridges than across them) |
| Cores and deltas are conserved | **Differentiable topological loss**: Ginzburg–Landau energy + soft topological charge |

Orientation is stored as a doubled-angle field `Ψ = A·e^{i2θ}`, so fingerprint cores (+½) and deltas (−½) become ordinary vortices.

## Results (pilot benchmark)

500 synthetic, degraded fingerprints with exact ground truth (≈15% of each image occluded), 3 random seeds, models of equal size (~118k parameters):

| Method | Angle error | Occluded regions | Topological Error Rate ↓ | Class accuracy ↑ |
|---|---|---|---|---|
| Classic smoothing | 8.32° | 16.98° | 44.4% | 55.6% |
| U-Net | 4.79° | 7.66° | 5.8 ± 1.1% | 94.2% |
| U-Net + topological loss | 4.75° | 7.62° | 5.3 ± 0.5% | 94.7% |
| **FingerWave** | **4.69°** | **7.36°** | **4.0 ± 0.2%** | **96.0%** |
| FingerWave without seeding | 10.07° | 12.58° | 37.7 ± 39% | 62.3% |
| FingerWave without anisotropy | 4.70° | 7.32° | 3.9 ± 0.5% | 96.1% |

![Benchmark results](https://raw.githubusercontent.com/Pavankrovvidi/pavan-krovvidi-fingerwave/main/results/benchmark_bars.png)

**Key findings**
- About **31% fewer topological errors** than a same-size U-Net. Every FingerWave run beat every U-Net run.
- **Boundary seeding is the essential ingredient.** Without it, training became unstable.
- Learned anisotropy gave **no extra benefit** on this benchmark. This is reported honestly.

**Next steps:** validation on real fingerprint databases (FVC, NIST), GPU-scale training, and a faster PDE (training is currently about 3.6× slower than the U-Net). See the full [research report](docs/FingerWave_Research_Report_Pavan_Kumar_Krovvidi.pdf) and the free [FingerWave Guide](docs/The_FingerWave_Guide_Pavan_Kumar_Krovvidi.pdf) (theory, mathematics and industry applications).

## Package contents

| Module | What it provides |
|---|---|
| `pavan_krovvidi_fingerwave.network` | `PavanKrovvidiFingerWaveNetwork`, `UNet` |
| `pavan_krovvidi_fingerwave.pde` | `WavefrontPDELayer` (anisotropic, stable, mass-conserving) |
| `pavan_krovvidi_fingerwave.topo_loss` | `topological_loss`, `count_defects` |
| `pavan_krovvidi_fingerwave.data` | SFinGe-style fingerprint benchmark generator with exact ground truth |
| `pavan_krovvidi_fingerwave.metrics` | Angle error, core/delta precision and recall, TER, class accuracy |
| `pavan_krovvidi_fingerwave.viz` | Orientation-field plots (`pip install "pavan-krovvidi-fingerwave[viz]"`) |

```python
import pavan_krovvidi_fingerwave as pk
pk.about()   # algorithm, author, components
pk.cite()    # citation
```

## Reproduce

```bash
pip install -e ".[dev]"
pytest                               # physics, topology and network tests
python examples/quickstart.py        # pretrained demo -> quickstart.png
# or open examples/FingerWave_Quickstart.ipynb (Colab notebook)
python benchmarks/run_benchmark.py   # full benchmark (~1.5 h on a 2-core CPU)
```

## Adapting FingerWave to your own data

1. **Choose the field Ψ.** Headless orientations (fibres, ridges, layers): doubled angle `Ψ = A·e^{i2θ}`. True directions (flow, movement): single angle `Ψ = A·e^{iθ}`. Plain quantities (temperature, potential): one channel, diffusion only.
2. **Prepare the inputs.** Give the encoder the raw signal, a rough estimate of the field and a quality measure (coherence, signal-to-noise ratio, sensor presence), and set `in_channels` to match.
3. **Choose the constraint.** Set the target net index for your domain in `topological_loss`, or replace the charge term with the rule your data must obey.
4. **Train and compare.** Report pixel error and a structural error rate against a same-size U-Net.

```python
from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork, topological_loss
model = PavanKrovvidiFingerWaveNetwork(in_channels=5)          # your inputs
loss = ((model(x) - target) ** 2).mean() + 0.05 * topological_loss(model(x), target_index=0.0)
```

Chapter 8 of the [FingerWave Guide](docs/The_FingerWave_Guide_Pavan_Kumar_Krovvidi.pdf) explains each step in detail.

## Possible applications

The method suits data that form a spatial field, carry direction, and are partly missing. Examples include latent fingerprints, diffusion MRI fibre orientation, cryo-EM map completion, and fast surrogates for electrostatics (Poisson–Boltzmann) in drug discovery. These applications are not yet tested.

## Citation

If you use this work, please cite it. GitHub's **"Cite this repository"** button uses [`CITATION.cff`](CITATION.cff).

```bibtex
@software{krovvidi2026fingerwave,
  author  = {Krovvidi, Pavan Kumar},
  orcid   = {0009-0001-3286-7173},
  title   = {Pavan Krovvidi FingerWave Neural Network: A Biologically Inspired,
             Boundary-Seeded Neural PDE},
  year    = {2026},
  doi     = {10.5281/zenodo.23766123},
  url     = {https://github.com/Pavankrovvidi/pavan-krovvidi-fingerwave}
}
```

## License and attribution

Copyright 2026 **Pavan Kumar Krovvidi**. Licensed under the [Apache License 2.0](LICENSE).
You may use, modify and distribute this work, including commercially. You must keep the copyright notices and include the [NOTICE](NOTICE) file that credits the author.

## Author

**Pavan Kumar Krovvidi**: AI Engineer and Independent Researcher, Vizianagaram / Hyderabad, India.
M.Sc. Microbiology · healthcare AI · founder of *Pure Vibe Coding*.
[LinkedIn](https://www.linkedin.com/in/pavan-kumar-krovvidi-927290250) · [GitHub](https://github.com/Pavankrovvidi) · [ORCID 0009-0001-3286-7173](https://orcid.org/0009-0001-3286-7173) · pavank9848@gmail.com

*Development of this project used AI assistance (Claude, Anthropic). The ideas, decisions and research direction are the author's own.*
