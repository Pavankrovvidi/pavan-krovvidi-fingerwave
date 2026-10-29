# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Pavan Kumar Krovvidi
# Part of the Pavan Krovvidi FingerWave Neural Network.
# This notice and the NOTICE file must be retained in all copies (Apache-2.0, Section 4).
"""
Pavan Krovvidi FingerWave Neural Network
"Every life has a unique print. So does every prediction."

A biologically inspired, boundary-seeded neural PDE created by Pavan Kumar Krovvidi.

    from pavan_krovvidi_fingerwave import PavanKrovvidiFingerWaveNetwork
    model = PavanKrovvidiFingerWaveNetwork.from_pretrained()

Run `pavan_krovvidi_fingerwave.about()` for details and `cite()` for the citation.
"""

from .network import PavanKrovvidiFingerWaveNetwork, UNet
from .pde import WavefrontPDELayer
from .topo_loss import count_defects, ginzburg_landau_energy, soft_charge_density, topological_loss

__version__ = "1.0.0"
__author__ = "Pavan Kumar Krovvidi"
__email__ = "pavank9848@gmail.com"
__orcid__ = "https://orcid.org/0009-0001-3286-7173"
__license__ = "Apache-2.0"
__credits__ = "Conceived and created by Pavan Kumar Krovvidi"
__algorithm__ = "Pavan Krovvidi FingerWave Neural Network"

__all__ = [
    "PavanKrovvidiFingerWaveNetwork", "UNet", "WavefrontPDELayer", "topological_loss",
    "ginzburg_landau_energy", "soft_charge_density", "count_defects", "about", "cite",
]

_CITATION = """Krovvidi, P. K. (2026). Pavan Krovvidi FingerWave Neural Network: A Biologically
Inspired, Boundary-Seeded Neural PDE (Version 1.0.0). https://github.com/Pavankrovvidi/pavan-krovvidi-fingerwave

@software{krovvidi2026fingerwave,
  author  = {Krovvidi, Pavan Kumar},
  orcid   = {0009-0001-3286-7173},
  title   = {Pavan Krovvidi FingerWave Neural Network: A Biologically Inspired,
             Boundary-Seeded Neural PDE},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/Pavankrovvidi/pavan-krovvidi-fingerwave},
  license = {Apache-2.0}
}"""


def cite():
    """Return (and print) the recommended citation."""
    print(_CITATION)
    return _CITATION


def about():
    """Print a short description of the algorithm and its author."""
    text = f"""
  Pavan Krovvidi FingerWave Neural Network  v{__version__}
  "Every life has a unique print. So does every prediction."

  Created by : Pavan Kumar Krovvidi  ({__email__})
  ORCID      : {__orcid__}
  Inspiration: fingerprint formation in the womb. Ridges start at a few reliable
               initiation sites and travel inward as waves; cores and deltas form
               where the waves meet.
  Components : 1. Boundary seeding     (confidence-driven initiation sites)
               2. Wavefront PDE layer  (stable anisotropic reaction-diffusion)
               3. Topological loss     (differentiable conservation of cores/deltas)
  Benchmark  : 500 synthetic fingerprints, topological errors 5.8% -> 4.0%
               vs a same-size U-Net (3 seeds; see the research report).
  License    : Apache-2.0. Keep the NOTICE file and copyright notices.
  Cite       : pavan_krovvidi_fingerwave.cite()
"""
    print(text)
    return text
