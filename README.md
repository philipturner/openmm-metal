# OpenMM Metal Plugin

This plugin adds the Metal platform that accelerates [OpenMM](https://openmm.org) on Metal 3 GPUs. It supports Apple, AMD, and Intel GPUs running macOS Ventura or higher. The current implementation uses Apple's OpenCL compatiblity layer (`cl2Metal`) to translate OpenCL kernels to AIR. It focuses on patches for OpenMM's code base that improve performance on macOS. The plugin makes the patches easily accessible to most users, who would otherwise wait for them to be upstreamed. It also adds optimizations for Intel Macs that cannot be upstreamed into the main code base, for various reasons.

The Metal plugin may transition to using the Metal API directly. This would provide greater control over the CPU-side command encoding process, solving a long-standing latency bottleneck for 1000-atom systems. However, it removes mixed and double precision on Intel Macs. <b>FP64 emulation is proven to work on Apple silicon, but is not a priority for molecular dynamics.</b> A backward-compatible OpenCL compilation path may or may not be maintained.

## Performance

Expect a 2-4x speedup for large simulations.

## Usage

In Finder, create an empty folder and right-click it. Select `New Terminal at Folder` from the menu, which launches a Terminal window. Copy and enter the following commands one-by-one.

```bash
conda install -c conda-forge openmm
git clone https://github.com/openmm/openmm
git clone https://github.com/philipturner/openmm-metal
cd openmm-metal

# requires password
bash build.sh --install --quick-tests
```

Next, you will benchmark OpenCL against Metal. In your originally empty folder, enter `openmm/examples`. Arrange the files by name and locate `benchmark.py`. Then, open the file with TextEdit or Xcode. Modify the top of the script as shown below:

```python
from __future__ import print_function
import openmm.app as app
import openmm as mm
import openmm.unit as unit
from datetime import datetime
import argparse
import os

# What you need to add:
mm.Platform.loadPluginsFromDirectory(mm.Platform.getDefaultPluginsDirectory())

# Rest of the code:
def cpuinfo():
    """Return CPU info"""
```

Press `Cmd + S` to save. Back at the Terminal window, type the following commands:

```
cd ../
cd openmm
cd examples
python3 benchmark.py --test apoa1rf --seconds 15 --platform OpenCL
python3 benchmark.py --test apoa1rf --seconds 15 --platform HIP
```

OpenMM's current energy minimizer hard-codes checks for the `CUDA`, `OpenCL`, and `HIP` platforms. The Metal backend is currently labeled `HIP` everywhere to bypass this limitation. The plugin name will change to `Metal` after the next OpenMM version release, which fixes the issue. To prevent source-breaking changes, check for both the `HIP` and `Metal` backends in your client code.

## Testing

<!--

For reference: ✅
For reference: ❌

-->

Quick tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | 1.1.0-dev | 1.0.0 | 1.0.0  |

| Test                           | Apple FP32 | AMD FP32 | Intel FP32 | 
| ------------------------------ | ----------- | --------- | --------- | 
| CMAPTorsion                    | ✅           | ✅         | ✅         |
| CMAPMotionRemover              | ✅           | ✅         | ✅         |
| CMMotionRemover                | ✅           | ✅         | ✅         |
| Checkpoints                    | ✅           | ✅         | ✅         |
| CompoundIntegrator             | ✅           | ✅         | ✅         |
| CustomAngleForce               | ✅           | ✅         | ✅         |
| CustomBondForce                | ✅           | ✅         | ✅         |
| CustomCVForce                  | ✅           | ✅         | ✅         |
| CustomCentroidBondForce        | ✅           | ✅         | ✅         |
| CustomCompoundBondForce        | ✅           | ✅         | ✅         |
| CustomExternalForce            | ✅           | ✅         | ✅         |
| CustomGBForce                  | ✅           | ✅         | ✅         |
| CustomHbondForce               | ✅           | ✅         | ✅         |
| CustomTorsionForce             | ✅           | ✅         | ✅         |
| DeviceQuery                    | ✅           | ✅         | ✅         |
| DispersionPME                  | ✅           | ✅         | ❌         |
| Ewald                          | ✅           | ✅         | ❌         |
| FFT                            | ✅           | ✅         | ❌         |
| GBSAOBCForce                   | ✅           | ✅         | ✅         |
| GayBerneForce                  | ✅           | ✅         | ❌         |
| HarmonicAngleForce             | ✅           | ✅         | ✅         |
| HarmonicBondForce              | ✅           | ✅         | ✅         |
| MultipleForces                 | ✅           | ✅         | ✅         |
| PeriodicTorsionForce           | ✅           | ✅         | ✅         |
| RBTorsionForce                 | ✅           | ✅         | ✅         |
| RMSDForce                      | ✅           | ✅         | ✅         |
| Random                         | ✅           | ✅         | ✅         |
| Settle                         | ✅           | ✅         | ✅         |
| Sort                           | ✅           | ✅         | ✅         |
| VariableVerlet                 | ✅           | ✅         | ✅         |
| AmoebaExtrapolatedPolarization | ✅           | ✅         | ❌         |
| AmoebaGeneralizedKirkwoodForce | ✅           | ✅         | ❌         |
| AmoebaMultipoleForce           | ✅           | ✅         | ❌         |
| AmoebaTorsionTorsionForce      | ✅           | ✅         | ✅         |
| HippoNonbondedForce            | ✅           | ✅         | ❌         |
| WcaDispersionForce             | ✅           | ✅         | ✅         |
| DrudeForce                     | ✅           | ✅         | ✅         |

Long tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | 1.1.0-dev | 1.0.0 | 1.0.0  |

| Test                           | Apple FP32 | AMD FP32 | Intel FP32 | 
| ------------------------------ | ----------- | --------- | --------- | 
| AndersenThermostat             | ✅           | ✅         | ✅         |
| CustomManyParticleForce        | ✅           | ✅         | ✅         |
| CustomNonbondedForce           | ✅           | ✅         | ✅         |
| LangevinIntegrator             | ✅           | ✅         | ✅         |
| LangevinMiddleIntegrator       | ✅           | ✅         | ✅         |
| LocalEnergyMinimizer           | ✅           | ❌         | ✅         |
| MonteCarloFlexibleBarostat     | ✅           | ✅         | ✅         |
| NonbondedForce                 | ✅           | ✅         | ❌         |
| VerletIntegrator               | ✅           | ✅         | ✅         |
| VirtualSites                   | ✅           | ✅         | ✅         |
| DrudeNoseHoover                | ✅           | ✅         | ✅         |

Very long tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | 1.1.0-dev | n/a | n/a   |

| Test                           | Apple FP32 | AMD FP32 | Intel FP32 | 
| ------------------------------ | ----------- | --------- | --------- | 
| BrownianIntegrator             | ✅           | -         | -         |
| CustomIntegrator               | ✅           | -         | -         |
| MonteCarloAnisotropicBarostat  | ✅           | -         | -         |
| MonteCarloBarostat             | ✅           | -         | -         |
| NoseHooverIntegrator           | ✅           | -         | -         |
| VariableLangevinIntegrator     | ✅           | -         | -         |
| DrudeLangevinIntegrator        | ✅           | -         | -         |
| DrudeSCFIntegrator             | ✅           | -         | -         |

## License

The Metal Platform uses OpenMM API under the terms of the MIT License. A copy of this license may
be found in the accompanying file [MIT.txt](licenses/MIT.txt).

The Metal Platform is based on the OpenCL Platform of OpenMM under the terms of the GNU Lesser General
Public License. A copy of this license may be found in the accompanying file
[LGPL.txt](licenses/LGPL.txt). It in turn incorporates the terms of the GNU General Public
License, which may be found in the accompanying file [GPL.txt](licenses/GPL.txt).

The Metal Platform uses [VkFFT](https://github.com/DTolm/VkFFT) by Dmitrii Tolmachev under the terms
of the MIT License. A copy of this license may be found in the accompanying file
[MIT-VkFFT.txt](licenses/MIT-VkFFT.txt).
