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

### Sequential Throughput

Due to its dependency on OpenMM 8.0.0, the Metal plugin can't implement an optimization inside the main code base that speeds up small systems. Therefore, there is an environment variable that can force-disable the usage of nearest neighbor lists inside Metal kernels. Turning off the neighbor list can substantially improve simulation speed for systems with under 3000 atoms.

```
export OPENMM_METAL_USE_NEIGHBOR_LIST=0 # accepted, force-disables neighor list
export OPENMM_METAL_USE_NEIGHBOR_LIST=1 # accepted, forces usage of neighbor list
export OPENMM_METAL_USE_NEIGHBOR_LIST=2 # runtime crash
unset OPENMM_METAL_USE_NEIGHBOR_LIST # accepted, automatically chooses whether to use list
```

Scaling behavior (Water Box, Amber Forcefield, PME):

| Atoms | Force No List | Force Use List | Auto |
| ----- | ----- | ----- | ----- |
| 6     | 1380 ns/day | 1120 ns/day | 1380 ns/day |
| 21    | 1260 ns/day | 1120 ns/day | 1260 ns/day |
| 96    | 1210 ns/day | 1040 ns/day | 1210 ns/day |
| 306   | 1010 ns/day |  860 ns/day | 1010 ns/day |
| 774   |  739 ns/day |  637 ns/day |  739 ns/day |
| 2661  |  490 ns/day |  438 ns/day |  490 ns/day |
| 4158  |  347 ns/day |  340 ns/day |  340 ns/day |

The above table is not a great example of the speedup possible by eliminating the sequential throughput bottleneck. Typically, this will provide an order of magnitude speedup. As in, not 18% speedup, but 18x speedup. Why that is not happening needs to be investigated further.


<!--

TODO: Switch to Metal and fuse several hundred commands into the same command buffer. Provide an option to specify how many commands to fuse. This overrides any default, and allows >1 command/command buffer while profiling. The downside is jumbled names for each group of X consecutive kernels.

-->

### Profiling

The Metal plugin can use the OpenCL queue profiling API to extract performance data about kernels. This API prevents batching of commands into command buffers, and reports the GPU start and end time of each buffer. The results are written to stdout in the JSON format used by https://ui.perfetto.dev.

```
export OPENMM_METAL_PROFILE_KERNELS=0 # accepted, does not profile
export OPENMM_METAL_PROFILE_KERNELS=1 # accepted, prints data to the console
export OPENMM_METAL_PROFILE_KERNELS=2 # runtime crash
unset OPENMM_METAL_PROFILE_KERNELS # accepted, does not profile
```

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
