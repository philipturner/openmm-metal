# OpenMM Metal Plugin

This plugin adds the Metal platform that accelerates [OpenMM](https://openmm.org) on Metal 3 GPUs. It supports Apple, AMD, and Intel GPUs running macOS Ventura or higher. The current implementation uses Apple's OpenCL compatiblity layer (`cl2Metal`) to translate OpenCL kernels to AIR. Its current focus is implementing patches for OpenMM's code base that improve performance on macOS. It distributes the patches in a way easily accessible to most users, who would otherwise wait for them to be upstreamed. It also adds optimizations for Intel Macs that cannot be upstreamed into the main code base, for various reasons.

<!--
> \* The current version supports macOS Monterey. Ventura will only be required after the transition to Metal.

The Metal plugin will eventually transition kernels directly to the Metal API. Doing so enables optimizations like SIMD-group reductions and indirect command buffers, but removes double precision support on AMD GPUs. Before the transition, `double` and/or `mixed` precision will be deactivated. The plugin will eventually use [double-single FP64 emulation](https://andrewthall.org/papers/df64_qf128.pdf) to bring back `mixed`, this time supporting all GPUs.

Another goal is to support machine learning potentials, similar to [openmm-torch](https://github.com/openmm/openmm-torch). This repository should provide a more direct pathway to [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph), the high-level MLIR compiler harnessed by tensorflow-metal and PyTorch. The plugin should create API (e.g. `MPSGraphForce`) for extracting the `MTLBuffer` backing an OpenMM class. The API should also facilitate construction of `MPSGraphTensor` and `MPSGraphTensorData` instances from the buffer. The ML potential (written in C++) should be made accessible from Swift - the language for using MPSGraph. Swift code will access all other OpenMM APIs through [PythonKit](https://github.com/pvieito/PythonKit).
-->

## Performance

Expect a 2-4x speedup depending on your workload.

## Usage

In Finder, create an empty folder and right-click it. Select `New Terminal at Folder` from the menu, which launches a Terminal window. Copy and enter the following commands one-by-one.

```bash
conda install -c conda-forge openmm
git clone https://github.com/openmm/openmm
git clone https://github.com/philipturner/openmm-metal
cd openmm-metal
bash build.sh
PLUGINS_DIR=/usr/local/openmm/lib/plugins

# requires password
sudo mkdir -p $PLUGINS_DIR
libs=(libOpenMMMetal libOpenMMAmoebaMetal libOpenMMDrudeMetal libOpenMMRPMDMetal)
for lib in $libs; do sudo cp ".build/${lib}.dylib" "$PLUGINS_DIR/${lib}.dylib"; done
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

Legend:
- F32 = single precision
- F64 = mixed precision
- Mixed precision emulation is planned for Apple silicon, but currently not a priority.
- Double precision is not tested or officially supported.

Quick tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | n/a   | n/a | n/a   |

| Test                           | Apple (F32) | Apple (F64) | AMD (F32) | AMD (F64) | Intel (F32) | Intel (F64) |
| ------------------------------ | ----------- | ----------- | --------- | --------- | ----------- | ----------- |
|                                |             |             | -         | -         |             |             |

Long tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | n/a   | n/a | n/a   |

| Test                           | Apple (F32) | Apple (F64) | AMD (F32) | AMD (F64) | Intel (F32) | Intel (F64) |
| ------------------------------ | ----------- | ----------- | --------- | --------- | ----------- | ----------- |
|                                |             |             | -         | -         |             |             |

Very long tests:

|                     | Apple | AMD | Intel |
| ------------------- | ----- | --- | ----- |
| Last tested version | n/a   | n/a | n/a   |


| Test                           | Apple (F32) | Apple (F64) | AMD (F32) | AMD (F64) | Intel (F32) | Intel (F64) |
| ------------------------------ | ----------- | ----------- | --------- | --------- | ----------- | ----------- |
|                                |             |             | -         | -         |             |             |

## License

The Metal Platform uses OpenMM API under the terms of the MIT License.  A copy of this license may
be found in the accompanying file [MIT.txt](licenses/MIT.txt).

The Metal Platform is based on the OpenCL Platform of OpenMM under the terms of the GNU Lesser General
Public License.  A copy of this license may be found in the accompanying file
[LGPL.txt](licenses/LGPL.txt).  It in turn incorporates the terms of the GNU General Public
License, which may be found in the accompanying file [GPL.txt](licenses/GPL.txt).

The Metal Platform uses [VkFFT](https://github.com/DTolm/VkFFT) by Dmitrii Tolmachev under the terms
of the MIT License.  A copy of this license may be found in the accompanying file
[MIT-VkFFT.txt](licenses/MIT-VkFFT.txt).
