# OpenMM Metal Plugin

> WARNING: Only use with M1/M2-series for now. This plugin has not been fully tested on Intel Macs and seems to cause bugs there. Precompiled binaries will be released once these issues are resolved.

This plugin adds the Metal platform that accelerates [OpenMM](https://openmm.org) on Metal 3 GPUs. It supports Apple, AMD, and Intel GPUs running macOS 13\* or higher. The current implementation uses Apple's OpenCL compatiblity layer (`cl2metal`) to translate OpenCL kernels to AIR. Its current focus is implementing patches for OpenMM's code base that improve performance on macOS. It distributes the patches in a way easily accessible to most users, who would otherwise wait for them to be upstreamed. As a beta version of the patches, this may cause unexpected bugs or performance regressions.

> \* The current version supports macOS Monterey. Ventura will only be required after the transition to Metal.

The Metal plugin will eventually transition kernels directly to the Metal API. Doing so enables optimizations like SIMD-group reductions and indirect command buffers, but removes double precision support on AMD GPUs. Before the transition, `double` and/or `mixed` precision will be deactivated. The plugin will eventually use [double-single FP64 emulation](https://andrewthall.org/papers/df64_qf128.pdf) to bring back `mixed`, this time supporting all GPUs. <!--This is orthogonal to Kahan summation being considered for the main OpenMM code base, which enhances `single` precision.-->

Another goal is to support machine learning potentials, similar to [openmm-torch](https://github.com/openmm/openmm-torch). This repository should provide a more direct pathway to [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph), the high-level MLIR compiler harnessed by tensorflow-metal and PyTorch. The plugin should create API (e.g. `MPSGraphForce`) for extracting the `MTLBuffer` backing an OpenMM class. The API should also facilitate construction of `MPSGraphTensor` and `MPSGraphTensorData` instances from the buffer. The ML potential (written in C++) should be made accessible from Swift - the language for using MPSGraph. Swift code will access all other OpenMM APIs through [PythonKit](https://github.com/pvieito/PythonKit).

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

OpenMM's current energy minimizer hard-codes checks for the `CUDA`, `OpenCL`, and `HIP` platforms. The Metal backend is currently labeled `HIP` everywhere to bypass this limitation. The plugin name will change to `Metal` once OpenMM provides integration internally. To prevent source-breaking changes, check for both the `HIP` and `Metal` backends in your client code.

---

TODO: For user convenience, attach pre-compiled binaries and `install.sh` to the first official release. The installer automatically finds the best binary version (based on macOS version compatibility) and the correct architecture. Then, it downloads and checks SHA256. Finally, make a way to query the version of each currently installed binary (e.g. a dynamically loaded symbol, text file in installation directory).

## Performance

Expect a 100-200% speedup depending on your workload.

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
