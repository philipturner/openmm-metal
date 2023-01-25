# OpenMM Metal Plugin

This plugin adds the Metal platform that accelerates [OpenMM](https://openmm.org) on Metal 3 GPUs. It supports Apple, AMD, and Intel GPUs running macOS 13 or higher. The current implementation uses Apple's OpenCL compatiblity layer (`cl2metal`) to translate OpenCL kernels to AIR. Its current focus is implementing patches for OpenMM's code base that improve performance on macOS. It distributes the patches in a way easily accessible to most users, who would otherwise wait for them to be upstreamed. As a beta version of the patches, this may cause unexpected bugs or performance regressions.

Eventually, this should shift most code directly to the Metal API. Doing so enables optimizations like SIMD-group reductions and indirect command buffers, but removes double precision support on AMD GPUs. Prior to the shift, this plugin will establish [double-single FP64 emulation](https://andrewthall.org/papers/df64_qf128.pdf) and remove `double` precision, leaving only `mixed`. This is orthogonal to Kahan summation being considered for the main OpenMM code base, which enhances `single` precision.

Another goal is to support machine learning potentials, similar to [openmm-torch](https://github.com/openmm/openmm-torch). This repository should provide a more direct pathway to [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph), the high-level ML framework harnessed by tensorflow-metal and PyTorch. The plugin may create API for extracting the `MTLBuffer` backing an OpenMM class. This backend should make the ML potential (written in C++) accessible from Swift - the language for using MPSGraph. Swift code should access all other OpenMM APIs through [PythonKit](https://github.com/pvieito/PythonKit).

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
