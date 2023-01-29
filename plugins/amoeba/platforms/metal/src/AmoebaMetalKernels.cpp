/* -------------------------------------------------------------------------- *
 *                               OpenMMAmoeba                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2021 Stanford University and the Authors.      *
 * Authors: Peter Eastman, Mark Friedrichs                                    *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "AmoebaMetalKernels.h"

using namespace OpenMM;
using namespace std;

/* -------------------------------------------------------------------------- *
 *                             AmoebaMultipole                                *
 * -------------------------------------------------------------------------- */

MetalCalcAmoebaMultipoleForceKernel::~MetalCalcAmoebaMultipoleForceKernel() {
    if (fft != NULL)
        delete fft;
}

void MetalCalcAmoebaMultipoleForceKernel::initialize(const System& system, const AmoebaMultipoleForce& force) {
    CommonCalcAmoebaMultipoleForceKernel::initialize(system, force);
    if (usePME)
        fft = new MetalFFT3D(dynamic_cast<MetalContext&>(cc), gridSizeX, gridSizeY, gridSizeZ, false);
}

void MetalCalcAmoebaMultipoleForceKernel::computeFFT(bool forward) {
    MetalArray& grid1 = dynamic_cast<MetalContext&>(cc).unwrap(pmeGrid1);
    MetalArray& grid2 = dynamic_cast<MetalContext&>(cc).unwrap(pmeGrid2);
    if (forward)
        fft->execFFT(grid1, grid2, true);
    else
        fft->execFFT(grid2, grid1, false);
}

/* -------------------------------------------------------------------------- *
 *                           HippoNonbondedForce                              *
 * -------------------------------------------------------------------------- */

MetalCalcHippoNonbondedForceKernel::~MetalCalcHippoNonbondedForceKernel() {
    if (sort != NULL)
        delete sort;
    if (hasInitializedFFT) {
        delete fftForward;
        delete dfftForward;
    }
}

void MetalCalcHippoNonbondedForceKernel::initialize(const System& system, const HippoNonbondedForce& force) {
    CommonCalcHippoNonbondedForceKernel::initialize(system, force);
    if (usePME) {
        MetalContext& cl = dynamic_cast<MetalContext&>(cc);
        sort = new MetalSort(cl, new SortTrait(), cc.getNumAtoms());
        fftForward = new MetalFFT3D(cl, gridSizeX, gridSizeY, gridSizeZ, true);
        dfftForward = new MetalFFT3D(cl, dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ, true);
        hasInitializedFFT = true;
    }
}

void MetalCalcHippoNonbondedForceKernel::computeFFT(bool forward, bool dispersion) {
    MetalArray& grid1 = dynamic_cast<MetalContext&>(cc).unwrap(pmeGrid1);
    MetalArray& grid2 = dynamic_cast<MetalContext&>(cc).unwrap(pmeGrid2);
    MetalFFT3D* fft = (dispersion ? dfftForward : fftForward);
    if (forward)
        fft->execFFT(grid1, grid2, true);
    else
        fft->execFFT(grid2, grid1, false);
}

void MetalCalcHippoNonbondedForceKernel::sortGridIndex() {
    sort->sort(dynamic_cast<MetalContext&>(cc).unwrap(pmeAtomGridIndex));
}
