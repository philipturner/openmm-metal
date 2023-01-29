/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2019 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
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

#include "MetalKernelFactory.h"
#include "MetalParallelKernels.h"
#include "openmm/common/CommonKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

KernelImpl* MetalKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    MetalPlatform::PlatformData& data = *static_cast<MetalPlatform::PlatformData*>(context.getPlatformData());
    if (data.contexts.size() > 1) {
        // We are running in parallel on multiple devices, so we may want to create a parallel kernel.
        
        if (name == CalcForcesAndEnergyKernel::Name())
            return new MetalParallelCalcForcesAndEnergyKernel(name, platform, data);
        if (name == CalcHarmonicBondForceKernel::Name())
            return new MetalParallelCalcHarmonicBondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomBondForceKernel::Name())
            return new MetalParallelCalcCustomBondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcHarmonicAngleForceKernel::Name())
            return new MetalParallelCalcHarmonicAngleForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomAngleForceKernel::Name())
            return new MetalParallelCalcCustomAngleForceKernel(name, platform, data, context.getSystem());
        if (name == CalcPeriodicTorsionForceKernel::Name())
            return new MetalParallelCalcPeriodicTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcRBTorsionForceKernel::Name())
            return new MetalParallelCalcRBTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCMAPTorsionForceKernel::Name())
            return new MetalParallelCalcCMAPTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomTorsionForceKernel::Name())
            return new MetalParallelCalcCustomTorsionForceKernel(name, platform, data, context.getSystem());
        if (name == CalcNonbondedForceKernel::Name())
            return new MetalParallelCalcNonbondedForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomNonbondedForceKernel::Name())
            return new MetalParallelCalcCustomNonbondedForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomExternalForceKernel::Name())
            return new MetalParallelCalcCustomExternalForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomHbondForceKernel::Name())
            return new MetalParallelCalcCustomHbondForceKernel(name, platform, data, context.getSystem());
        if (name == CalcCustomCompoundBondForceKernel::Name())
            return new MetalParallelCalcCustomCompoundBondForceKernel(name, platform, data, context.getSystem());
    }
    MetalContext& cl = *data.contexts[0];
    if (name == CalcForcesAndEnergyKernel::Name())
        return new MetalCalcForcesAndEnergyKernel(name, platform, cl);
    if (name == UpdateStateDataKernel::Name())
        return new MetalUpdateStateDataKernel(name, platform, cl);
    if (name == ApplyConstraintsKernel::Name())
        return new CommonApplyConstraintsKernel(name, platform, cl);
    if (name == VirtualSitesKernel::Name())
        return new CommonVirtualSitesKernel(name, platform, cl);
    if (name == CalcHarmonicBondForceKernel::Name())
        return new CommonCalcHarmonicBondForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomBondForceKernel::Name())
        return new CommonCalcCustomBondForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcHarmonicAngleForceKernel::Name())
        return new CommonCalcHarmonicAngleForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomAngleForceKernel::Name())
        return new CommonCalcCustomAngleForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcPeriodicTorsionForceKernel::Name())
        return new CommonCalcPeriodicTorsionForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcRBTorsionForceKernel::Name())
        return new CommonCalcRBTorsionForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCMAPTorsionForceKernel::Name())
        return new CommonCalcCMAPTorsionForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomTorsionForceKernel::Name())
        return new CommonCalcCustomTorsionForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcNonbondedForceKernel::Name())
        return new MetalCalcNonbondedForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomNonbondedForceKernel::Name())
        return new CommonCalcCustomNonbondedForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcGBSAOBCForceKernel::Name())
        return new CommonCalcGBSAOBCForceKernel(name, platform, cl);
    if (name == CalcCustomGBForceKernel::Name())
        return new CommonCalcCustomGBForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomExternalForceKernel::Name())
        return new CommonCalcCustomExternalForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomHbondForceKernel::Name())
        return new CommonCalcCustomHbondForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomCentroidBondForceKernel::Name())
        return new CommonCalcCustomCentroidBondForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomCompoundBondForceKernel::Name())
        return new CommonCalcCustomCompoundBondForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcCustomCVForceKernel::Name())
        return new MetalCalcCustomCVForceKernel(name, platform, cl);
    if (name == CalcRMSDForceKernel::Name())
        return new CommonCalcRMSDForceKernel(name, platform, cl);
    if (name == CalcCustomManyParticleForceKernel::Name())
        return new CommonCalcCustomManyParticleForceKernel(name, platform, cl, context.getSystem());
    if (name == CalcGayBerneForceKernel::Name())
        return new CommonCalcGayBerneForceKernel(name, platform, cl);
    if (name == IntegrateVerletStepKernel::Name())
        return new CommonIntegrateVerletStepKernel(name, platform, cl);
    if (name == IntegrateLangevinStepKernel::Name())
        return new CommonIntegrateLangevinStepKernel(name, platform, cl);
    if (name == IntegrateLangevinMiddleStepKernel::Name())
        return new CommonIntegrateLangevinMiddleStepKernel(name, platform, cl);
    if (name == IntegrateBrownianStepKernel::Name())
        return new CommonIntegrateBrownianStepKernel(name, platform, cl);
    if (name == IntegrateVariableVerletStepKernel::Name())
        return new CommonIntegrateVariableVerletStepKernel(name, platform, cl);
    if (name == IntegrateVariableLangevinStepKernel::Name())
        return new CommonIntegrateVariableLangevinStepKernel(name, platform, cl);
    if (name == IntegrateCustomStepKernel::Name())
        return new CommonIntegrateCustomStepKernel(name, platform, cl);
    if (name == ApplyAndersenThermostatKernel::Name())
        return new CommonApplyAndersenThermostatKernel(name, platform, cl);
    if (name == IntegrateNoseHooverStepKernel::Name())
        return new CommonIntegrateNoseHooverStepKernel(name, platform, cl);
    if (name == ApplyMonteCarloBarostatKernel::Name())
        return new CommonApplyMonteCarloBarostatKernel(name, platform, cl);
    if (name == RemoveCMMotionKernel::Name())
        return new CommonRemoveCMMotionKernel(name, platform, cl);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
