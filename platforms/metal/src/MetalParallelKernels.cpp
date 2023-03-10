/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2011-2019 Stanford University and the Authors.      *
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

#include "MetalParallelKernels.h"

using namespace OpenMM;
using namespace std;

/**
 * Get the current clock time, measured in microseconds.
 */
#ifdef _MSC_VER
    #include <Windows.h>
    static long long getTime() {
        FILETIME ft;
        GetSystemTimeAsFileTime(&ft); // 100-nanoseconds since 1-1-1601
        ULARGE_INTEGER result;
        result.LowPart = ft.dwLowDateTime;
        result.HighPart = ft.dwHighDateTime;
        return result.QuadPart/10;
    }
#else
    #include <sys/time.h> 
    static long long getTime() {
        struct timeval tod;
        gettimeofday(&tod, 0);
        return 1000000*tod.tv_sec+tod.tv_usec;
    }
#endif

class MetalParallelCalcForcesAndEnergyKernel::BeginComputationTask : public MetalContext::WorkTask {
public:
    BeginComputationTask(ContextImpl& context, MetalContext& cl, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, void* pinnedMemory, int& numTiles) : context(context), cl(cl), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), pinnedMemory(pinnedMemory), numTiles(numTiles) {
    }
    void execute() {
        // Copy coordinates over to this device and execute the kernel.

        if (cl.getContextIndex() > 0)
            cl.getQueue().enqueueWriteBuffer(cl.getPosq().getDeviceBuffer(), CL_FALSE, 0, cl.getPaddedNumAtoms()*cl.getPosq().getElementSize(), pinnedMemory);
        kernel.beginComputation(context, includeForce, includeEnergy, groups);
        if (cl.getNonbondedUtilities().getUsePeriodic())
            cl.getNonbondedUtilities().getInteractionCount().download(&numTiles, false);
    }
private:
    ContextImpl& context;
    MetalContext& cl;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    void* pinnedMemory;
    int& numTiles;
};

class MetalParallelCalcForcesAndEnergyKernel::FinishComputationTask : public MetalContext::WorkTask {
public:
    FinishComputationTask(ContextImpl& context, MetalContext& cl, MetalCalcForcesAndEnergyKernel& kernel,
            bool includeForce, bool includeEnergy, int groups, double& energy, long long& completionTime, void* pinnedMemory, bool& valid, int& numTiles) :
            context(context), cl(cl), kernel(kernel), includeForce(includeForce), includeEnergy(includeEnergy), groups(groups), energy(energy),
            completionTime(completionTime), pinnedMemory(pinnedMemory), valid(valid), numTiles(numTiles) {
    }
    void execute() {
        // Execute the kernel, then download forces.
        
        energy += kernel.finishComputation(context, includeForce, includeEnergy, groups, valid);
        if (includeForce) {
            if (cl.getContextIndex() > 0) {
                int numAtoms = cl.getPaddedNumAtoms();
                void* dest = (cl.getUseDoublePrecision() ? (void*) &((mm_double4*) pinnedMemory)[(cl.getContextIndex()-1)*numAtoms] : (void*) &((mm_float4*) pinnedMemory)[(cl.getContextIndex()-1)*numAtoms]);
                cl.getQueue().enqueueReadBuffer(cl.getForce().getDeviceBuffer(), CL_TRUE, 0,
                        numAtoms*cl.getForce().getElementSize(), dest);
            }
            else
                cl.getQueue().finish();
        }
        completionTime = getTime();
        if (cl.getNonbondedUtilities().getUsePeriodic() && numTiles > cl.getNonbondedUtilities().getInteractingTiles().getSize()) {
            valid = false;
            cl.getNonbondedUtilities().updateNeighborListSize();
        }
    }
private:
    ContextImpl& context;
    MetalContext& cl;
    MetalCalcForcesAndEnergyKernel& kernel;
    bool includeForce, includeEnergy;
    int groups;
    double& energy;
    long long& completionTime;
    void* pinnedMemory;
    bool& valid;
    int& numTiles;
};

MetalParallelCalcForcesAndEnergyKernel::MetalParallelCalcForcesAndEnergyKernel(string name, const Platform& platform, MetalPlatform::PlatformData& data) :
        CalcForcesAndEnergyKernel(name, platform), data(data), completionTimes(data.contexts.size()), contextNonbondedFractions(data.contexts.size()),
        tileCounts(data.contexts.size()), pinnedPositionBuffer(NULL), pinnedPositionMemory(NULL), pinnedForceBuffer(NULL), pinnedForceMemory(NULL) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcForcesAndEnergyKernel(name, platform, *data.contexts[i])));
}

MetalParallelCalcForcesAndEnergyKernel::~MetalParallelCalcForcesAndEnergyKernel() {
    if (pinnedPositionBuffer != NULL)
        delete pinnedPositionBuffer;
    if (pinnedForceBuffer != NULL)
        delete pinnedForceBuffer;
}

void MetalParallelCalcForcesAndEnergyKernel::initialize(const System& system) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system);
    for (int i = 0; i < (int) contextNonbondedFractions.size(); i++)
        contextNonbondedFractions[i] = 1/(double) contextNonbondedFractions.size();
}

void MetalParallelCalcForcesAndEnergyKernel::beginComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups) {
    MetalContext& cl0 = *data.contexts[0];
    int elementSize = (cl0.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
    if (!contextForces.isInitialized()) {
        contextForces.initialize<mm_float4>(cl0, &cl0.getForceBuffers().getDeviceBuffer(),
                data.contexts.size()*cl0.getPaddedNumAtoms(), "contextForces");
        int bufferBytes = (data.contexts.size()-1)*cl0.getPaddedNumAtoms()*elementSize;
        pinnedPositionBuffer = new cl::Buffer(cl0.getContext(), CL_MEM_ALLOC_HOST_PTR, bufferBytes);
        pinnedPositionMemory = cl0.getQueue().enqueueMapBuffer(*pinnedPositionBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bufferBytes);
        pinnedForceBuffer = new cl::Buffer(cl0.getContext(), CL_MEM_ALLOC_HOST_PTR, bufferBytes);
        pinnedForceMemory = cl0.getQueue().enqueueMapBuffer(*pinnedForceBuffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, bufferBytes);
    }

    // Copy coordinates over to each device and execute the kernel.
    
    cl0.getQueue().enqueueReadBuffer(cl0.getPosq().getDeviceBuffer(), CL_TRUE, 0, cl0.getPaddedNumAtoms()*elementSize, pinnedPositionMemory);
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        data.contextEnergy[i] = 0.0;
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new BeginComputationTask(context, cl, getKernel(i), includeForce, includeEnergy, groups, pinnedPositionMemory, tileCounts[i]));
    }
}

double MetalParallelCalcForcesAndEnergyKernel::finishComputation(ContextImpl& context, bool includeForce, bool includeEnergy, int groups, bool& valid) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new FinishComputationTask(context, cl, getKernel(i), includeForce, includeEnergy, groups, data.contextEnergy[i], completionTimes[i], pinnedForceMemory, valid, tileCounts[i]));
    }
    data.syncContexts();
    double energy = 0.0;
    for (int i = 0; i < (int) data.contextEnergy.size(); i++)
        energy += data.contextEnergy[i];
    if (includeForce && valid) {
        // Sum the forces from all devices.
        
        MetalContext& cl = *data.contexts[0];
        int numAtoms = cl.getPaddedNumAtoms();
        int elementSize = (cl.getUseDoublePrecision() ? sizeof(mm_double4) : sizeof(mm_float4));
        cl.getQueue().enqueueWriteBuffer(contextForces.getDeviceBuffer(), CL_FALSE, numAtoms*elementSize,
                numAtoms*(data.contexts.size()-1)*elementSize, pinnedForceMemory);
        cl.reduceBuffer(contextForces, cl.getLongForceBuffer(), data.contexts.size());
        
        // Balance work between the contexts by transferring a little nonbonded work from the context that
        // finished last to the one that finished first.
        
        if (cl.getComputeForceCount() < 200) {
            int firstIndex = 0, lastIndex = 0;
            for (int i = 0; i < (int) completionTimes.size(); i++) {
                if (completionTimes[i] < completionTimes[firstIndex])
                    firstIndex = i;
                if (completionTimes[i] > completionTimes[lastIndex])
                    lastIndex = i;
            }
            double fractionToTransfer = min(0.001, contextNonbondedFractions[lastIndex]);
            contextNonbondedFractions[firstIndex] += fractionToTransfer;
            contextNonbondedFractions[lastIndex] -= fractionToTransfer;
            double startFraction = 0.0;
            for (int i = 0; i < (int) contextNonbondedFractions.size(); i++) {
                double endFraction = startFraction+contextNonbondedFractions[i];
                if (i == contextNonbondedFractions.size()-1)
                    endFraction = 1.0; // Avoid roundoff error
                data.contexts[i]->getNonbondedUtilities().setAtomBlockRange(startFraction, endFraction);
                startFraction = endFraction;
            }
        }
    }
    return energy;
}

class MetalParallelCalcHarmonicBondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcHarmonicBondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcHarmonicBondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcHarmonicBondForceKernel::MetalParallelCalcHarmonicBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcHarmonicBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcHarmonicBondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcHarmonicBondForceKernel::initialize(const System& system, const HarmonicBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcHarmonicBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcHarmonicBondForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomBondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomBondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomBondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomBondForceKernel::MetalParallelCalcCustomBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomBondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomBondForceKernel::initialize(const System& system, const CustomBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcHarmonicAngleForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcHarmonicAngleForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcHarmonicAngleForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcHarmonicAngleForceKernel::MetalParallelCalcHarmonicAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcHarmonicAngleForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcHarmonicAngleForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcHarmonicAngleForceKernel::initialize(const System& system, const HarmonicAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcHarmonicAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcHarmonicAngleForceKernel::copyParametersToContext(ContextImpl& context, const HarmonicAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomAngleForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomAngleForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomAngleForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomAngleForceKernel::MetalParallelCalcCustomAngleForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomAngleForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomAngleForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomAngleForceKernel::initialize(const System& system, const CustomAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomAngleForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomAngleForceKernel::copyParametersToContext(ContextImpl& context, const CustomAngleForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcPeriodicTorsionForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcPeriodicTorsionForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcPeriodicTorsionForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcPeriodicTorsionForceKernel::MetalParallelCalcPeriodicTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcPeriodicTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcPeriodicTorsionForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcPeriodicTorsionForceKernel::initialize(const System& system, const PeriodicTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcPeriodicTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcPeriodicTorsionForceKernel::copyParametersToContext(ContextImpl& context, const PeriodicTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcRBTorsionForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcRBTorsionForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcRBTorsionForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcRBTorsionForceKernel::MetalParallelCalcRBTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcRBTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcRBTorsionForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcRBTorsionForceKernel::initialize(const System& system, const RBTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcRBTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcRBTorsionForceKernel::copyParametersToContext(ContextImpl& context, const RBTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCMAPTorsionForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCMAPTorsionForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCMAPTorsionForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCMAPTorsionForceKernel::MetalParallelCalcCMAPTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCMAPTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCMAPTorsionForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCMAPTorsionForceKernel::initialize(const System& system, const CMAPTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCMAPTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCMAPTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CMAPTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomTorsionForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomTorsionForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomTorsionForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomTorsionForceKernel::MetalParallelCalcCustomTorsionForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomTorsionForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomTorsionForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomTorsionForceKernel::initialize(const System& system, const CustomTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomTorsionForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomTorsionForceKernel::copyParametersToContext(ContextImpl& context, const CustomTorsionForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcNonbondedForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, MetalCalcNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, bool includeDirect, bool includeReciprocal, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), includeDirect(includeDirect), includeReciprocal(includeReciprocal), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy, includeDirect, includeReciprocal);
    }
private:
    ContextImpl& context;
    MetalCalcNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy, includeDirect, includeReciprocal;
    double& energy;
};

MetalParallelCalcNonbondedForceKernel::MetalParallelCalcNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new MetalCalcNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcNonbondedForceKernel::initialize(const System& system, const NonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, includeDirect, includeReciprocal, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const NonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

void MetalParallelCalcNonbondedForceKernel::getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getPMEParameters(alpha, nx, ny, nz);
}

void MetalParallelCalcNonbondedForceKernel::getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const {
    dynamic_cast<const MetalCalcNonbondedForceKernel&>(kernels[0].getImpl()).getLJPMEParameters(alpha, nx, ny, nz);
}

class MetalParallelCalcCustomNonbondedForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomNonbondedForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomNonbondedForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomNonbondedForceKernel::MetalParallelCalcCustomNonbondedForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomNonbondedForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomNonbondedForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomNonbondedForceKernel::initialize(const System& system, const CustomNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const CustomNonbondedForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomExternalForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomExternalForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomExternalForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomExternalForceKernel::MetalParallelCalcCustomExternalForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomExternalForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomExternalForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomExternalForceKernel::initialize(const System& system, const CustomExternalForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomExternalForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomExternalForceKernel::copyParametersToContext(ContextImpl& context, const CustomExternalForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomHbondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomHbondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomHbondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomHbondForceKernel::MetalParallelCalcCustomHbondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomHbondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomHbondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomHbondForceKernel::initialize(const System& system, const CustomHbondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomHbondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomHbondForceKernel::copyParametersToContext(ContextImpl& context, const CustomHbondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}

class MetalParallelCalcCustomCompoundBondForceKernel::Task : public MetalContext::WorkTask {
public:
    Task(ContextImpl& context, CommonCalcCustomCompoundBondForceKernel& kernel, bool includeForce,
            bool includeEnergy, double& energy) : context(context), kernel(kernel),
            includeForce(includeForce), includeEnergy(includeEnergy), energy(energy) {
    }
    void execute() {
        energy += kernel.execute(context, includeForce, includeEnergy);
    }
private:
    ContextImpl& context;
    CommonCalcCustomCompoundBondForceKernel& kernel;
    bool includeForce, includeEnergy;
    double& energy;
};

MetalParallelCalcCustomCompoundBondForceKernel::MetalParallelCalcCustomCompoundBondForceKernel(std::string name, const Platform& platform, MetalPlatform::PlatformData& data, const System& system) :
        CalcCustomCompoundBondForceKernel(name, platform), data(data) {
    for (int i = 0; i < (int) data.contexts.size(); i++)
        kernels.push_back(Kernel(new CommonCalcCustomCompoundBondForceKernel(name, platform, *data.contexts[i], system)));
}

void MetalParallelCalcCustomCompoundBondForceKernel::initialize(const System& system, const CustomCompoundBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).initialize(system, force);
}

double MetalParallelCalcCustomCompoundBondForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    for (int i = 0; i < (int) data.contexts.size(); i++) {
        MetalContext& cl = *data.contexts[i];
        ComputeContext::WorkThread& thread = cl.getWorkThread();
        thread.addTask(new Task(context, getKernel(i), includeForces, includeEnergy, data.contextEnergy[i]));
    }
    return 0.0;
}

void MetalParallelCalcCustomCompoundBondForceKernel::copyParametersToContext(ContextImpl& context, const CustomCompoundBondForce& force) {
    for (int i = 0; i < (int) kernels.size(); i++)
        getKernel(i).copyParametersToContext(context, force);
}
