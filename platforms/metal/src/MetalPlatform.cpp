/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2018 Stanford University and the Authors.      *
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

#include "MetalContext.h"
#include "MetalPlatform.h"
#include "MetalKernelFactory.h"
#include "MetalKernels.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/hardware.h"
#include <algorithm>
#include <cctype>
#include <sstream>
#ifdef __APPLE__
#include "sys/sysctl.h"
#endif


using namespace OpenMM;
using namespace std;

#ifdef OPENMM_COMMON_BUILDING_STATIC_LIBRARY
extern "C" void registerMetalPlatform() {
    if (MetalPlatform::isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}
#else
extern "C" OPENMM_EXPORT_COMMON void registerPlatforms() {
    if (MetalPlatform::isPlatformSupported())
        Platform::registerPlatform(new MetalPlatform());
}
#endif

MetalPlatform::MetalPlatform() {
    deprecatedPropertyReplacements["MetalDeviceIndex"] = MetalDeviceIndex();
    deprecatedPropertyReplacements["MetalDeviceName"] = MetalDeviceName();
    deprecatedPropertyReplacements["MetalPrecision"] = MetalPrecision();
    deprecatedPropertyReplacements["MetalUseCpuPme"] = MetalUseCpuPme();
    deprecatedPropertyReplacements["MetalDisablePmeStream"] = MetalDisablePmeStream();
    MetalKernelFactory* factory = new MetalKernelFactory();
    registerKernelFactory(CalcForcesAndEnergyKernel::Name(), factory);
    registerKernelFactory(UpdateStateDataKernel::Name(), factory);
    registerKernelFactory(ApplyConstraintsKernel::Name(), factory);
    registerKernelFactory(VirtualSitesKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomBondForceKernel::Name(), factory);
    registerKernelFactory(CalcHarmonicAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomAngleForceKernel::Name(), factory);
    registerKernelFactory(CalcPeriodicTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcRBTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCMAPTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomTorsionForceKernel::Name(), factory);
    registerKernelFactory(CalcNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomNonbondedForceKernel::Name(), factory);
    registerKernelFactory(CalcGBSAOBCForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomGBForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomExternalForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomHbondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCentroidBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCompoundBondForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomCVForceKernel::Name(), factory);
    registerKernelFactory(CalcRMSDForceKernel::Name(), factory);
    registerKernelFactory(CalcCustomManyParticleForceKernel::Name(), factory);
    registerKernelFactory(CalcGayBerneForceKernel::Name(), factory);
    registerKernelFactory(IntegrateVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateNoseHooverStepKernel::Name(), factory);
    registerKernelFactory(IntegrateLangevinStepKernel::Name(), factory);
    registerKernelFactory(IntegrateLangevinMiddleStepKernel::Name(), factory);
    registerKernelFactory(IntegrateBrownianStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableVerletStepKernel::Name(), factory);
    registerKernelFactory(IntegrateVariableLangevinStepKernel::Name(), factory);
    registerKernelFactory(IntegrateCustomStepKernel::Name(), factory);
    registerKernelFactory(ApplyAndersenThermostatKernel::Name(), factory);
    registerKernelFactory(ApplyMonteCarloBarostatKernel::Name(), factory);
    registerKernelFactory(RemoveCMMotionKernel::Name(), factory);
    platformProperties.push_back(MetalDeviceIndex());
    platformProperties.push_back(MetalDeviceName());
    platformProperties.push_back(MetalPlatformIndex());
    platformProperties.push_back(MetalPlatformName());
    platformProperties.push_back(MetalPrecision());
    platformProperties.push_back(MetalUseCpuPme());
    platformProperties.push_back(MetalDisablePmeStream());
    setPropertyDefaultValue(MetalDeviceIndex(), "");
    setPropertyDefaultValue(MetalDeviceName(), "");
    setPropertyDefaultValue(MetalPlatformIndex(), "");
    setPropertyDefaultValue(MetalPlatformName(), "");
    setPropertyDefaultValue(MetalPrecision(), "single");
    setPropertyDefaultValue(MetalUseCpuPme(), "false");
    setPropertyDefaultValue(MetalDisablePmeStream(), "false");
}

double MetalPlatform::getSpeed() const {
    return 50;
}

bool MetalPlatform::supportsDoublePrecision() const {
    return true;
}

bool MetalPlatform::isPlatformSupported() {
    // Return false for Metal implementations that are known
    // to be buggy (Apple OS X prior to 10.10).

#ifdef __APPLE__
    char str[256];
    size_t size = sizeof(str);
    int ret = sysctlbyname("kern.osrelease", str, &size, NULL, 0);
    if (ret != 0)
        return false;

    int major, minor, micro;
    if (sscanf(str, "%d.%d.%d", &major, &minor, &micro) != 3)
        return false;

    if (major < 14 || (major == 14 && minor < 3))
        // 14.3.0 is the darwin release corresponding to OS X 10.10.3. Versions prior to that
        // contained a number of serious bugs in the Apple Metal libraries.
        // (See https://github.com/SimTk/openmm/issues/395 for example.)
        return false;
#endif

    // Make sure at least one Metal implementation is installed.

    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
        if (platforms.size() == 0)
            return false;
    }
    catch (...) {
        return false;
    }
    return true;
}

const string& MetalPlatform::getPropertyValue(const Context& context, const string& property) const {
    const ContextImpl& impl = getContextImpl(context);
    const PlatformData* data = reinterpret_cast<const PlatformData*>(impl.getPlatformData());
    string propertyName = property;
    if (deprecatedPropertyReplacements.find(property) != deprecatedPropertyReplacements.end())
        propertyName = deprecatedPropertyReplacements.find(property)->second;
    map<string, string>::const_iterator value = data->propertyValues.find(propertyName);
    if (value != data->propertyValues.end())
        return value->second;
    return Platform::getPropertyValue(context, property);
}

void MetalPlatform::setPropertyValue(Context& context, const string& property, const string& value) const {
}

void MetalPlatform::contextCreated(ContextImpl& context, const map<string, string>& properties) const {
    const string& platformPropValue = (properties.find(MetalPlatformIndex()) == properties.end() ?
            getPropertyDefaultValue(MetalPlatformIndex()) : properties.find(MetalPlatformIndex())->second);
    const string& devicePropValue = (properties.find(MetalDeviceIndex()) == properties.end() ?
            getPropertyDefaultValue(MetalDeviceIndex()) : properties.find(MetalDeviceIndex())->second);
    string precisionPropValue = (properties.find(MetalPrecision()) == properties.end() ?
            getPropertyDefaultValue(MetalPrecision()) : properties.find(MetalPrecision())->second);
    string cpuPmePropValue = (properties.find(MetalUseCpuPme()) == properties.end() ?
            getPropertyDefaultValue(MetalUseCpuPme()) : properties.find(MetalUseCpuPme())->second);
    string pmeStreamPropValue = (properties.find(MetalDisablePmeStream()) == properties.end() ?
            getPropertyDefaultValue(MetalDisablePmeStream()) : properties.find(MetalDisablePmeStream())->second);
    transform(precisionPropValue.begin(), precisionPropValue.end(), precisionPropValue.begin(), ::tolower);
    transform(cpuPmePropValue.begin(), cpuPmePropValue.end(), cpuPmePropValue.begin(), ::tolower);
    transform(pmeStreamPropValue.begin(), pmeStreamPropValue.end(), pmeStreamPropValue.begin(), ::tolower);
    vector<string> pmeKernelName;
    pmeKernelName.push_back(CalcPmeReciprocalForceKernel::Name());
    if (!supportsKernels(pmeKernelName))
        cpuPmePropValue = "false";
    int threads = getNumProcessors();
    char* threadsEnv = getenv("OPENMM_CPU_THREADS");
    if (threadsEnv != NULL)
        stringstream(threadsEnv) >> threads;
    context.setPlatformData(new PlatformData(context.getSystem(), platformPropValue, devicePropValue, precisionPropValue, cpuPmePropValue,
            pmeStreamPropValue, threads, NULL));
}

void MetalPlatform::linkedContextCreated(ContextImpl& context, ContextImpl& originalContext) const {
    Platform& platform = originalContext.getPlatform();
    string platformPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalPlatformIndex());
    string devicePropValue = platform.getPropertyValue(originalContext.getOwner(), MetalDeviceIndex());
    string precisionPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalPrecision());
    string cpuPmePropValue = platform.getPropertyValue(originalContext.getOwner(), MetalUseCpuPme());
    string pmeStreamPropValue = platform.getPropertyValue(originalContext.getOwner(), MetalDisablePmeStream());
    int threads = reinterpret_cast<PlatformData*>(originalContext.getPlatformData())->threads.getNumThreads();
    context.setPlatformData(new PlatformData(context.getSystem(), platformPropValue, devicePropValue, precisionPropValue, cpuPmePropValue,
            pmeStreamPropValue, threads, &originalContext));
}

void MetalPlatform::contextDestroyed(ContextImpl& context) const {
    PlatformData* data = reinterpret_cast<PlatformData*>(context.getPlatformData());
    delete data;
}

MetalPlatform::PlatformData::PlatformData(const System& system, const string& platformPropValue, const string& deviceIndexProperty,
        const string& precisionProperty, const string& cpuPmeProperty, const string& pmeStreamProperty, int numThreads, ContextImpl* originalContext) :
            removeCM(false), stepCount(0), computeForceCount(0), time(0.0), hasInitializedContexts(false), threads(numThreads)  {
    int platformIndex = -1;
    if (platformPropValue.length() > 0)
        stringstream(platformPropValue) >> platformIndex;
    vector<string> devices;
    size_t searchPos = 0, nextPos;
    while ((nextPos = deviceIndexProperty.find_first_of(", ", searchPos)) != string::npos) {
        devices.push_back(deviceIndexProperty.substr(searchPos, nextPos-searchPos));
        searchPos = nextPos+1;
    }
    devices.push_back(deviceIndexProperty.substr(searchPos));
    PlatformData* originalData = NULL;
    if (originalContext != NULL)
        originalData = reinterpret_cast<PlatformData*>(originalContext->getPlatformData());
    try {
        for (int i = 0; i < (int) devices.size(); i++) {
            if (devices[i].length() > 0) {
                int deviceIndex;
                stringstream(devices[i]) >> deviceIndex;
                contexts.push_back(new MetalContext(system, platformIndex, deviceIndex, precisionProperty, *this, (originalData == NULL ? NULL : originalData->contexts[i])));
            }
        }
        if (contexts.size() == 0)
            contexts.push_back(new MetalContext(system, platformIndex, -1, precisionProperty, *this, (originalData == NULL ? NULL : originalData->contexts[0])));
    }
    catch (...) {
        // If an exception was thrown, do our best to clean up memory.
        
        for (int i = 0; i < (int) contexts.size(); i++)
            delete contexts[i];
        throw;
    }
    stringstream deviceIndex, deviceName;
    for (int i = 0; i < (int) contexts.size(); i++) {
        if (i > 0) {
            deviceIndex << ',';
            deviceName << ',';
        }
        deviceIndex << contexts[i]->getDeviceIndex();
        deviceName << contexts[i]->getDevice().getInfo<CL_DEVICE_NAME>();
    }
    platformIndex = contexts[0]->getPlatformIndex();

    useCpuPme = (cpuPmeProperty == "true" && !contexts[0]->getUseDoublePrecision());
    disablePmeStream = (pmeStreamProperty == "true");
    propertyValues[MetalPlatform::MetalDeviceIndex()] = deviceIndex.str();
    propertyValues[MetalPlatform::MetalDeviceName()] = deviceName.str();
    propertyValues[MetalPlatform::MetalPlatformIndex()] = contexts[0]->intToString(platformIndex);
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    propertyValues[MetalPlatform::MetalPlatformName()] = platforms[platformIndex].getInfo<CL_PLATFORM_NAME>();
    propertyValues[MetalPlatform::MetalPrecision()] = precisionProperty;
    propertyValues[MetalPlatform::MetalUseCpuPme()] = useCpuPme ? "true" : "false";
    propertyValues[MetalPlatform::MetalDisablePmeStream()] = disablePmeStream ? "true" : "false";
    contextEnergy.resize(contexts.size());
}

MetalPlatform::PlatformData::~PlatformData() {
    for (int i = 0; i < (int) contexts.size(); i++)
        delete contexts[i];
}

void MetalPlatform::PlatformData::initializeContexts(const System& system) {
    if (hasInitializedContexts)
        return;
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->initialize();
    hasInitializedContexts = true;
}

void MetalPlatform::PlatformData::syncContexts() {
    for (int i = 0; i < (int) contexts.size(); i++)
        contexts[i]->getWorkThread().flush();
}
