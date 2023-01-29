#---------------------------------------------------
# OpenMM Metal Platform
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMMetalPlugin)

ENABLE_TESTING()

# It seems that on linux and mac, everything is trying to be installed in /usr/local/openmm
# But if every install target is prefixed with /openmm/, on Windows the install files
# end up in C:/Program Files/OpenMM/openmm/ which is ugly.
# Better to set CMAKE_INSTALL_PREFIX to /usr/local/openmm and leave /openmm/ off the
# install target names.  Plus, the user now has the opportunity to install into /usr/local/
# if she so chooses.  --cmb
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
  SET(CMAKE_INSTALL_PREFIX ${OPENMM_INSTALL_PREFIX} CACHE PATH "Where to install OpenMM" FORCE)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")

# We need to know where OpenMM source code is located so we can access required files.
SET(OPENMM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../openmm" CACHE PATH "Where OpenMM source code is located")








### OPENCL_INCLUDE_DIR ###
# Try OPENCL_DIR variable before looking elsewhere
find_path(OPENCL_INCLUDE_DIR
    NAMES OpenCL/opencl.h CL/opencl.h
    PATHS $ENV{OPENCL_DIR}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)
# Next look in environment variables set by OpenCL SDK installations
find_path(OPENCL_INCLUDE_DIR
    NAMES OpenCL/opencl.h CL/opencl.h
    PATHS
        $ENV{CUDA_PATH}
        $ENV{AMDAPPSDKROOT}
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)
# On Macs, look inside the platform SDK
if(DEFINED CMAKE_OSX_SYSROOT)
    find_path(OPENCL_INCLUDE_DIR
        NAMES opencl.h opencl.h
        PATHS
            "${CMAKE_OSX_SYSROOT}/System/Library/Frameworks/OpenCL.framework/Headers"
        NO_DEFAULT_PATH
    )
endif(DEFINED CMAKE_OSX_SYSROOT)
# As a last resort, look in default system areas followed by other possible locations
find_path(OPENCL_INCLUDE_DIR
    NAMES OpenCL/opencl.h CL/opencl.h
    PATHS
        "C:/CUDA"
        "/usr/local/cuda"
        "/usr/local/streamsdk"
        "/usr"
        "${CUDA_TOOLKIT_ROOT_DIR}"
    PATH_SUFFIXES "include"
)

### OPENCL_LIBRARY ###
if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(path_suffixes "lib/x86_64")
    else("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
      set(path_suffixes "lib/x86")
    endif("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
elseif(MSVC)
    if(CMAKE_CL_64)
      set(path_suffixes "lib/x64" "lib/x86_64")
    else(CMAKE_CL_64)
      set(path_suffixes "lib/Win32" "lib/x86")
    endif(CMAKE_CL_64)
else(MSVC)
    set(path_suffixes "lib")
endif("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
# Try OPENCL_DIR variable before looking elsewhere
find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
      $ENV{OPENCL_DIR}
      ${OPENCL_LIB_SEARCH_PATH}
    PATH_SUFFIXES ${path_suffixes}
    NO_DEFAULT_PATH
)
# Next look in environment variables set by OpenCL SDK installations
find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
      $ENV{CUDA_PATH}
      $ENV{AMDAPPSDKROOT}
    PATH_SUFFIXES ${path_suffixes}
    NO_DEFAULT_PATH
)
# As a last resort, look in default system areas followed by other possible locations
find_library(OPENCL_LIBRARY
    NAMES OpenCL
    PATHS
        "C:/CUDA"
        "/usr/local/cuda"
        "/usr/local/streamsdk"
        "/usr"
        "${CUDA_TOOLKIT_ROOT_DIR}"
    PATH_SUFFIXES ${path_suffixes} "lib"
)

if(OPENCL_FOUND)
  set(OPENCL_LIBRARIES ${OPENCL_LIBRARY})
  mark_as_advanced(CLEAR OPENCL_INCLUDE_DIR)
  mark_as_advanced(CLEAR OPENCL_LIBRARY)
else(OPENCL_FOUND)
  set(OPENCL_LIBRARIES)
  mark_as_advanced(OPENCL_INCLUDE_DIR)
  mark_as_advanced(OPENCL_LIBRARY)
endif(OPENCL_FOUND)





















# Determine what architecture we are compiling for.
INCLUDE(${OPENMM_SOURCE_DIR}/cmake_modules/TargetArch.cmake)
target_architecture(TARGET_ARCH)
if ("${TARGET_ARCH}" MATCHES "x86_64|i386")
    set(X86 ON)
endif()
if ("${TARGET_ARCH}" MATCHES "arm")
    set(ARM ON)
    if ("${TARGET_ARCH}" MATCHES "armv8")
        add_definitions(-D__ARM64__=1)
    else()
        add_definitions(-D__ARM__=1)
    endif()
endif()
if ("${TARGET_ARCH}" MATCHES "ppc")
    set(PPC ON)
    add_definitions(-D__PPC__=1)
endif()

INCLUDE_DIRECTORIES(${OPENMM_DIR}/include)
INCLUDE_DIRECTORIES(${OPENMM_DIR}/include/openmm/reference)
LINK_DIRECTORIES("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")

SET(PTHREADS_LIB pthread)

# Specify the C++ version we are building for.
SET(CMAKE_CXX_STANDARD 11)

# Build 64 bit binaries compatible with OS X 12.0
IF (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
    SET (CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "The minimum version of OS X to support" FORCE)
ENDIF (NOT CMAKE_OSX_DEPLOYMENT_TARGET)
IF (NOT CMAKE_OSX_ARCHITECTURES)
    SET (CMAKE_OSX_ARCHITECTURES "${CMAKE_SYSTEM_PROCESSOR}" CACHE STRING "The processor architectures to build for" FORCE)
ENDIF (NOT CMAKE_OSX_ARCHITECTURES)
IF (NOT CMAKE_OSX_SYSROOT)
    EXECUTE_PROCESS(COMMAND "xcrun" "--show-sdk-path" OUTPUT_VARIABLE XCRUN_OSX_SYSROOT RESULT_VARIABLE XCRUN_OSX_SYSROOT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE)
    IF (XCRUN_OSX_SYSROOT_STATUS EQUAL 0)
        SET (CMAKE_OSX_SYSROOT "${XCRUN_OSX_SYSROOT}" CACHE STRING "SDK Path" FORCE)
    ENDIF (XCRUN_OSX_SYSROOT_STATUS EQUAL 0)
    UNSET(XCRUN_OSX_SYSROOT)
    UNSET(XCRUN_OSX_SYSROOT_STATUS)
ENDIF (NOT CMAKE_OSX_SYSROOT)

# Improve the linking behavior of Mac libraries
SET (CMAKE_INSTALL_NAME_DIR "@rpath")
IF (X86)
  SET(EXTRA_COMPILE_FLAGS "-msse2")
ENDIF()
SET(EXTRA_COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -stdlib=libc++")
SET(EXTRA_LINK_FLAGS ${EXTRA_COMPILE_FLAGS})

# What type of libraries to build
SET(OPENMM_BUILD_SHARED_LIB ON)
SET(OPENMM_BUILD_STATIC_LIB OFF)

SET(OPENMM_LIBRARY_NAME OpenMM)

IF(NOT EXECUTABLE_OUTPUT_PATH)
  SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
      CACHE INTERNAL "Single output directory for building all executables.")
ENDIF(NOT EXECUTABLE_OUTPUT_PATH)

# Include OpenMM utiliity libraries
SET(OPENMM_LIB_SUBDIRS libraries/jama libraries/quern libraries/hilbert libraries/csha1 libraries/lbfgs libraries/pocketfft)
FOREACH(subdir ${OPENMM_LIB_SUBDIRS})
    INCLUDE_DIRECTORIES(BEFORE ${OPENMM_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

ADD_SUBDIRECTORY(platforms/metal)

# Amoeba plugin

SET(OPENMM_BUILD_AMOEBA_PLUGIN ON CACHE BOOL "Build Amoeba plugin")
IF(OPENMM_BUILD_AMOEBA_PLUGIN)
    ADD_SUBDIRECTORY(plugins/amoeba/platforms/metal)
ENDIF(OPENMM_BUILD_AMOEBA_PLUGIN)

# RPMD plugin

SET(OPENMM_BUILD_RPMD_PLUGIN ON CACHE BOOL "Build RPMD plugin")
IF(OPENMM_BUILD_RPMD_PLUGIN)
   ADD_SUBDIRECTORY(plugins/rpmd/platforms/metal)
ENDIF(OPENMM_BUILD_RPMD_PLUGIN)

# Drude plugin

SET(OPENMM_BUILD_DRUDE_PLUGIN ON CACHE BOOL "Build Drude plugin")
IF(OPENMM_BUILD_DRUDE_PLUGIN)
   ADD_SUBDIRECTORY(plugins/drude/platforms/metal)
ENDIF(OPENMM_BUILD_DRUDE_PLUGIN)
