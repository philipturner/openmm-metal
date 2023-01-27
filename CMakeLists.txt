#---------------------------------------------------
# OpenMM Metal Platform
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.17)

PROJECT(OpenMMMetalPlugin)

ENABLE_TESTING()

set(CMAKE_INSTALL_PREFIX /usr/local/openmm)

# We need to know where OpenMM is installed so we can access the headers and libraries.
SET(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")

# We need to know where OpenMM source code is located so we can access required files.
SET(OPENMM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../openmm" CACHE PATH "Where OpenMM source code is located")

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

# What type of libraries to build
SET(OPENMM_BUILD_SHARED_LIB ON)
SET(OPENMM_BUILD_STATIC_LIB OFF)

SET(OPENMM_LIBRARY_NAME OpenMM)

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}
  CACHE INTERNAL "Single output directory for building all executables.")
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}
  CACHE INTERNAL "Single output directory for building all libraries.")

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
