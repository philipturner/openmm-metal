#---------------------------------------------------
# OpenMM Metal Amoeba Implementation
#
# Creates OpenMMAmoebaMetal library.
#
# macOS:
#   libOpenMMAmoebaMetal.dylib
#----------------------------------------------------

SET(OPENMM_AMOEBA_LIBRARY_NAME OpenMMAmoeba)
SET(SHARED_AMOEBA_TARGET ${OPENMM_AMOEBA_LIBRARY_NAME})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(OPENMM_SOURCE_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} ${OPENMM_SOURCE_DIR}/plugins/amoeba/platforms/common)


# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMMAMOEBAOPENCL_LIBRARY_NAME OpenMMAmoebaMetal)

SET(SHARED_TARGET ${OPENMMAMOEBAOPENCL_LIBRARY_NAME})


# These are all the places to search for header files which are
# to be part of the API.
SET(API_INCLUDE_DIRS) # start empty
FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    # append
    SET(API_INCLUDE_DIRS ${API_INCLUDE_DIRS}
                         ${subdir}/include
                         ${subdir}/include/internal)
ENDFOREACH(subdir)

# We'll need both *relative* path names, starting with their API_INCLUDE_DIRS,
# and absolute pathnames.
SET(API_REL_INCLUDE_FILES)   # start these out empty
SET(API_ABS_INCLUDE_FILES)

FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)	# returns full pathnames
    SET(API_ABS_INCLUDE_FILES ${API_ABS_INCLUDE_FILES} ${fullpaths})

    FOREACH(pathname ${fullpaths})
        GET_FILENAME_COMPONENT(filename ${pathname} NAME)
        SET(API_REL_INCLUDE_FILES ${API_REL_INCLUDE_FILES} ${dir}/${filename})
    ENDFOREACH(pathname)
ENDFOREACH(dir)

# Encode the common kernel sources into a C++ class.

SET(KERNEL_SOURCE_DIR ${OPENMM_SOURCE_DIR}/plugins/amoeba/platforms/common/src)
SET(KERNEL_SOURCE_CLASS CommonAmoebaKernelSources)
SET(KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.cpp)
SET(KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.h)
FILE(GLOB COMMON_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cc)
ADD_CUSTOM_COMMAND(OUTPUT ${KERNELS_CPP} ${KERNELS_H}
    COMMAND ${CMAKE_COMMAND}
    ARGS -D KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR} -D KERNELS_CPP=${KERNELS_CPP} -D KERNELS_H=${KERNELS_H} -D KERNEL_SOURCE_CLASS=${KERNEL_SOURCE_CLASS} -D KERNEL_FILE_EXTENSION=cc -P ${OPENMM_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake
    DEPENDS ${COMMON_KERNELS}
)
SET_SOURCE_FILES_PROPERTIES(${KERNELS_CPP} ${KERNELS_H} PROPERTIES GENERATED TRUE)
ADD_CUSTOM_TARGET(AmoebaCommonKernels DEPENDS ${KERNELS_CPP} ${KERNELS_H})

# collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)

FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    FILE(GLOB_RECURSE src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.c)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

SET(COMMON_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/CommonAmoebaKernelSources.cpp)
SET(SOURCE_FILES ${SOURCE_FILES} ${COMMON_KERNELS_CPP})

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(BEFORE ${OPENMM_SOURCE_DIR}/plugins/amoeba/platforms/common/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_SOURCE_DIR}/platforms/metal/include)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_BINARY_DIR}/platforms/metal/src)
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/src)

# Create the library

# Build the shared plugin library.

ADD_LIBRARY(${SHARED_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_ABS_INCLUDE_FILES})
ADD_DEPENDENCIES(${SHARED_TARGET} AmoebaCommonKernels)

TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME} ${PTHREADS_LIB})
TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}Metal)
TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${SHARED_AMOEBA_TARGET})
SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_BUILDING_SHARED_LIBRARY")

INSTALL_TARGETS(/lib/plugins RUNTIME_DIRECTORY /lib/plugins ${SHARED_TARGET})

INSTALL(TARGETS ${SHARED_TARGET} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/plugins)
# Ensure that links to the main OpenCL library will be resolved.
IF (APPLE)
    SET(OPENCL_LIBRARY libOpenMMMetal.dylib)
    INSTALL(CODE "EXECUTE_PROCESS(COMMAND install_name_tool -change ${OPENCL_LIBRARY} @loader_path/${OPENCL_LIBRARY} ${CMAKE_INSTALL_PREFIX}/lib/plugins/lib${SHARED_TARGET}.dylib)")

ENDIF (APPLE)
if(OPENMM_BUILD_OPENCL_TESTS)
    SUBDIRS (tests)
endif(OPENMM_BUILD_OPENCL_TESTS)
