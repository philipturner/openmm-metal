#---------------------------------------------------
# OpenMM Metal Platform
#
# Creates OpenMMMetal library.
#
# macOS:
#   libOpenMMMetal.dylib
#----------------------------------------------------

set(OPENMM_BUILD_OPENCL_TESTS TRUE CACHE BOOL "Whether to build Metal test cases")
if(OPENMM_BUILD_OPENCL_TESTS)
    SUBDIRS (tests)
    if(OPENMM_BUILD_LONG_TESTS)
      SUBDIRS(long_tests)
      if(OPENMM_BUILD_VERY_LONG_TESTS)
        SUBDIRS(very_long_tests)
      endif(OPENMM_BUILD_VERY_LONG_TESTS)
    endif(OPENMM_BUILD_LONG_TESTS)
endif(OPENMM_BUILD_OPENCL_TESTS)

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(OPENMM_SOURCE_SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR} ${OPENMM_SOURCE_DIR}/platforms/common)


# Collect up information about the version of the OpenMM library we're building
# and make it available to the code so it can be built into the binaries.

SET(OPENMMOPENCL_LIBRARY_NAME OpenMMMetal)

SET(SHARED_TARGET ${OPENMMOPENCL_LIBRARY_NAME})

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

# collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)

FOREACH(subdir ${OPENMM_SOURCE_SUBDIRS})
    FILE(GLOB_RECURSE src_files  ${subdir}/src/*.cpp ${subdir}/src/*.c)
    FILE(GLOB incl_files ${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    IF(MSVC)
         FILE(GLOB_RECURSE kernel_files ${subdir}/src/kernels/*.cl)
         SET(SOURCE_FILES ${SOURCE_FILES} ${kernel_files})
    ENDIF(MSVC)
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})
    INCLUDE_DIRECTORIES(BEFORE ${subdir}/include)
ENDFOREACH(subdir)

INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/src)

# Encode the common kernel sources into a C++ class

SET(KERNEL_SOURCE_DIR ${OPENMM_SOURCE_DIR}/platforms/common/src)
SET(KERNEL_SOURCE_CLASS CommonKernelSources)
SET(KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.cpp)
SET(KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.h)
FILE(GLOB COMMON_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cc)

ADD_CUSTOM_COMMAND(OUTPUT ${KERNELS_CPP} ${KERNELS_H}
    COMMAND ${CMAKE_COMMAND}
    ARGS -D KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR} -D KERNELS_CPP=${KERNELS_CPP} -D KERNELS_H=${KERNELS_H} -D KERNEL_SOURCE_CLASS=${KERNEL_SOURCE_CLASS} -D KERNEL_FILE_EXTENSION=cc -P ${OPENMM_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake
    DEPENDS ${COMMON_KERNELS}
)
SET_SOURCE_FILES_PROPERTIES(${KERNELS_CPP} ${KERNELS_H} PROPERTIES GENERATED TRUE)
ADD_CUSTOM_TARGET(CommonKernels DEPENDS ${KERNELS_CPP} ${KERNELS_H})

# Encode the kernel sources into a C++ class

SET(KERNEL_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
SET(KERNEL_SOURCE_CLASS MetalKernelSources)
SET(KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.cpp)
SET(KERNELS_H ${CMAKE_CURRENT_BINARY_DIR}/src/${KERNEL_SOURCE_CLASS}.h)
FILE(GLOB OPENCL_KERNELS ${KERNEL_SOURCE_DIR}/kernels/*.cl)
ADD_CUSTOM_COMMAND(OUTPUT ${KERNELS_CPP} ${KERNELS_H}
    COMMAND ${CMAKE_COMMAND}
    ARGS -D KERNEL_SOURCE_DIR=${KERNEL_SOURCE_DIR} -D KERNELS_CPP=${KERNELS_CPP} -D KERNELS_H=${KERNELS_H} -D KERNEL_SOURCE_CLASS=${KERNEL_SOURCE_CLASS} -D KERNEL_FILE_EXTENSION=cl -P ${OPENMM_SOURCE_DIR}/cmake_modules/EncodeKernelFiles.cmake
    DEPENDS ${OPENCL_KERNELS}
)
SET_SOURCE_FILES_PROPERTIES(${KERNELS_CPP} ${KERNELS_H} PROPERTIES GENERATED TRUE)
ADD_CUSTOM_TARGET(MetalKernels DEPENDS ${KERNELS_CPP} ${KERNELS_H})

SET(COMMON_KERNELS_CPP ${CMAKE_CURRENT_BINARY_DIR}/src/CommonKernelSources.cpp)
SET(SOURCE_FILES ${SOURCE_FILES} ${KERNELS_CPP} ${KERNELS_H} ${COMMON_KERNELS_CPP})
INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_BINARY_DIR}/src)

# Build the shared library.

ADD_LIBRARY(${SHARED_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_ABS_INCLUDE_FILES})
ADD_DEPENDENCIES(${SHARED_TARGET} CommonKernels)

TARGET_LINK_LIBRARIES(${SHARED_TARGET} ${OPENMM_LIBRARY_NAME}  ${OPENCL_LIBRARIES} ${PTHREADS_LIB})
SET_TARGET_PROPERTIES(${SHARED_TARGET} PROPERTIES LINK_FLAGS "${EXTRA_LINK_FLAGS}" COMPILE_FLAGS "${EXTRA_COMPILE_FLAGS} -DOPENMM_COMMON_BUILDING_SHARED_LIBRARY")

INSTALL_TARGETS(/lib/plugins RUNTIME_DIRECTORY /lib/plugins ${SHARED_TARGET})

# Install headers

FILE(GLOB CORE_HEADERS include/*.h src/opencl.hpp ${KERNELS_H})
INSTALL_FILES(/include/openmm/metal FILES ${CORE_HEADERS})
