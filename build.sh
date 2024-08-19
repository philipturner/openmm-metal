#!/bin/bash

# Installing the plugins requires 'sudo' privileges to modify an external
# directory. It will ask for your password.
install_plugins=false

# The tests will only run if `make test` is called from the same script that
# invoked `make install`. If '--install-plugins' is not set, tests will fail.
run_all_tests=false

# Only run tests that give you useful feedback in a reasonable amount of time.
# On the reference system (32-core M1 Max, 32 GB), this means tests that take
# less than ~20 seconds.
run_most_tests=false

# Only run tests that give you useful feedback in a short amount of time.
# On the reference system (32-core M1 Max, 32 GB), this means tests that take
# less than ~5 seconds.
run_quick_tests=false

# TODO: After incorporating mixed precision, re-partition the test suite.

# Very quick tests: ones that routinely fail when something is wrong
# TestMetalCheckpointsSingle
# TestMetalDispersionPMESingle
# TestMetalEwaldSingle
# TestMetalGBSAOBCForceSingle
# Use cmake -R to run these tests, otherwise compiling the same as --quick-tests.

if [[ $# != 0 ]]; then
  invalid_input=false
  if [[ $# -gt 3 ]]; then
    echo "Too many arguments."
    invalid_input=true
  fi
  
  if [[ $invalid_input == false ]]; then
    for param in "$@"; do
      if [[ $param == "--install" ]]; then
        if [[ $install_plugin == true ]]; then
          echo "Duplicate argument '--install'."
          invalid_input=true
        else
          install_plugins=true
        fi
      elif [[ $param == "--all-tests" ]]; then
        if [[ $run_all_tests == true ]]; then
          echo "Duplicate argument '--all-tests'."
          invalid_input=true
        else
          run_all_tests=true
        fi
      elif [[ $param == "--most-tests" ]]; then
        if [[ $run_all_tests == true ]]; then
          echo "Duplicate argument '--most-tests'."
          invalid_input=true
        else
          run_most_tests=true
        fi
      elif [[ $param == "--quick-tests" ]]; then
        if [[ $run_all_tests == true ]]; then
          echo "Duplicate argument '--quick-tests'."
          invalid_input=true
        else
          run_quick_tests=true
        fi
      else
        echo "Unrecognized argument '${param}'."
        invalid_input=true
      fi
    done
  fi
  
  if [[ $invalid_input == true ]]; then
    echo "Usage: build.sh [--install] [--all-tests] [--most-tests] [--quick-tests]"
    exit -1
  fi
fi

if [[ ! -d ".build" ]]; then
    mkdir ".build"
fi
if [[ ! -d ".build" ]]; then
    echo "Malformed .build directory"
    exit -1
fi
cd ".build"
touch "python_messaging.txt"

# Check that OpenMM is installed.
echo '
message = "success"
try:
    import openmm
except ImportError:
    message = "failure"

f = open("python_messaging.txt", "w")
f.write(message)
f.close()
' > "python_script.py"
python python_script.py

if [[ "success" != `cat python_messaging.txt` ]]; then
  # If this fails, try replacing 'python python_script.py' with 'python3
  # python_script.py'. The last one was what worked originally. Now, the first
  # one works and the last one doesn't.
  echo "OpenMM not installed."
  exit -1
fi

# Extract OpenMM version.
echo '
import openmm

f = open("python_messaging.txt", "w")
f.write(openmm.version.version)
f.close()
' > "python_script.py"
python python_script.py

openmm_version=`cat python_messaging.txt`
old_IFS=$IFS
IFS='.'
read -a strarr <<< "$openmm_version"
openmm_version_major=$((strarr[0] + 0)) # Force convert to integer.
openmm_version_minor=$((strarr[1] + 0)) # Force convert to integer.
IFS=$old_IFS

# TODO: Require OpenMM 8.1.0 when the HIP workaround is removed.
if [[ $openmm_version_major -lt 8 ]]; then
  echo "OpenMM must be version 8.0.0 or higher."
  exit -1
fi
if [[ $openmm_version_major == 8 ]]; then
  if [[ $openmm_version_minor -lt 0 ]]; then
    echo "OpenMM must be version 8.0.0 or higher."
    exit -1
  fi
fi

# Extract OpenMM install location.
# WARNING: The machine building this must link against the same location as you
# would expect on client machines. The location seems different on x86_64.
echo '
import openmm
import os

path = openmm.version.openmm_library_path
path = os.path.normpath(path)
path, ignored = os.path.split(path)
path = path.replace(os.path.expanduser("~"), "~", 1)

f = open("python_messaging.txt", "w")
f.write(path)
f.close()
' > "python_script.py"
python python_script.py

openmm_parent_dir=`cat python_messaging.txt`
openmm_lib_dir="${openmm_parent_dir}/lib"
openmm_include_dir="${openmm_parent_dir}/include"

# Creates:
# libOpenMMMetal.dylib
# libOpenMMAmoebaMetal.dylib
# libOpenMMDrudeMetal.dylib
# libOpenMMRPMDMetal.dylib
if [[ $run_all_tests == true ]]; then
  build_tests_flags="-DOPENMM_BUILD_OPENCL_TESTS=1 -DOPENMM_BUILD_LONG_TESTS=1 -DOPENMM_BUILD_VERY_LONG_TESTS=1"
elif [[ $run_most_tests == true ]]; then
  build_tests_flags="-DOPENMM_BUILD_OPENCL_TESTS=1 -DOPENMM_BUILD_LONG_TESTS=1 -DOPENMM_BUILD_VERY_LONG_TESTS=0"
elif [[ $run_quick_tests == true ]]; then
  build_tests_flags="-DOPENMM_BUILD_OPENCL_TESTS=1 -DOPENMM_BUILD_LONG_TESTS=0"
else
  build_tests_flags="-DOPENMM_BUILD_OPENCL_TESTS=0"
fi
cmake .. \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_INSTALL_PREFIX="/usr/local/openmm" \
  -DOPENMM_DIR=${openmm_parent_dir} \
  ${build_tests_flags} \

# 4 CPU cores is the most compatible amount of cores across all Mac systems.
# It doesn't eat into M1 efficiency cores which harm performance, and doesn't
# cause load imbalance on quad-core Intel Macs.

# Nevermind - 8 cores is fastest on my local machine. Perhaps using the E-cores
# wouldn't _hurt_ on base M1 and double-overloading a quad-core Intel Mac
# wouldn't hurt. It won't change performance on the hexa-core Intel Mac mini.
make -j8

if [[ $install_plugins == true ]]; then
  sudo make install
fi
if [[ $run_all_tests == true ]]; then
  make test
elif [[ $run_most_tests == true ]]; then
  make test
elif [[ $run_quick_tests == true ]]; then
  make test
fi

mv "platforms/metal/libOpenMMMetal.dylib" "libOpenMMMetal.dylib"
mv "plugins/amoeba/platforms/metal/libOpenMMAmoebaMetal.dylib" "libOpenMMAmoebaMetal.dylib"
mv "plugins/drude/platforms/metal/libOpenMMDrudeMetal.dylib" "libOpenMMDrudeMetal.dylib"
mv "plugins/rpmd/platforms/metal/libOpenMMRPMDMetal.dylib" "libOpenMMRPMDMetal.dylib"
