#!/bin/bash

# By default, only run tests that give you useful feedback in a reasonable
# amount of time.
enable_long_tests=false

# Installing the plugin requires 'sudo' privileges to modify an external
# directory.
install_plugin=false

if [[ $# != 0 ]]; then
  invalid_input=false
  if [[ $# -gt 2 ]]; then
    echo "Too many arguments."
    invalid_input=true
  fi
  
  if [[ $invalid_input == false ]]; then
    for param in "$@"; do
      if [[ $param == "--enable-long-tests" ]]; then
        if [[ $enable_long_tests == true ]]; then
          echo "Duplicate argument '--enable-long-tests'."
          invalid_input=true
        else
          enable_long_tests=true
        fi
      elif [[ $param == "--install-plugin" ]]; then
        if [[ $install_plugin == true ]]; then
          echo "Duplicate argument '--install-plugin'."
          invalid_input=true
        else
          install_plugin=true
        fi
      else
        echo "Unrecognized argument '${param}'."
        invalid_input=true
      fi
    done
  fi
  
  if [[ $invalid_input == true ]]; then
    echo "Usage: build.sh [--platform=[macOS, iOS, tvOS]; default=\"macOS\"]"
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
python3 python_script.py

if [[ "success" != `cat python_messaging.txt` ]]; then
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
python3 python_script.py

openmm_version=`cat python_messaging.txt`
old_IFS=$IFS
IFS='.'
read -a strarr <<< "$openmm_version"
openmm_version_major=$((strarr[0] + 0)) # Force convert to integer.
IFS=$old_IFS

# TODO: Check version minor when HIP workaround is patched. Also incorporate
# this check into the installer script.
if [[ $openmm_version_major -lt 8 ]]; then
    echo "OpenMM must be version 8.0.0 or higher."
    exit -1
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
python3 python_script.py

openmm_parent_dir=`cat python_messaging.txt`
openmm_lib_dir="${openmm_parent_dir}/lib"
openmm_include_dir="${openmm_parent_dir}/include"

# Creates:
# libOpenMMMetal.dylib
# libOpenMMAmoebaMetal.dylib
# libOpenMMDrudeMetal.dylib
# libOpenMMRPMDMetal.dylib
cmake .. -DCMAKE_INSTALL_PREFIX="/usr/local/openmm" \
  -DOPENMM_DIR=${openmm_parent_dir} \

# 4 CPU cores is the most compatible amount of cores across all Mac systems.
# It doesn't eat into M1 efficiency cores which harm performance, and doesn't
# cause load imbalance on quad-core Intel Macs.

# Nevermind - 8 cores is fastest on my local machine. Perhaps using the E-cores
# wouldn't _hurt_ on base M1 and double-overloading a quad-core Intel Mac
# wouldn't hurt. It won't change performance on the hexa-core Intel Mac mini.
make -j8

mv "platforms/metal/libOpenMMMetal.dylib" "libOpenMMMetal.dylib"
mv "plugins/amoeba/platforms/metal/libOpenMMAmoebaMetal.dylib" "libOpenMMAmoebaMetal.dylib"
mv "plugins/drude/platforms/metal/libOpenMMDrudeMetal.dylib" "libOpenMMDrudeMetal.dylib"
mv "plugins/rpmd/platforms/metal/libOpenMMRPMDMetal.dylib" "libOpenMMRPMDMetal.dylib"
