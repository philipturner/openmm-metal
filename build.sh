#!/bin/bash

if [[ -d "./build" ]]; then
    mkdir ".build"
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
# would expect on client machines.
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

# Need:
# libOpenMMMetal.dylib
# libOpenMMAmoebaMetal.dylib
# libOpenMMDrudeMetal.dylib
# libOpenMMRPMDMetal.dylib

cmake ..

# TODO:
# Embed the binaries into a supermassive Bash script, just like other GitHub
# packages. This prevents needing to mess with cross-compilation.

# TODO:
# Give security-conscious users the ability to install from source.
