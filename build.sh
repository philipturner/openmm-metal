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
' >> "python_script.py"
python3 python_script.py

# Fail if not installed.
if [[ "success" != `cat python_messaging.txt` ]]; then
    echo "OpenMM not installed."
    exit -1
fi

# Extract OpenMM version.
echo '
import openmm

f = open("python_messaging.txt", "w")
f.write(openmm.__version__)
f.close()
' >> "python_script.py"
python3 python_script.py

# Fail if too early.
echo `cat python_messaging.txt`
