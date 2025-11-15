#!/bin/bash

# Check if the argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 'venv_name'"
    exit 1
fi

# Prepare the environment
source $1/bin/activate
source /home/spack/spack/share/spack/setup-env.sh
spack load cuda@12.8
