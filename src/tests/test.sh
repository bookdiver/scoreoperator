#!/bin/bash

# Assign the arguments to variables
file=$1
func=$2
using_gpu=${3:-false}

# Check the value of using_gpu and set the JAX_PLATFORM_NAME accordingly
if $using_gpu ; then
    JAX_PLATFORM_NAME="gpu"
else
    JAX_PLATFORM_NAME="cpu"
fi

# Run the pytest command
JAX_PLATFORM_NAME=$JAX_PLATFORM_NAME pytest $file.py::$func -W ignore -rP