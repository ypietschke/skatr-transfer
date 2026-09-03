#!/usr/bin/bash

# project directory
export SKATR_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# modules
module load mpi/openmpi/4.1-gnu-14.2
module load devel/miniforge  

conda activate eor_env

# auto completion
eval "$(python main.py -sc install=bash)"

# complete stacktrace
export HYDRA_FULL_ERROR=1

# resolve multithreading conflict
export MKL_THREADING_LAYER=GNU

export OMP_NUM_THREADS=32
export NUMEXPR_MAX_THREADS=32
