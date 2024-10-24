#!/bin/bash

# Slurm job options (name, compute nodes, job time)
#SBATCH --job-name=benchio_c
#SBATCH --time=0:20:00
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. dc116-s1234567)
#SBATCH --account=[budget code]
# We use the "standard" partition as we are running on CPU nodes
#SBATCH --partition=standard
# Use "short" whenever runtime is under 20 minutes and if allowed by node count
#SBATCH --qos=short

# Change to the submission directory
cd $SLURM_SUBMIT_DIR

export OMP_NUM_THREADS=1

# ADIOS2 dependencies
module use /work/z19/shared/sfarr/modulefiles
module load adios/2.8.3

# UCX MPI library for better ARCHER2 scaling
module load craype-network-ucx
module load cray-mpich-ucx
module load libfabric

# Remember to set striping on unstriped/fullstriped/striped directories. See README.md for instructions on Lustre.

# Launch the parallel job. Change depending on desired number of repeat runs and configuration.
for i in {1..10}; do
    srun --unbuffered --hint=nomultithread --distribution=block:block ./benchio -n1 1000 -n2 1000 -n3 1000 -sc global -st fullstriped
done