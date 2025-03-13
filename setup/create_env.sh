#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=env_creation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=01:00:00

# Exit immediately if a command exits with a non-zero status
set -e

# Start timer
start_time=$(date +%s)

# Create environment name with static name
env_prefix=lingua

# Create the conda environment
micromamba create -n $env_prefix python=3.11 -y -c anaconda
micromamba activate $env_prefix

echo "Currently in env $(which python)"
micromamba install -c nvidia/label/cuda-12.3.0 cuda-toolkit -y
# Install packages
pip install torch==2.5.0 xformers --index-url https://download.pytorch.org/whl/cu121
pip install ninja
pip install --requirement requirements.txt

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"


