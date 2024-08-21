#!/bin/bash

# Get number of CPUs
num_cpus=$(nproc)
echo "Number of CPUs: $num_cpus"

# Get total CPU memory (RAM) in GB
cpu_memory_gb=$(free -g | awk '/^Mem:/{print $2}')
echo "CPU Memory: ${cpu_memory_gb} GB"

# Get GPU memory in GB (using nvidia-smi for NVIDIA GPUs)
if command -v nvidia-smi &> /dev/null; then
    gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    gpu_memory_gb=$(echo "$gpu_memory" | awk '{s+=$1} END {print s/1024}')
    echo "Total GPU Memory: ${gpu_memory_gb} GB"
else
    echo "nvidia-smi command not found or no NVIDIA GPU detected."
fi
