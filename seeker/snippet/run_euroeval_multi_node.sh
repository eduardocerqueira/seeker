#date: 2025-12-29T17:14:24Z
#url: https://api.github.com/gists/7bccfdbb0c2cddf188321101d6f3c4ff
#owner: https://api.github.com/users/tvosch

#!/bin/bash

#SBATCH --gpus-per-task=...
#SBATCH --nodes=...
#SBATCH --ntasks-per-node=1    # Run 1 Ray cluster instance per node
#SBATCH --partition=...
#SBATCH --account=...
#SBATCH --time=...

# Load modules, source environments etc.
# (Add your specific module loads or conda/venv activation here)

# Multi-Node Ray Cluster Configuration
# Number of GPUs available on this node (set by SLURM)
export NUM_GPUS=$SLURM_GPUS_PER_NODE

# Get the IP address of the head node (node 0) for cluster communication
export HEAD_IPADDRESS=$(hostname -i)

# Dynamically find an available port for Ray to avoid conflicts
export RAY_PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

# HuggingFace and vLLM Configuration
export HF_TOKEN= "**********"
export HF_HOME=...             # Cache directory for HuggingFace models

# Uncomment if you encounter compilation issues with FlashInfer (the default)
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# EuroEval Task Configuration
export MODEL=...                 # Specify your model (e.g., "meta-llama/Llama-3-8B")
export DATASET=...               # Specify your dataset (e.g., "arc_challenge")

# Node-Specific Execution Function
run_on_node() {
    if [[ "$SLURM_NODEID" == "0" ]]; then
        # HEAD NODE: Initialize the Ray cluster
        export VLLM_HOST_IP=$HEAD_IPADDRESS
        
        # Start Ray head node with specified resources
        ray start --head --port=${RAY_PORT} --num-gpus=${NUM_GPUS} --node-ip-address=${HEAD_IPADDRESS}
        
        # Wait for Ray cluster to initialize
        sleep 10
        
        # Display cluster status for debugging
        ray status
        ray list nodes
        
        # Run EuroEval
        euroeval --model ${MODEL} \
                 --dataset ${DATASET} \
                 --clear-model-cache \
                 --trust-remote-code \
                 --evaluate-test-split \
                 --cache-dir ${TMPDIR} \
    else
        # WORKER NODES: Connect to the head node and block until job completes
        ray start --address=${HEAD_IPADDRESS}:${RAY_PORT} --num-gpus=${NUM_GPUS} --block
    fi
}

# Export the function so it's available in the srun subshell
export -f run_on_node

# Run the function on all allocated nodes simultaneously
srun bash -c 'run_on_node'cated nodes simultaneously
srun bash -c 'run_on_node'