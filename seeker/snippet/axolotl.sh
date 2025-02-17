#date: 2025-02-17T16:44:51Z
#url: https://api.github.com/gists/b655f77aaac55b7b1c9b7968723d68aa
#owner: https://api.github.com/users/jeromeku

#!/bin/bash

# Author: Yenting Lin (ytl@ieee.org) Fed 17, 2025
# This script was modified from stas00/ml-engineering - https://github.com/stas00/ml-engineering/blob/master/orchestration/slurm/launchers/accelerate-launcher.slurm

# Usage: 
# Preprocessing [Optional]: 
#     sbatch -N 1 axolotl.sh PATH_TO_AXOLOTL_CONFIG preprocess
# Training: 
#     sbatch -N NUMBER_OF_NODES axolotl.sh PATH_TO_AXOLOTL_CONFIG train

# this is a multi-node SLURM script using `accelerate` launcher

#SBATCH --job-name=axolotl
# -----------------EDIT THIS----------------------------------------------------------------
#SBATCH --partition=YOUR_SLURM_PARTITION_HERE
# -----------------^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------------------------------------
#SBATCH --cpus-per-task=96
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per node
#SBATCH --gres=gpu:8                 # EDIT this if it's not 8-gpus per node
#SBATCH --exclusive
#SBATCH --output=~/logs/axolotl/%x-%j.out
#SBATCH --error=~/logs/axolotl/%x-%j.err
#SBATCH --overcommit

echo "START TIME: $(date)"

CONFIG_PATH="$1"
echo "CONFIG_PATH: ${CONFIG_PATH}"

COMMAND="${2:-train}" # could be train, inference, preprocess, see https://axolotl-ai-cloud.github.io/axolotl/docs/getting-started.html#sec-common-tasks
echo "COMMAND: ${COMMAND}"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x


# -----------------ACTIVATE YOUR ENV----------------------------------------------------------------
# source ~/.bashrc
# source ~/axolotl/.venv/bin/activate
# export PATH=/usr/local/cuda-12.4/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
# export HF_TOKEN= "**********"
# -----------------^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------------------------------------

LOG_PATH="~/logs/axolotl/main_log.txt"

# -----------------EDIT THIS----------------------------------------------------------------
# EDIT the path to accelerate config file and fill it with actual Accelerate config
ACCELERATE_CONFIG_FILE=~/accelerate_config.yaml # see another file in the gist
# -----------------^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------------------------------------

export ACCELERATE_LOG_LEVEL=info
export TRANSFORMERS_VERBOSITY=info
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200000 # force crashing on nccl issues like hanging broadcast after 2 hours

NNODES=$SLURM_NNODES
if [ "$COMMAND" = "preprocess" ]; then
    NUM_PROCESSES=1
else
    GPUS_PER_NODE=$(echo $SLURM_JOB_GPUS | awk -F',' '{print NF}')
    NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
fi

# define the node 0 hostname:port
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

# note `\$SLURM_PROCID` we don't want it interpolated till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s|tr -dc '0-9'): --tee 3 \
    "

# -----------------EDIT THIS TO YOUR TIMEZONE----------------------------------------------------------------
TIMESTAMP=$(TZ=Asia/Taipei date +"%Y%m%d_%H%M")
# -----------------^^^^^^^^^^^^^^^^^^^^^^^^----------------------------------------------------------------
MODEL_PATH="~/ckpt/run_${TIMESTAMP}"

echo "Full model path will be: ${MODEL_PATH}"
echo "Generated timestamp: ${TIMESTAMP}"

export PROGRAM="\
    -m axolotl.cli.${COMMAND} \
    ${CONFIG_PATH} \
    --output_dir=${MODEL_PATH} \
"


export CMD="$LAUNCHER $PROGRAM"

echo $CMD

# Singularity execution command
#singularity exec --nv --bind /data/home/ytl/axolotl:/workspace/axolotl --bind /fsx-project:/fsx-project docker://winglian/axolotl:main-latest /bin/bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$CMD" 2>&1 | tee -a $LOG_PATH

echo "END TIME: $(date)"