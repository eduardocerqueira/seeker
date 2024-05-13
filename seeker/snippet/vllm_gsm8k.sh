#date: 2024-05-13T16:49:23Z
#url: https://api.github.com/gists/bccf5a4408fde61249956dd4c3f56438
#owner: https://api.github.com/users/akoksal

#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 0-00:20:00
#SBATCH -o tutorial.out
#SBATCH -e tutorial.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=HIDDENMAIL@gmail.com

source PATH/.bashrc
mamba activate ENVNAME

python3 vllm_gsm8k.py --model_name mistralai/Mixtral-8x7B-v0.1 --gpu_count 2