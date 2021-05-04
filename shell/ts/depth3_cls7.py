#!/bin/bash
#SBATCH --verbose
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=ts3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=leic.monitor@gmail.com
#SBATCH --output=hpc_outputs/slurm_%j.out
#SBATCH --error=hpc_outputs/slurm_%j.err 
#SBATCH --gres=gpu:1080ti:1

cd /home/lc3909/wl_and_swl
module purge

nvidia-smi

set -x

# echo ${SLURM_ARRAY_TASK_ID}

python3.7m main_tch_std_class7.py --teacher gin_dim32_hop3 --student swl_gnn_dim32_gin3_mint0_mott0_stack0_idnt0 --train_per_class ${SLURM_ARRAY_TASK_ID}