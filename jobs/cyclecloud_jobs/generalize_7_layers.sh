#!/bin/bash
#SBATCH --array=3-5
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=dz-spectra-pretrain-7-layers
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=20480

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/generalize.yaml --ablat-id $SLURM_ARRAY_TASK_ID
