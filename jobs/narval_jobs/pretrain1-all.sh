#!/bin/bash
#SBATCH --array=0-9,11
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vdz-spectra-pretrain-5-7-layers
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/pretrain_5_7_layers.yaml --ablat-id $SLURM_ARRAY_TASK_ID
