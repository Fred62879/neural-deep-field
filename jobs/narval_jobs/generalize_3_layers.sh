#!/bin/bash
#SBATCH --array=0-5
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vdz-spectra-genlz-3-layers-2000-spectra
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/generalize_3_layers.yaml --ablat-id $SLURM_ARRAY_TASK_ID
