#!/bin/bash
#SBATCH --array=0-17
#SBATCH --time=9:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200-spectra-no-skip-latents-mlp-capacity
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/3200-spectra-no-skip.yaml --ablat-id $SLURM_ARRAY_TASK_ID
