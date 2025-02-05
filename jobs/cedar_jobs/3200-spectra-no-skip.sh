#!/bin/bash
#SBATCH --array=10,16
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200-spectra-no-skip-latents-mlp-capacity
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=4000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/cedar_configs/3200-spectra-no-skip.yaml --ablat-id $SLURM_ARRAY_TASK_ID
