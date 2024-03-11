#!/bin/bash
#SBATCH --array=2,8,14
#SBATCH --time=7:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200-spectra-skip-multi-conversion-latents-mlp-capacity-sc-7hrs
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/3200-spectra-skip-add-multi-conversion-sc.yaml --ablat-id $SLURM_ARRAY_TASK_ID
