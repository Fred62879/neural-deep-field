#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vdz-spectra-clsfy-small-bsz-bn
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=0

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/classify_small_bsz.yaml --ablat-id $SLURM_ARRAY_TASK_ID
