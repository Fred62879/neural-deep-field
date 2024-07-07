#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vdz-spectra-gasnet-baseline
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=200G

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/gasnet_baseline.yaml --ablat-id $SLURM_ARRAY_TASK_ID
