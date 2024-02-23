#!/bin/bash
#SBATCH --array=1
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=1600_spectra_no_skip_bn_false
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/1600-spectra-no-skip.yaml --ablat-id $SLURM_ARRAY_TASK_ID
