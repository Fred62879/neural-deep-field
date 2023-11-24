#!/bin/bash
#SBATCH --array=1-6
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=400_spectra_arr
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../

python app/main_astro.py --config configs/400_spectra.yaml --ablat-id $SLURM_ARRAY_TASK_ID
