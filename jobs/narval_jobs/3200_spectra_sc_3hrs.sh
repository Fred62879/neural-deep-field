#!/bin/bash
#SBATCH --array=0,1,3,6,7,9,12,13,15
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200_spectra_latents_mlp_capacity_sc_3hrs
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id $SLURM_ARRAY_TASK_ID
