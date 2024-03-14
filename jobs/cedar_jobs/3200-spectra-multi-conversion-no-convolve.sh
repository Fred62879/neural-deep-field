#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200-spectra-multi-conversion-no-convolve
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=4000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/cedar_configs/3200-spectra-multi-conversion-no-convolve.yaml