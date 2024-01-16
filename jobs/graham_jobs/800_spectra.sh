#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=800_spectra
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../

python app/main_astro.py --config configs/800_spectra.yaml
