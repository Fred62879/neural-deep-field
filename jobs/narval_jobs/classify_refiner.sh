#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vdz-spectra-clsfy-refiner
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=200G

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/classify_refiner.yaml
