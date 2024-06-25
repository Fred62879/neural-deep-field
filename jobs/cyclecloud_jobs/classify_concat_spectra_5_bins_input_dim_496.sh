#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-classify-based-on-concat-spectra-5-bins-input-dim-496
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=204800

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/classify_input_dim_ablation.yaml --ablat-id 3
