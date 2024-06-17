#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-ssim-pretrain-bn
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=40960

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml
