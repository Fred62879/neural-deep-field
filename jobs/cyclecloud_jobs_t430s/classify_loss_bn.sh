#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-classify-all-spectra-based-on-loss-bn
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=40960

source /shared/home/ztxie-t430s/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs_t430s/classify_bn.yaml --classify-based-on-loss
