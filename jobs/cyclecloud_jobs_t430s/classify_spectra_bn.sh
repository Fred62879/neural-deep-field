#!/bin/bash
#SBATCH --time=19:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-classify-all-spectra-bn-5-bins
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=0

source /shared/home/ztxie-t430s/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs_t430s/classify_bn.yaml --ablat-id 0
