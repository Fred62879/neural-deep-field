#!/bin/bash
#SBATCH --time=26:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-classify-small-bsz-5-layers-bn
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=40960

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/classify_small_bsz.yaml --log-fname 'clsfy_small_bsz_bn_5_layers' --classifier-decoder-batch-norm
