#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-classify-large-bsz-5-layers-skip
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=40960

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/classify_large_bsz.yaml --classifier-decoder-num-hidden-layers 5 --classifier-decoder-skip-all-layers --log-fname 'clsfy_large_bsz_no_bn_5_layers'
