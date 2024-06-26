#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-sanity-check-sample-16-bins-all-spectra
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=0

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/sanity_check_ablat_bins.yaml --ablat-id 0
