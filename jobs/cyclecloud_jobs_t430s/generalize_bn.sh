#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=vdz-spectra-generalize-1k-spectra-bn-save-redshift-data
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=40960

source /shared/home/ztxie-t430s/envs/wisp/bin/activate
cd ../../

# train 22hrs

python app/main_astro.py --config configs/cyclecloud_configs_t430s/generalize_bn.yaml
