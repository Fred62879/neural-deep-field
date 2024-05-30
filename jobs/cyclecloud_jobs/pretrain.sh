#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --job-name=dz-spectra-pretrain
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --mem=20480

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml