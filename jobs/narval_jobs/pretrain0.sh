#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=dz-spectra-3-512-regu-no-skip-regu-beta-32
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=20000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/pretrain_tmp.yaml --ablat-id 0
