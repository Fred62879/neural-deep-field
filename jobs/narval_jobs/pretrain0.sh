#!/bin/bash
#SBATCH --time=14:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=vipers-spectra-pretrain
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

#python app/main_astro.py --config configs/narval_configs/pretrain.yaml
python app/main_astro.py --config configs/narval_configs/pretrain_tmp.yaml
