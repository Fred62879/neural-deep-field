#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200_spectra_latents_mlp_capacity_sc
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 0
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 1
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 2
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 3
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 4
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 5

python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 6
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 7
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 8
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 9
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 10
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 11

python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 12
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 13
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 14
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 15
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 16
python app/main_astro.py --config configs/3200-spectra.yaml --ablat-id 17
