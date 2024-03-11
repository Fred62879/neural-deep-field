#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=1600-spectra-sc-all
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/1600-spectra-no-skip-sc.yaml --ablat-id 0
python app/main_astro.py --config configs/1600-spectra-no-skip-sc.yaml --ablat-id 1
python app/main_astro.py --config configs/1600-spectra-skip-concat-sc.yaml --ablat-id 0
python app/main_astro.py --config configs/1600-spectra-skip-concat-sc.yaml --ablat-id 1

python app/main_astro.py --config configs/1600-spectra-skip-add-convert-input-sc.yaml --ablat-id 0
python app/main_astro.py --config configs/1600-spectra-skip-add-convert-input-sc.yaml --ablat-id 1
python app/main_astro.py --config configs/1600-spectra-skip-add-multi-conversion-sc.yaml --ablat-id 0
python app/main_astro.py --config configs/1600-spectra-skip-add-multi-conversion-sc.yaml --ablat-id 1
python app/main_astro.py --config configs/1600-spectra-skip-add-single-conversion-sc.yaml --ablat-id 0
python app/main_astro.py --config configs/1600-spectra-skip-add-single-conversion-sc.yaml --ablat-id 1
