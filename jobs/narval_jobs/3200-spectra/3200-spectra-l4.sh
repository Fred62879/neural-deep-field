#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=3200-spectra-skip-multi-conversion-l4
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=4000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/3200-spectra-skip-multi-conversion-spectra-no-convolve.yaml
python app/main_astro.py --config configs/narval_configs/3200-spectra-skip-multi-conversion-spectra-convolve-sigma-25.yaml
python app/main_astro.py --config configs/narval_configs/3200-spectra-skip-multi-conversion-spectra-convolve-sigma-50.yaml
