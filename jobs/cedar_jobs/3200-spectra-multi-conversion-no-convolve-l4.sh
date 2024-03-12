#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:v100l:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=infer-3200-spectra-multi-conversion
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=4000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/cedar_configs/3200-spectra-multi-conversion-no-convolve.yaml
python app/main_astro.py --config configs/cedar_configs/3200-spectra-multi-conversion-convolve-sigma-25.yaml
python app/main_astro.py --config configs/cedar_configs/3200-spectra-multi-conversion-convolve-sigma-50.yaml
