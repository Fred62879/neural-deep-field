#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=800_spectra_3_layers
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/800-spectra-3-layers-skip.yaml
python app/main_astro.py --config configs/800-spectra-3-layers-no-skip.yaml

#python app/main_astro.py --config configs/800-spectra-3-layers-skip-same-dim.yaml
#python app/main_astro.py --config configs/800-spectra-3-layers-skip-same-dim-sep-layers.yaml
