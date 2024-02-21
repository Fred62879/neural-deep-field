#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=1600_spectra_sc_regu_beta
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/envs/wisp_env/bin/activate
cd ../../

#python app/main_astro.py --config configs/1600-spectra-3-layers-skip.yaml
#python app/main_astro.py --config configs/1600-spectra-3-layers-no-skip.yaml
#python app/main_astro.py --config configs/1600-spectra-3-layers-skip-same-dim.yaml
#python app/main_astro.py --config configs/1600-spectra-regu-beta-16-3-layers-skip-same-dim-sep-layers.yaml

python app/main_astro.py --config configs/1600-spectra-regu-beta-1-3-layers-skip-same-dim-sep-layers.yaml
python app/main_astro.py --config configs/1600-spectra-regu-beta-2-3-layers-skip-same-dim-sep-layers.yaml
python app/main_astro.py --config configs/1600-spectra-regu-beta-4-3-layers-skip-same-dim-sep-layers.yaml
python app/main_astro.py --config configs/1600-spectra-regu-beta-8-3-layers-skip-same-dim-sep-layers.yaml
