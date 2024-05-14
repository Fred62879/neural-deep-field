#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=dz-spectra-generalize-5-512-skip-w-wo-regu-weighted
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=20000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/pretrain.yaml --infer-log-dir '20240513-143729_ssim_5_512_skip_regu' --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --regularize-spectra-latents

python app/main_astro.py --config configs/narval_configs/pretrain.yaml --infer-log-dir '20240513-143835_ssim_5_512_skip_no_regu' --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers

python app/main_astro.py --config configs/narval_configs/generalize.yaml --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --regularize-spectra-latents --pretrain-log-dir '20240513-143729_ssim_5_512_skip_regu' --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240513-143729_ssim_5_512_skip_regu/pretrain_spectra/global_restframe_l2_loss.npy' --log-fname 'genlz_5_512_skip_regu_weighted_143729'

python app/main_astro.py --config configs/narval_configs/generalize.yaml --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --pretrain-log-dir '20240513-143835_ssim_5_512_skip_no_regu' --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240513-143835_ssim_5_512_skip_no_regu/pretrain_spectra/global_restframe_l2_loss.npy' --log-fname 'genlz_5_512_skip_no_regu_weighted_143835'
