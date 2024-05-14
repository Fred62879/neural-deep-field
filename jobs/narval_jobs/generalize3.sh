#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=dz-spectra-generalize-5-512-skip-regu-beta-32
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=20000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/pretrain.yaml --infer-log-dir '20240514-024145_ssim_5_512_skip_regu_beta_32' --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --regularize-spectra-latents --spectra-latents-regu-beta 32

python app/main_astro.py --config configs/narval_configs/generalize.yaml --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --regularize-spectra-latents --spectra-latents-regu-beta 32 --pretrain-log-dir '20240514-024145_ssim_5_512_skip_regu_beta_32' --log-fname 'genlz_5_512_skip_regu_beta_32_024145'

python app/main_astro.py --config configs/narval_configs/generalize.yaml --decoder-num-hidden-layers 5 --decoder-latents-skip-all-layers --regularize-spectra-latents --spectra-latents-regu-beta 32 --pretrain-log-dir '20240514-024145_ssim_5_512_skip_regu_beta_32' --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240514-024145_ssim_5_512_skip_regu_beta_32/pretrain_spectra/global_restframe_l2_loss.npy' --log-fname 'genlz_5_512_skip_regu_beta_32_weighted_024145'
