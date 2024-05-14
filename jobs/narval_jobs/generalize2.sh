#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=dz-spectra-generalize-3-512-no-skip-regu-beta-32
#SBATCH --output=./outputs/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=20000

source ~/envs/wisp_env/bin/activate
cd ../../

python app/main_astro.py --config configs/narval_configs/pretrain.yaml --infer-log-dir '20240514-024138_ssim_3_512_no_skip_regu_beta_32' --regularize-spectra-latents --spectra-latents-regu-beta 32

python app/main_astro.py --config configs/narval_configs/generalize.yaml --regularize-spectra-latents --spectra-latents-regu-beta 32 --pretrain-log-dir '20240514-024138_ssim_3_512_no_skip_regu_beta_32' --log-fname 'genlz_3_512_no_skip_regu_beta_32_024138'

python app/main_astro.py --config configs/narval_configs/generalize.yaml --regularize-spectra-latents --spectra-latents-regu-beta 32 --pretrain-log-dir '20240514-024138_ssim_3_512_no_skip_regu_beta_32' --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240514-024138_ssim_3_512_no_skip_regu_beta_32/pretrain_spectra/global_restframe_l2_loss.npy' --log-fname 'genlz_3_512_no_skip_regu_beta_32_weighted_024138'
