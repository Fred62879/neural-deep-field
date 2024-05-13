#!/bin/bash


python app/main_astro.py --config configs/autodecoder.yaml

#python app/main_astro.py --config configs/autodecoder_sc.yaml

#python app/main_astro.py --config configs/autodecoder_sc.yaml --sanity-check-no-freeze --infer-log-dir '20240512-141601_sc_no_weight_no_freeze_222050'

#python app/main_astro.py --config configs/autodecoder_sc.yaml --use-global-spectra-loss-as-lambdawise-weights --infer-log-dir '20240512-143837_sc_l2_weight_222050'

#python app/main_astro.py --config configs/autodecoder_sc.yaml --use-global-spectra-loss-as-lambdawise-weights --sanity-check-no-freeze --infer-log-dir '20240512-145900_sc_l2_weight_no_freeze_222050'

#python app/main_astro.py --config configs/autodecoder_sc.yaml --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240511-222050/pretrain_spectra/global_restframe_ssim1d_loss.npy' --infer-log-dir '20240512-152423_sc_ssim_weight_222050'

#python app/main_astro.py --config configs/autodecoder_sc.yaml --use-global-spectra-loss-as-lambdawise-weights --global-restframe-spectra-loss-fname '20240511-222050/pretrain_spectra/global_restframe_ssim1d_loss.npy' --sanity-check-no-freeze --infer-log-dir '20240512-120215_sc_ssim_weight_no_freeze_222050'

# python app/main_astro.py --config configs/3200-spectra-skip-add-convert-input-sc.yaml --ablat-id 0
