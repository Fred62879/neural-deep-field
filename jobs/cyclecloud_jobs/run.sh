#!/bin/bash

source /shared/home/ztxie/envs/wisp/bin/activate
cd ../../

python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml --wave-multiplier 1 --infer-log-dir 20240606-222737_pretrain_multiplier_1
python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml --wave-multiplier 4 --infer-log-dir 20240606-223125_pretrain_multiplier_4
python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml --wave-multiplier 16 --infer-log-dir 20240606-223513_pretrain_multiplier_16
python app/main_astro.py --config configs/cyclecloud_configs/pretrain.yaml --wave-multiplier 32 --infer-log-dir 20240606-223901_pretrain_multiplier_32
