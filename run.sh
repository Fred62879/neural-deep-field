#!/bin/bash

python app/main_astro.py --config configs/dense.yaml
python app/main_astro.py --config configs/hash.yaml
python app/main_astro.py --config configs/pe.yaml

python app/main_astro.py --config configs/dense_pretrain_conv.yaml
python app/main_astro.py --config configs/hash_pretrain_conv.yaml
python app/main_astro.py --config configs/pe_pretrain_conv.yaml

python app/main_astro.py --config configs/dense_pretrain_unconv.yaml
python app/main_astro.py --config configs/hash_pretrain_unconv.yaml
python app/main_astro.py --config configs/pe_pretrain_unconv.yaml
