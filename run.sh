#!/bin/bash

python app/main_astro.py --config configs/512_main_train_unconvolved.yaml
python app/main_astro.py --config configs/512_main_train_convolved.yaml
