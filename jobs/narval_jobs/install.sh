#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=install
#SBATCH --output=./outputs/install.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

module load python/3.9.6
virtualenv --no-download ~/envs/wisp_env

source ~/envs/wisp_env/bin/activate

module load cuda/11.7
pip install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu117/torch-1.13.0%2Bcu117-cp310-cp310-linux_x86_64.whl

cd ~/scratch/vision/code/kaolin
pip install -r tools/build_requirements.txt -r tools/viz_requirements.txt -r tools/requirements.txt
python setup.py develop

cd ~/scratch/vision/code/implicit-universe-wisp
pip install tifffile==2023.4.12
pip install -r requirements.txt
python setup.py develop
