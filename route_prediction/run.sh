#!/bin/bash
#SBATCH --job-name=run
#SBATCH --partition=dsag
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=route_prediction/logs/runtime/%x_%j.out
#SBATCH --error=route_prediction/logs/runtime/%x_%j.err

. /etc/profile
module load lang/python/3.13.0

source "$HOME/LaDe/LaDe/.venvs/tropical_cproute/bin/activate"
# pip install datasets

python route_prediction/run.py
