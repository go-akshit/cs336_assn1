#!/bin/bash
#SBATCH --job-name=bpe_train
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=bpe_train_%j.out
#SBATCH --error=bpe_train_%j.err
python3 train_bpe_datasets.py
