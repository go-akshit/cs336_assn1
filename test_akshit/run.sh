#!/bin/bash
#SBATCH --job-name=bpe_train_tiny
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:60:00
#SBATCH --output=bpe_train_tiny_%j.out
#SBATCH --error=bpe_train_tiny_%j.err
#SBATCH --mem=50G
python3 train_bpe_datasets.py
