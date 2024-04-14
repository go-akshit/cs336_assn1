#!/bin/bash
#SBATCH --job-name=bpe_owt
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=bpe_owt_%j.out
#SBATCH --error=bpe_owt_%j.err
#SBATCH --mem=100G
python3 train_bpe_datasets.py