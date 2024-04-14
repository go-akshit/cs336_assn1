#!/bin/bash
#SBATCH --job-name=encode_iterable_memory_usage
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:60:00
#SBATCH --output=encode_iterable_memory_usage_%j.out
#SBATCH --error=encode_iterable_memory_usage_%j.err
#SBATCH --mem=50G
python3 train_bpe_datasets.py