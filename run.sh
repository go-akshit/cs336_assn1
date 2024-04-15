#!/bin/bash
#SBATCH --job-name=first2
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --output=first2_%j.out
#SBATCH --error=first2_%j.err
#SBATCH --mem=100G
python3 cs336_basics/training_together.py --tokenize==True --experiment_name="first" --tokenized_input_train_file="./cs336_basics/mydata/TinyStoriesV2-GPT4-train_tokenized_2"
