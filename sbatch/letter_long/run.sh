#!/bin/bash

# Define your seeds and eps
seeds=(0 1 2 3 4 5 6)  # Example seed values
eps=(0.0 0.1 0.05 0.15 0.2)  # Example ep values

# Loop through seeds and eps and run sbatch for each file
for seed in "${seeds[@]}"; do
    for ep in "${eps[@]}"; do
        file_name="sbatch/letter_long/train_${seed}_${ep}.sh"
        sbatch "$file_name"
        echo "Submitted $file_name"
    done
done

echo "All scripts submitted to sbatch."