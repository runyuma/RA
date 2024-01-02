# Define your seeds and eps
seeds = [0,1, 2, 3, 4, 5, 6]  # Example seed values
eps = [0.0, 0.05, 0.1, 0.15]  # Example ep values

template = """#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load miniconda/3.9
# module load cuda/11.2 cudnn/11.2-8.1.1.33

conda activate ra

cd /tudelft.net/staff-umbrella/rarma/src/RA/
python train/train_long_letter.py --render f --save t --seed {seed} --iters 80000 --save_freq 2500 --device 1 --ep {ep}
"""

# Create files
for seed in seeds:
    for ep in eps:
        # Create a filename based on the seed and ep
        filename = f"sbatch/letter_long/train_{seed}_{ep}.sh"
        
        # Replace placeholders in the template
        file_content = template.format(seed=seed, ep=ep)
        
        # Write to a file
        with open(filename, 'w') as file:
            file.write(file_content)
            print(f"Created file: {filename}")

print("All files created.")