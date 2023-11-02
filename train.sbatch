#!/bin/sh

SBATCH --partition=general
SBATCH --qos=short
SBATCH --time=04:00:00
SBATCH --cpus-per-task=2
SBATCH --gres=gpu:1
SBATCH --mem=16000
SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/11.2 cudnn/11.2-8.1.1.33

conda activate ra

cd /tudelft.net/staff-umbrella/rarma/src/RA/

srun python train/train.py