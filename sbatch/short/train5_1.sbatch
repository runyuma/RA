#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --time=35:59:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load miniconda/3.9
# module load cuda/11.2 cudnn/11.2-8.1.1.33

conda activate ra

cd /tudelft.net/staff-umbrella/rarma/src/RA/
python train/train.py --render f --save t --seed 5 --iters 80000 --save_freq 2500 --device 1 --ep 0.1