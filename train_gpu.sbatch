#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=medium
#SBATCH --ntasks=1
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
previous=$(/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/tail -n '+2')
/usr/bin/nvidia-smi

module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/11.2 cudnn/11.2-8.1.1.33

conda activate ra

cd /tudelft.net/staff-umbrella/rarma/src/RA/
srun python train/train.py --render f --save t --seed 4 --iters 80000 --save_freq 2500 --device 1