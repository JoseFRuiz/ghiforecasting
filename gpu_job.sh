#!/bin/bash
#SBATCH --job-name=gputest
#SBATCH --output=multi_gpu.out
#SBATCH --error=mult_gpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=1
#SBATCH --mem-per-cpu=20000mb
#SBATCH --distribution=cyclic:cyclic
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Load modules
module purge
module load cuda/11.1.0

# Activate conda env
source ~/.bashrc
source activate ghifo_py39

# Run training script
python train_gnn.py

conda deactivate
