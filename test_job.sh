#!/bin/bash
#SBATCH --job-name=test_imports
#SBATCH --output=test_imports.out
#SBATCH --error=test_imports.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000mb
#SBATCH --time=00:30:00

echo "Date      = $(date)"
echo "host      = $(hostname -s)"
echo "Directory = $(pwd)"

# Load modules
module purge
module load cuda/12.9.1
module load tensorflow/2.18

# Print loaded modules
echo "Loaded modules:"
module list

# Activate conda env
source ~/.bashrc
source activate ghifo_py38

echo "Testing ultra-fast graph building..."
python train_gnn_ultra_fast.py

echo "Test completed!" 