#!/bin/bash
#SBATCH --job-name=ghi_forecast
#SBATCH --output=multi_gpu.out
#SBATCH --error=mult_gpu.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jfruizmu@unal.edu.co
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Increased for better data loading
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
module load tensorflow/2.6.0

# Print loaded modules
echo "Loaded modules:"
module list

# Activate conda env
source ~/.bashrc
source activate ghifo_py38

# Print GPU information
nvidia-smi
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU Available: $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))')"

# Run training script
python train_joint_fixed.py
python train_individual.py

conda deactivate
