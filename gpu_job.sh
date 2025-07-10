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
#SBATCH --partition=hpg-b200 # instead of gpu
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00

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

# Set environment variables for better GPU compatibility
export TF_CPP_MIN_LOG_LEVEL=2  # Reduce TensorFlow logging
export CUDA_VISIBLE_DEVICES=0   # Use first GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Allow GPU memory growth

# Print GPU information
nvidia-smi
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "GPU Available: $(python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))')"

# Check package versions
echo "Checking package versions:"
python -c "import spektral; print(f'Spektral version: {spektral.__version__}')"
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

# Test GPU functionality
echo "Testing GPU functionality..."
python -c "
import tensorflow as tf
print('TensorFlow GPU test:')
print(f'GPU devices: {tf.config.list_physical_devices(\"GPU\")}')
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f'GPU computation test: {c}')
        print('GPU test successful!')
else:
    print('No GPU devices found!')
"

# Run training script
# python train_joint_fixed.py
# python train_individual.py
python train_gnn.py

conda deactivate
