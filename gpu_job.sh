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
module load cudnn/9.6.0

# Set up CUDA environment manually
export CUDA_HOME=/apps/compilers/cuda/11.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Set up cuDNN environment
export CUDNN_HOME=$HPC_cuDNN_DIR
export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$LD_LIBRARY_PATH

# Set environment variables for GPU
export CUDA_VISIBLE_DEVICES=0
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1
export TF_USE_CUDNN=1
export TF_CUDNN_USE_AUTOTUNE=0  # Disable autotune for better compatibility
export TF_CUDNN_DETERMINISTIC=1

# Disable TensorRT warnings
export TF_ENABLE_AUTO_MIXED_PRECISION=0
export TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS=1

# Activate conda env
source ~/.bashrc
source activate ghifo_py39

# Print environment information
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDNN_HOME: $CUDNN_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Print GPU information
nvidia-smi
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo "CUDA version: $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_build_info()[\"cuda_version\"])')"
echo "cuDNN version: $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_build_info()[\"cudnn_version\"])')"

# Run training script with environment variables
CUDA_VISIBLE_DEVICES=0 python train_joint.py

conda deactivate
