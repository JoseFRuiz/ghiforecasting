import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

# Verify TensorFlow is using the CPU
print("Devices available:", tf.config.list_physical_devices())
print("Is built with CUDA:", tf.test.is_built_with_cuda())
