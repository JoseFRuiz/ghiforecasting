	���/�&*@���/�&*@!���/�&*@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���/�&*@�Ҩ���?1=�U�;@AL��1%�?I�3��g@r0*	��|?5�T@2E
Iterator::Root!�> �M�?!�����G@)�z�G�?1/n�wV�?@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat��;��?!�b_�H*<@)�<��@�?1��ٟ�::@:Preprocessing2T
Iterator::Root::ParallelMapV2��Ҥt�?!і�0.0@)��Ҥt�?1і�0.0@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceڮ��؀?!M���#@)ڮ��؀?1M���#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate|)<hv݋?!�1�Ll0@)E���V	v?1����@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipO��C�?!�}x<J@)Ow�x�p?1Q��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap �4�O�?!��Zl:3@)r���	c?1�K{�p@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���$xCZ?!�sZXQ��?)���$xCZ?1�sZXQ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�56.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIu�܇"�L@Q�e#x�@E@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Ҩ���?�Ҩ���?!�Ҩ���?      ��!       "	=�U�;@=�U�;@!=�U�;@*      ��!       2	L��1%�?L��1%�?!L��1%�?:	�3��g@�3��g@!�3��g@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qu�܇"�L@y�e#x�@E@