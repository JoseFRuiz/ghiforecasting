	p�h��#@p�h��#@!p�h��#@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0p�h��#@`cD�В?1����a�@AP�����?I��H��S@r0*	<�O���T@2E
Iterator::Root��R$_	�?!b�
��LG@)vS�k%t�?1��GfF;@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat|���G��?!mUtK��<@)������?1��p:@:Preprocessing2T
Iterator::Root::ParallelMapV2�7�ܘ��?!ȇ�ݔS3@)�7�ܘ��?1ȇ�ݔS3@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�*P��Ä?!��8��%(@)�*P��Ä?1��8��%(@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/M��.�?!�����1@)�D��f�r?1.P�d��@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipQ������?!�T�])�J@)�fh<q?1��&�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapb�*�3�?!⤬G�3@)�l�_?1�ަ�j@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�8�Վ�\?!�R�|� @)�8�Վ�\?1�R�|� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�48.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��j�QYH@Q|�$��I@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	`cD�В?`cD�В?!`cD�В?      ��!       "	����a�@����a�@!����a�@*      ��!       2	P�����?P�����?!P�����?:	��H��S@��H��S@!��H��S@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��j�QYH@y|�$��I@