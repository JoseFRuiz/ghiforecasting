	����g%@����g%@!����g%@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0����g%@0�AC�w?1/�$@Ac{-�1�?I�o'�@r0*	-��燐S@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatl?���?!v٤!2;@)8en�ݓ?1�z�P��8@:Preprocessing2E
Iterator::RootF$a�N�?!��蘆F@)&8��䝓?1�f�4U8@:Preprocessing2T
Iterator::Root::ParallelMapV2g�R@���?!�8��5@)g�R@���?1�8��5@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS�r/0+�?!���x)@)S�r/0+�?1���x)@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�3�����?!*���{�3@)T� �!�v?1�ꔃ�@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip3SZK �?!�[JK@)^�pX�q?1��O�J@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��V%�}`?!��xyt@)��V%�}`?1��xyt@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapb0�̕�?!�}ӷ�5@)8fٓ��\?1���`��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�52.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����J@Q=K�oG@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0�AC�w?0�AC�w?!0�AC�w?      ��!       "	/�$@/�$@!/�$@*      ��!       2	c{-�1�?c{-�1�?!c{-�1�?:	�o'�@�o'�@!�o'�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����J@y=K�oG@