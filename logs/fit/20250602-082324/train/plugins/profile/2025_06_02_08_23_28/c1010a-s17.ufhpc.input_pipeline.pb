	��P���$@��P���$@!��P���$@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0��P���$@�?OI?1	4�4@A�	i�A'�?I��M(@r0*	���MbpS@2E
Iterator::Root �ҥI�?!B7$ޤ�F@)��K��$�?1�m6�^=@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�;��?!�e17o-9@)1AG�Z�?1��{57@:Preprocessing2T
Iterator::Root::ParallelMapV2B��	܊?!� ���0@)B��	܊?1� ���0@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�O=���?!���Ļ,@)�O=���?1���Ļ,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��A�"L�?!/�j7p�5@)�h>�nw?1���6n@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�&P�"��?!���![K@)�KU��o?1.E5�k�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��Q���?!1�8,8@)��A�]?1
 q��=@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��-</[?!1���@)��-</[?11���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI얧O�J@Q�iX�0G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�?OI?�?OI?!�?OI?      ��!       "		4�4@	4�4@!	4�4@*      ��!       2	�	i�A'�?�	i�A'�?!�	i�A'�?:	��M(@��M(@!��M(@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q얧O�J@y�iX�0G@