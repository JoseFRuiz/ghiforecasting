	����g(@����g(@!����g(@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0����g(@a\:�<�?1BȗPQ@Avq�-�?I��%!�@r0*	�A`��RT@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�����?!�Є`��<@)����0��?1��V�:@:Preprocessing2E
Iterator::Root@�t�_�?!c�$�F@)4�Op��?1�����t6@:Preprocessing2T
Iterator::Root::ParallelMapV2�KR�b�?!=`���5@)�KR�b�?1=`���5@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��3KԄ?!�l"*2)@)��3KԄ?1�l"*2)@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�H��_��?!QǗ;��3@)�mm�y�x?1aC�4�@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip��t?�?!�U�Q�K@)k�ѯ�o?1����@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV�F�a?!y�}M��@)V�F�a?1y�}M��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�	m9��?!"�r  U6@)Ow�x�`?1�.�&�S@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���#K@QZC�l�F@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	a\:�<�?a\:�<�?!a\:�<�?      ��!       "	BȗPQ@BȗPQ@!BȗPQ@*      ��!       2	vq�-�?vq�-�?!vq�-�?:	��%!�@��%!�@!��%!�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���#K@yZC�l�F@