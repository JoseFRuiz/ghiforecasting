	����A$@����A$@!����A$@	�	{���?�	{���?!�	{���?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9����A$@�u8�Jw�?1ϺFˁN@A�fF?N�?IXWj1�@Yڍ>��?r0*	q=
ף�S@2E
Iterator::Root�p����?!��^��8F@)`����#�?1I$׺[1;@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�{�5Z�?!H��'%�8@)K#f�y��?1����3�6@:Preprocessing2T
Iterator::Root::ParallelMapV2ݚt["�?!wo�E+@1@)ݚt["�?1wo�E+@1@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��_vO�?!���Θ*+@)��_vO�?1���Θ*+@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate ��h�'�?!�Iφ7@)���[1�?1��g��#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip���ם�?! 6�<�K@)���o'q?1`���@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�|�y�?!��x��9@) �t���[?1E��2�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{��X?!h\���?){��X?1h\���?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�51.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�	{���?IU�v��J@Q�46T��G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�u8�Jw�?�u8�Jw�?!�u8�Jw�?      ��!       "	ϺFˁN@ϺFˁN@!ϺFˁN@*      ��!       2	�fF?N�?�fF?N�?!�fF?N�?:	XWj1�@XWj1�@!XWj1�@B      ��!       J	ڍ>��?ڍ>��?!ڍ>��?R      ��!       Z	ڍ>��?ڍ>��?!ڍ>��?b      ��!       JGPUY�	{���?b qU�v��J@y�46T��G@