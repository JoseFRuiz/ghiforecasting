	���D�@���D�@!���D�@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0���D�@ddY0��?1�r�Sr�@A!�˛Õ?I6ɏ�K@r0*	�I+OS@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatB?S�[�?!�`#s�;@)��C�b�?1�y@t�9@:Preprocessing2E
Iterator::Root�u�X��?!I��1�E@)Ϡ��?1n���c7@:Preprocessing2T
Iterator::Root::ParallelMapV2Y�_"�:�?!��=�3@)Y�_"�:�?1��=�3@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicehx�輸?!�y��'@)hx�輸?1�y��'@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��25	ސ?!uٙS5@) �d�F ~?1%R9�#@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipI�Q}�?!�D�oL@)��C���r?1h�ո�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�ù�Z?!�7� @)�ù�Z?1�7� @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[{C�?!��RC7@)v�A]�PV?1���6�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�48.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�N��taJ@QM�SF��G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	ddY0��?ddY0��?!ddY0��?      ��!       "	�r�Sr�@�r�Sr�@!�r�Sr�@*      ��!       2	!�˛Õ?!�˛Õ?!!�˛Õ?:	6ɏ�K@6ɏ�K@!6ɏ�K@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�N��taJ@yM�SF��G@