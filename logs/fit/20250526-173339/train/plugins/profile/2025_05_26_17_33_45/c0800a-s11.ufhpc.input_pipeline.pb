	TW>��@TW>��@!TW>��@	�I>#��@�I>#��@!�I>#��@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9TW>��@��Rx��?1�Գ ��@A|��S:�?I�$?�WL@Y�eM,��?r0*	R����U@2E
Iterator::Root&����?!Lbݒ�TF@)�bb�qm�?1�`s0�^;@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat6ɏ�k�?!���oF\;@)�pY�� �?1�]DW�9@:Preprocessing2T
Iterator::Root::ParallelMapV2�w�~܎?!dG�5J1@)�w�~܎?1dG�5J1@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�?�@�w�?!��q�4@)`cD�Ђ?1���%@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicedyW=`�?!�V9@M$@)dyW=`�?1�V9@M$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipM�Nϻ��?!��"mo�K@)��L�nq?1���s�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap.�R���?!�d�{{7@)a���)a?1@��O�:@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�e3��V?!��o��^�?)�e3��V?1��o��^�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�47.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�I>#��@IP�a�<H@Q}j�;H@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Rx��?��Rx��?!��Rx��?      ��!       "	�Գ ��@�Գ ��@!�Գ ��@*      ��!       2	|��S:�?|��S:�?!|��S:�?:	�$?�WL@�$?�WL@!�$?�WL@B      ��!       J	�eM,��?�eM,��?!�eM,��?R      ��!       Z	�eM,��?�eM,��?!�eM,��?b      ��!       JGPUY�I>#��@b qP�a�<H@y}j�;H@