	'3�Vz�&@'3�Vz�&@!'3�Vz�&@	2��@��?2��@��?!2��@��?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9'3�Vz�&@�|�H�F�?1�٭e2L@A����[�?I��4cѤ@Y?�nJy�?r0*	̡E��}W@2E
Iterator::Root����O��?!G����K@)���t�?1�z�MX@@:Preprocessing2T
Iterator::Root::ParallelMapV2Z����?!�`�vX5@)Z����?1�`�vX5@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�)t^c��?!r7�{�9@)���jׄ�?1Y3�S5@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceU�wE�?!skO{�&@)U�wE�?1skO{�&@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipO��C�?!�>qzw�F@)>���4`p?1V��P�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorv���/Jp?!������@)v���/Jp?1������@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��ֈ`�?!m?a��6-@)��{�qi?1�OG\kq
@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapa��*�?!UqXC'0@)S%��R�W?1NU����?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�56.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92��@��?I���ԋ�L@QH��FE@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�|�H�F�?�|�H�F�?!�|�H�F�?      ��!       "	�٭e2L@�٭e2L@!�٭e2L@*      ��!       2	����[�?����[�?!����[�?:	��4cѤ@��4cѤ@!��4cѤ@B      ��!       J	?�nJy�??�nJy�?!?�nJy�?R      ��!       Z	?�nJy�??�nJy�?!?�nJy�?b      ��!       JGPUY2��@��?b q���ԋ�L@yH��FE@