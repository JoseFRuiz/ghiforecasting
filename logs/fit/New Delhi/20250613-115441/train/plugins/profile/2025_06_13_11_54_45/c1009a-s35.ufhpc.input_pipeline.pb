	`��8m$@`��8m$@!`��8m$@	���h/�?���h/�?!���h/�?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9`��8m$@cb�qm�h?1" 8��I@A������?I�E��(&@Y���<HO�?r0*	㥛� �]@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�2��A��?!*�fƈB@):\�=셢?19�k�	>>@:Preprocessing2E
Iterator::Root�S����?!e����A@)��e�-�?1yi�9��7@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat���=��?!Q�{�L�/@)�����?1�����-@:Preprocessing2T
Iterator::Root::ParallelMapV2�}�Az��?!�N��L'@)�}�Az��?1�N��L'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip��a�ó?!Λ��!P@)��w�Go�?1>��n�#@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Y��U��?!0;"N@)�Y��U��?10;"N@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap~�4bf��?!x�O�HC@)��Քd]?1=ˌ ���?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�R����W?!�MEvr�?)�R����W?1�MEvr�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�51.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���h/�?IS�4�J@Q�b�(|�G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	cb�qm�h?cb�qm�h?!cb�qm�h?      ��!       "	" 8��I@" 8��I@!" 8��I@*      ��!       2	������?������?!������?:	�E��(&@�E��(&@!�E��(&@B      ��!       J	���<HO�?���<HO�?!���<HO�?R      ��!       Z	���<HO�?���<HO�?!���<HO�?b      ��!       JGPUY���h/�?b qS�4�J@y�b�(|�G@