	\��M�$@\��M�$@!\��M�$@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0\��M�$@j��%!q?1�2��@A��'��?I|DL�$:@r0*	�$���S@2E
Iterator::Root�GS=��?!�:,JI@)2��z�p�?1�pn�_A@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat qW�"��?!�����7@)�/-�ܑ?1a����5@:Preprocessing2T
Iterator::Root::ParallelMapV22V��W�?!��1S�.@)2V��W�?1��1S�.@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�э����?!^����'@)�э����?1^����'@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatewhX��֎?!u�k7��2@)#-��#�v?1����@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip>?�m�?!��ӵ�H@)�
E��Sp?1|���{�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��R�h\?!u��9�Z@)��R�h\?1u��9�Z@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��/�1"�?!,k>���4@)�h9�Cm[?1����#� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�51.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���C�I@Q�k9�-H@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	j��%!q?j��%!q?!j��%!q?      ��!       "	�2��@�2��@!�2��@*      ��!       2	��'��?��'��?!��'��?:	|DL�$:@|DL�$:@!|DL�$:@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���C�I@y�k9�-H@