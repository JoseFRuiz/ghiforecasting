	G��ұ@G��ұ@!G��ұ@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0G��ұ@|_\���?1����O�@At�^���?I����@r0*	��x�&�R@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat4K�Բ�?!�H[��;@)���_w��?1`��b9@:Preprocessing2E
Iterator::RootuʣaQ�?!����HF@)�x?n�|�?1�u����7@:Preprocessing2T
Iterator::Root::ParallelMapV2.�&�?!�_�A}�4@).�&�?1�_�A}�4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceFCƣT?!+D{�l)@)FCƣT?1+D{�l)@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��҈�}�?!������3@)f�ʉvu?1�R�!*�@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip-�Yf��?!W�pK�K@)C���-r?1��5d@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorV��Dׅ_?!�g��G@)V��Dׅ_?1�g��G@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�A��Ր?!ph9ԩ5@)��{�qY?1��U��^ @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�47.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI%Z�J@Q�ڥ�n�G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	|_\���?|_\���?!|_\���?      ��!       "	����O�@����O�@!����O�@*      ��!       2	t�^���?t�^���?!t�^���?:	����@����@!����@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q%Z�J@y�ڥ�n�G@