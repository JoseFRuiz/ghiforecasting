	U0*�+@U0*�+@!U0*�+@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0U0*�+@c_��`��?1�\��X�@A�%!���?I�&�5�@r0*	�|?5^�V@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatW	�3��?!	7�<@)�G��[�?1s�A�:@:Preprocessing2E
Iterator::Root<O<g�?!R���wF@)�=]ݱؖ?1r��e6h8@:Preprocessing2T
Iterator::Root::ParallelMapV2�`�d7�?!��h�n�4@)�`�d7�?1��h�n�4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����qn�?!fX�B>�$@)����qn�?1fX�B>�$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�5�U�ũ?!�xo-�K@)���z?1C$�h�@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenatea��*�?!����0@)~�.rOw?1c�R�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�\��J�?!����ӊ3@)��f��e?1c�̄}@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�>s֧c?!B���j@)�>s֧c?1B���j@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�56.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIߴ�-gM@Q!K=Ҙ�D@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	c_��`��?c_��`��?!c_��`��?      ��!       "	�\��X�@�\��X�@!�\��X�@*      ��!       2	�%!���?�%!���?!�%!���?:	�&�5�@�&�5�@!�&�5�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qߴ�-gM@y!K=Ҙ�D@