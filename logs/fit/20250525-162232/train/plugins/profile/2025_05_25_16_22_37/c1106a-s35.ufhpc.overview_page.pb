�	c��K�Q@c��K�Q@!c��K�Q@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0c��K�Q@�vi�a�?1�z�ю�@AxADj�Ŕ?I�5�ڋ@r0*	MbX9�Q@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatt��%�?!?�=���<@)J|����?1<��]�:@:Preprocessing2T
Iterator::Root::ParallelMapV27�X�O�?!6_���5@)7�X�O�?16_���5@:Preprocessing2E
Iterator::Root��$y��?!���B��E@)v��2SZ�?1��z�5d5@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice �4�O�?!|GH/YB&@) �4�O�?1|GH/YB&@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateJ��	�y�?!�K!��2@)����Sv?1���w@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�O��ͤ?!fT�`bL@)���� �s?1^NE@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�����?!ʯ�=�#5@)�V���\?1y ��&@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorg|_\��V?!���$�?)g|_\��V?1���$�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�47.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI]����I@Q�`U�@uH@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�vi�a�?�vi�a�?!�vi�a�?      ��!       "	�z�ю�@�z�ю�@!�z�ю�@*      ��!       2	xADj�Ŕ?xADj�Ŕ?!xADj�Ŕ?:	�5�ڋ@�5�ڋ@!�5�ڋ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q]����I@y�`U�@uH@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropi�ۺ���?!i�ۺ���?0"&
CudnnRNNCudnnRNN ��H�?!�����d�?";
gradients/split_2_grad/concatConcatV2,�}?$�?!�����=�?";
gradients/split_1_grad/concatConcatV2Qd�<��?!�����?"9
gradients/split_grad/concatConcatV2�/�d��?!���Zs3�?"(

concat_1_0ConcatV28r��V��?!j��ԅ�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad}ۚ�R�?!�h��E��?"$
concatConcatV2��Kһ�{?!~ 1w3�?""
split_1Split�ə;Yz?! �d��6�?" 
splitSplit�no42z?!�r�Jk�?Q      Y@Y9��8�c@a�q�YW@q�XZ��B@y��)�<��?"�
both�Your program is POTENTIALLY input-bound because 3.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�47.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�37.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 