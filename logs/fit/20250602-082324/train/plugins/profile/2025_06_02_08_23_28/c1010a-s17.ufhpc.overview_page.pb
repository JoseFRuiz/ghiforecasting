�	��P���$@��P���$@!��P���$@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0��P���$@�?OI?1	4�4@A�	i�A'�?I��M(@r0*	���MbpS@2E
Iterator::Root �ҥI�?!B7$ޤ�F@)��K��$�?1�m6�^=@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�;��?!�e17o-9@)1AG�Z�?1��{57@:Preprocessing2T
Iterator::Root::ParallelMapV2B��	܊?!� ���0@)B��	܊?1� ���0@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�O=���?!���Ļ,@)�O=���?1���Ļ,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��A�"L�?!/�j7p�5@)�h>�nw?1���6n@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�&P�"��?!���![K@)�KU��o?1.E5�k�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��Q���?!1�8,8@)��A�]?1
 q��=@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��-</[?!1���@)��-</[?11���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI얧O�J@Q�iX�0G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�?OI?�?OI?!�?OI?      ��!       "		4�4@	4�4@!	4�4@*      ��!       2	�	i�A'�?�	i�A'�?!�	i�A'�?:	��M(@��M(@!��M(@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q얧O�J@y�iX�0G@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�ڄ�	��?!�ڄ�	��?0"&
CudnnRNNCudnnRNN������?!?P>��0�?";
gradients/split_2_grad/concatConcatV2x���4i�?!��R��?";
gradients/split_1_grad/concatConcatV2_�Ϯ'��?!�3y��G�?"9
gradients/split_grad/concatConcatV2�rqvr�?!W�R�6��?"(

concat_1_0ConcatV2���Q�?!��|��?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradc+�u�g{?!d5�RL(�?"$
concatConcatV2�":��w?!���0X�?""
split_1Split{j1qw?!�}??���?" 
splitSplitW1U�ԣu?!K(��E��?Q      Y@Y�\h{
@a�T?',X@qM�d��X@y��/%��?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 