�	����g%@����g%@!����g%@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0����g%@0�AC�w?1/�$@Ac{-�1�?I�o'�@r0*	-��燐S@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatl?���?!v٤!2;@)8en�ݓ?1�z�P��8@:Preprocessing2E
Iterator::RootF$a�N�?!��蘆F@)&8��䝓?1�f�4U8@:Preprocessing2T
Iterator::Root::ParallelMapV2g�R@���?!�8��5@)g�R@���?1�8��5@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceS�r/0+�?!���x)@)S�r/0+�?1���x)@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�3�����?!*���{�3@)T� �!�v?1�ꔃ�@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip3SZK �?!�[JK@)^�pX�q?1��O�J@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��V%�}`?!��xyt@)��V%�}`?1��xyt@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapb0�̕�?!�}ӷ�5@)8fٓ��\?1���`��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�52.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�����J@Q=K�oG@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	0�AC�w?0�AC�w?!0�AC�w?      ��!       "	/�$@/�$@!/�$@*      ��!       2	c{-�1�?c{-�1�?!c{-�1�?:	�o'�@�o'�@!�o'�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�����J@y=K�oG@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropY�(��?!Y�(��?0"&
CudnnRNNCudnnRNNU���s�?!%Zx���?"(

concat_1_0ConcatV2�]�(e�?! �?";
gradients/split_2_grad/concatConcatV2#;h�DǕ?!�A�2X��?";
gradients/split_1_grad/concatConcatV2%Ѿ����?!=�9N�?"9
gradients/split_grad/concatConcatV2��PȖ��?!���|m�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad�]c�'�z?!�G�Ԣ�?"$
concatConcatV2����L�v?!2�}���?""
split_1Split�����u?!ɬ�h��?" 
splitSplit�0@�t?!�s1$�?Q      Y@Y�\h{
@a�T?',X@q8�0���X@y�q�$>µ?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�52.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 