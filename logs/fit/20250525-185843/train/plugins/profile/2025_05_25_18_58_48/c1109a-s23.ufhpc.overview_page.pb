�	�i>"&@�i>"&@!�i>"&@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�i>"&@vS�k%t�?1EF$a@A¥c�3��?I> Й�	@r0*	bX9�R@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat��a���?!10C<<@)�2��V�?1� 6M�:@:Preprocessing2T
Iterator::Root::ParallelMapV2~ R�8��?!�#;�/f5@)~ R�8��?1�#;�/f5@:Preprocessing2E
Iterator::Root>�#d �?!O�9�D@)�ګ���?1Q� �4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�`U��N�?!����*@)�`U��N�?1����*@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�t?� �?!^h}~E�4@)"S>U�w?1!r����@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipx_���?!��9�� M@)��1ZGUs?1h��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap4d<J%<�?!�b��q@7@)��!��Z?1��	a	@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��{�qY?!�yi��)@)��{�qY?1�yi��)@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�47.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIz6�i�I@Q��_��:H@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	vS�k%t�?vS�k%t�?!vS�k%t�?      ��!       "	EF$a@EF$a@!EF$a@*      ��!       2	¥c�3��?¥c�3��?!¥c�3��?:	> Й�	@> Й�	@!> Й�	@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qz6�i�I@y��_��:H@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�%�Iy{�?!�%�Iy{�?0"&
CudnnRNNCudnnRNN٬��{C�?!j�}z_�?";
gradients/split_2_grad/concatConcatV2c�:u�T�?!��P":�?"9
gradients/split_grad/concatConcatV2��:C�?!ѪB]˹�?";
gradients/split_1_grad/concatConcatV2 ���|��?!�]nP�5�?"(

concat_1_0ConcatV2��C�B�?!rl���?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad��Z0�&�?!�ה0��?"$
concatConcatV2�q�uV�|?!������?" 
splitSplitjEĳ��z?!E�P:�?""
split_1Split3Ss˦z?!�+��o�?Q      Y@Y9��8�c@a�q�YW@q�i|�fZC@y��(�m�?"�
both�Your program is POTENTIALLY input-bound because 3.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�47.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�38.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 