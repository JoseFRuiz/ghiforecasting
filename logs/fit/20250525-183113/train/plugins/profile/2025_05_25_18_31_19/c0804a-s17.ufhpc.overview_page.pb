�	e�VA�@e�VA�@!e�VA�@	>��*|�?>��*|�?!>��*|�?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9e�VA�@�D��]�?1T�J�Ó@A]��$?�?IgHū�@YgDio���?r0*	�/�$�P@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat,����?!�`���=@)�*8� �?1��tb:@:Preprocessing2E
Iterator::Root�]~p�?!S �&F@)9DܜJ�?1빧�R7@:Preprocessing2T
Iterator::Root::ParallelMapV2<��fԌ?!!���4@)<��fԌ?1!���4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��IӠh~?!^em &!&@)��IӠh~?1^em &!&@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate"r�z�f�?!P8�F|2@)�%�"�dt?1�~�ή@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipn��)"�?!���� �K@)ۥ���o?1�Ӌɧ�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�� @��]?!u�{6m�@)�� @��]?1u�{6m�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapJ���?!D�U�4@)��M�qZ?1�_�@{>@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 7.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9>��*|�?I�BتNI@Q(��s�G@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�D��]�?�D��]�?!�D��]�?      ��!       "	T�J�Ó@T�J�Ó@!T�J�Ó@*      ��!       2	]��$?�?]��$?�?!]��$?�?:	gHū�@gHū�@!gHū�@B      ��!       J	gDio���?gDio���?!gDio���?R      ��!       Z	gDio���?gDio���?!gDio���?b      ��!       JGPUY>��*|�?b q�BتNI@y(��s�G@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�N��w�?!�N��w�?0"&
CudnnRNNCudnnRNN�}wm;�?!��e��Y�?";
gradients/split_2_grad/concatConcatV2�5BJ��?!h�иA�?"9
gradients/split_grad/concatConcatV2B�b�4�?!qz��I��?";
gradients/split_1_grad/concatConcatV2�U��u�?!ɯ��;�?"(

concat_1_0ConcatV2��M��?!��$E��?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad��R���?!39����?"$
concatConcatV2��*}9�|?!��2�
�?" 
splitSplit �pH�z?!�p�N�?�?""
split_1Split�!�ݭqz?!�W��t�?Q      Y@Y9��8�c@a�q�YW@q�깑��<@y��0|���?"�
both�Your program is POTENTIALLY input-bound because 7.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�43.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�28.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 