�	��kЗ�@��kЗ�@!��kЗ�@	߈x�$@߈x�$@!߈x�$@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9��kЗ�@�b��J�?1!yv��@An�HJz�?I��Th@Y��U�P��?r0*	�v���R@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�=�-�?!vLݽ;>@)����(y�?1�~��<@:Preprocessing2E
Iterator::RootxD����?!1��A�D@)�el�f�?1�K9�n�5@:Preprocessing2T
Iterator::Root::ParallelMapV2�1 ǎ?!s��4@)�1 ǎ?1s��4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlices.�Ue߅?!�!��j�,@)s.�Ue߅?1�!��j�,@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateU3k) �?!�:p���4@)�	L�ut?1�����:@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�>;�b�?!��8�3M@)m7�7M�m?1>-Z6R@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap+�@.q�?!�1��0W7@)����^?1ݵo� "@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�t?� ?[?!�mṅ�@)�t?� ?[?1�mṅ�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�44.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��x�$@Iya�sG@Qdb/�!
I@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�b��J�?�b��J�?!�b��J�?      ��!       "	!yv��@!yv��@!!yv��@*      ��!       2	n�HJz�?n�HJz�?!n�HJz�?:	��Th@��Th@!��Th@B      ��!       J	��U�P��?��U�P��?!��U�P��?R      ��!       Z	��U�P��?��U�P��?!��U�P��?b      ��!       JGPUY��x�$@b qya�sG@ydb/�!
I@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop/(��g�?!/(��g�?0"&
CudnnRNNCudnnRNNNJ6hB�?!�</�#U�?";
gradients/split_2_grad/concatConcatV2�_B
=$�?!�O��E6�?";
gradients/split_1_grad/concatConcatV2���'�?!�sHF��?"9
gradients/split_grad/concatConcatV2��TFI��?!=�ak[-�?"(

concat_1_0ConcatV2H���+�?!&����?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad�s��?!���O���?"$
concatConcatV2�)y(Ă|?!H���	�?""
split_1Split���0{?!O
W�b?�?" 
splitSplit�HOK,z?!ᨓm�s�?Q      Y@YI�$I�$@aܶm۶MW@qel�:`�1@y��o�O��?"�
device�Your program is NOT input-bound because only 3.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�44.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�17.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 