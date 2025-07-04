�	6�;N�y*@6�;N�y*@!6�;N�y*@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails06�;N�y*@B@��
�?1��c${�@Ap�4(��?I+�w�7�@r0*	W-���Q@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeaty��n�U�?!��Ga��>@)�F;n�ݔ?1;���L�<@:Preprocessing2E
Iterator::Root�y9�c�?!x'�k�F@)�}�ෑ?1\��%��8@:Preprocessing2T
Iterator::Root::ParallelMapV2���W:�?!�sY\;�4@)���W:�?1�sY\;�4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice6��ā?!�
�g�(@)6��ā?1�
�g�(@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipZh�4��?!���>�KK@)�����k?16���c@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenated~�$A�?!n*2���0@)�T�-��i?1"g�@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor���pzW?!��"+C @)���pzW?1��"+C @:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap�]�o%�?!� ���2@)�ܚt["W?1�`o�( @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�54.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIύ���L@Q1r�sypE@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	B@��
�?B@��
�?!B@��
�?      ��!       "	��c${�@��c${�@!��c${�@*      ��!       2	p�4(��?p�4(��?!p�4(��?:	+�w�7�@+�w�7�@!+�w�7�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qύ���L@y1r�sypE@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�+`yy��?!�+`yy��?0"&
CudnnRNNCudnnRNN��@�&!�?!� ���?";
gradients/split_2_grad/concatConcatV2���tC�?!��&�o�?";
gradients/split_1_grad/concatConcatV2�,[�:�?!8Q�.��?"9
gradients/split_grad/concatConcatV2+�+f11�?!Y�`�� �?"(

concat_1_0ConcatV2�W}��?!��V�l�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad	��ևpx?!��(y��?"$
concatConcatV2���)Tsu?!��W�_��?""
split_1Split�N�4�s?!�j�8��?" 
splitSplit�X��os?!9��]��?Q      Y@Y�R��s@aj��PbLX@q�<,���X@y�P����?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�54.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 