�	�Ù_�)%@�Ù_�)%@!�Ù_�)%@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�Ù_�)%@N^��Y?1ڬ�\me@AN��1�M�?I��u���@r0*	G�z��S@2T
Iterator::Root::ParallelMapV2�,g~�?!�Ch��G:@)�,g~�?1�Ch��G:@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat,����?!?pfl8@)�xy:W��?1t�����6@:Preprocessing2E
Iterator::Root�K����?!*}EX�MH@)��(_�B�?1a�"'�S6@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice4�y�S��?!����j#)@)4�y�S��?1����j#)@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip��2��?!ւ��5�I@)X�%���s?1H=X�A@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�~� ��?!�Y�vS2@)�9:Z�r?1ң��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��/�1"�?!B��m��4@)a���)a?1�1���@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�'*�TV?!Oz�*�M�?)�'*�TV?1Oz�*�M�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�51.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI!Z���I@Qߥ�"H@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	N^��Y?N^��Y?!N^��Y?      ��!       "	ڬ�\me@ڬ�\me@!ڬ�\me@*      ��!       2	N��1�M�?N��1�M�?!N��1�M�?:	��u���@��u���@!��u���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q!Z���I@yߥ�"H@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackpropـv�$J�?!ـv�$J�?0"&
CudnnRNNCudnnRNNQA �Ī�?!�P{���?"L
3model_1/batch_normalization_7/AssignMovingAvg_1/mulMul��i�F�?!�,T�?";
gradients/split_2_grad/concatConcatV2v:
˰3�?!�A���?";
gradients/split_1_grad/concatConcatV2�h��K�?!-�T�n�?"9
gradients/split_grad/concatConcatV2��/0\$�?!���Ċ��?"(

concat_1_0ConcatV2�-Oq�ށ?!�ू�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad ��~��z?!`���L�?"$
concatConcatV2�I�ԏuw?!��L�
{�?" 
splitSplit����t?!&�ƶ���?Q      Y@Y��� �
@a����/X@qv;&o�X@y��:uuM�?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�51.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 