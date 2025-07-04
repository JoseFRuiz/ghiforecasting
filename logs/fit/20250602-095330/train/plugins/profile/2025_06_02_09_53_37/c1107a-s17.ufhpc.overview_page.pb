�	j�t�,%@j�t�,%@!j�t�,%@	3��a-�?3��a-�?!3��a-�?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9j�t�,%@%���wz?1!�> �M@A#������?I�o�4(�@Y��>+�?r0*	�E����X@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatٯ;�y�?!a,c�9@)�O��e�?1�T
;�8@:Preprocessing2T
Iterator::Root::ParallelMapV2,g~5�?!l0"��7@),g~5�?1l0"��7@:Preprocessing2E
Iterator::RootXᖏ���?!S����MG@)��ƠB�?1;e��6@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��뉮�?!��T��.@)��뉮�?1��T��.@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@�z��{�?!K7�>�,5@)�����w?1�VQ,�@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Ziph��s��?!�[< 6�J@)zUg��s?1�Ė��@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap��?�@��?!˖�.�*7@)�Z'.�+`?1������?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS%��R�W?!�M�v�?)S%��R�W?1�M�v�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no92��a-�?I�̎�� K@Q��;�F@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	%���wz?%���wz?!%���wz?      ��!       "	!�> �M@!�> �M@!!�> �M@*      ��!       2	#������?#������?!#������?:	�o�4(�@�o�4(�@!�o�4(�@B      ��!       J	��>+�?��>+�?!��>+�?R      ��!       Z	��>+�?��>+�?!��>+�?b      ��!       JGPUY2��a-�?b q�̎�� K@y��;�F@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�$)����?!�$)����?0"&
CudnnRNNCudnnRNNm��É�?!�yt��$�?";
gradients/split_2_grad/concatConcatV2��h?Vw�?!
�oq|��?"9
gradients/split_grad/concatConcatV2%���[�?!G����=�?";
gradients/split_1_grad/concatConcatV2�O�"Ɉ?!�8_��?"(

concat_1_0ConcatV2����g�?!}J����?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad#��V�{?!\�X�!�?"$
concatConcatV2������w?!q�T>Q�?""
split_1Split��r#��v?!�fIV�~�?" 
splitSplit�?4�Iu?!��,I��?Q      Y@Y��� �
@a����/X@q�W3Ѳ�0@y(�+�'�?"�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�16.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 