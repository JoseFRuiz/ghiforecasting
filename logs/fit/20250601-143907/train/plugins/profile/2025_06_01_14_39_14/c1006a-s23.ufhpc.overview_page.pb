�	M�d��'@M�d��'@!M�d��'@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0M�d��'@M���D�?173��p�@A��3�c��?I�0|DL�@r0*	*\����U@2E
Iterator::Root�:�G�?!`�eO��G@)��mr�?1�V��3;@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatt�%z��?!%z9k�2:@)d:tzލ�?1��lɽ�7@:Preprocessing2T
Iterator::Root::ParallelMapV2�\�&��?!,�6&4@)�\�&��?1,�6&4@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice䃞ͪυ?!|*/~�D(@)䃞ͪυ?1|*/~�D(@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���{�?!*f�_�s3@)�H� Oz?1�Co�F@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�J��q��?!�>��SJ@)G���R{q?1`NⳲs@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�`��_?!�eU�@)�`��_?1�eU�@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[`���f�?!�oIR�5@)�ôo�^?1�JxH�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�50.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI-˵��J@Q�4J N�G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	M���D�?M���D�?!M���D�?      ��!       "	73��p�@73��p�@!73��p�@*      ��!       2	��3�c��?��3�c��?!��3�c��?:	�0|DL�@�0|DL�@!�0|DL�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q-˵��J@y�4J N�G@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop=��=��?!=��=��?0"&
CudnnRNNCudnnRNN}��R.J�?!|~�$U��?";
gradients/split_2_grad/concatConcatV2��7wO�?!���-t�?"9
gradients/split_grad/concatConcatV2����}�?!�5:e!��?";
gradients/split_1_grad/concatConcatV2zz_wJ�?!r�BK'�?"(

concat_1_0ConcatV2�DxF"�?!� �˯s�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad�r��x?!���T��?"$
concatConcatV2�Kٕhu?!!�F&��?""
split_1Split=���4Qt?!X����?" 
splitSplit�@��'s?!�r��?Q      Y@Y�R��s@aj��PbLX@q�^p�I�=@y�2Z��?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�50.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�29.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 