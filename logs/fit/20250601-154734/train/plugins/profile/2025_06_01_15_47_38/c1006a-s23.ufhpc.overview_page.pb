�	�70�QL)@�70�QL)@!�70�QL)@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�70�QL)@��FtϺ�?1Mi�-�@A�C�.l�?I^c���z@r0*	Y9��v�\@2T
Iterator::Root::ParallelMapV2���}���?!�pTm:4C@)���}���?1�pTm:4C@:Preprocessing2E
Iterator::Root��gϱ?!p�xRN@)ϺFˁ�?12���;6@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�fh<�?!��I��1@)�-�R�?1�~5�1/@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateN�t"��?!0uЛ�1@)�.Q�5��?1M\��v"@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���+҃?!�L�� @)���+҃?1�L�� @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipXs�`��?!�/��C@)k�ѯ�o?1�N{��
@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorҏ�S��[?!1{�� ��?)ҏ�S��[?11{�� ��?:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[[x^*6�?!�;�*#�2@)�Д�~PW?1�e��u��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�54.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI���'�K@Qv��SF@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��FtϺ�?��FtϺ�?!��FtϺ�?      ��!       "	Mi�-�@Mi�-�@!Mi�-�@*      ��!       2	�C�.l�?�C�.l�?!�C�.l�?:	^c���z@^c���z@!^c���z@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���'�K@yv��SF@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop>_d]ˏ�?!>_d]ˏ�?0"&
CudnnRNNCudnnRNN�V�)I(�?!�
;���?";
gradients/split_2_grad/concatConcatV2%H1g��?!�Ƨ2;n�?";
gradients/split_1_grad/concatConcatV2���C�?!��;J��?"9
gradients/split_grad/concatConcatV2�������?!ly-��?"(

concat_1_0ConcatV2^�/ԃ��?!y��<�i�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad��'���x?!����?"$
concatConcatV2.�� ?�u?!}U0��?""
split_1Split�h
/�t?!�u�4��?" 
splitSplit�lknt^s?!�hRw��?Q      Y@YDq�@a��u<JX@q<�b�$�X@yt,����?"�
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
Refer to the TF2 Profiler FAQb�99.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 