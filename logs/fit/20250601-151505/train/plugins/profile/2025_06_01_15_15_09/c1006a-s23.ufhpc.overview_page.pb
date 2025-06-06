�	C�O�}�(@C�O�}�(@!C�O�}�(@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0C�O�}�(@�:�*��?15ӽN�@A>�h��?I�_��C@r0*	����M�Z@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��a��?!C�V��C@)����P1�?1Kȝ9�;@:Preprocessing2E
Iterator::Root�tۈ'�?!���A@)pxADjڕ?11�<4@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�	���I�?!����Ql5@)8�*5{��?1�o����3@:Preprocessing2T
Iterator::Root::ParallelMapV2�p�q�t�?!�jWF.@)�p�q�t�?1�jWF.@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���ο]�?!u���$@)���ο]�?1u���$@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip�������?!����0P@)l��C6p?1)����@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap���Or��?!�8� ��C@)���=�Z?1epN�+��?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorCY��Z�Z?!8��}v�?)CY��Z�Z?18��}v�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI%P��K@Qۯf��F@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:�*��?�:�*��?!�:�*��?      ��!       "	5ӽN�@5ӽN�@!5ӽN�@*      ��!       2	>�h��?>�h��?!>�h��?:	�_��C@�_��C@!�_��C@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q%P��K@yۯf��F@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop�;7�'��?!�;7�'��?0"&
CudnnRNNCudnnRNN��`�2�?!X��:���?";
gradients/split_2_grad/concatConcatV2�������?!d��i�?"9
gradients/split_grad/concatConcatV2��I�ky�?!!;�@���?";
gradients/split_1_grad/concatConcatV2��;UB>�?!�)J��?"(

concat_1_0ConcatV2�+��y`�?!42�1%j�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradKA�%��x?!���^��?"$
concatConcatV2��;�"fu?!�>��*��?""
split_1Split��4t?!��1a��?" 
splitSplitE03�s?!\{�c�?Q      Y@Y�R��s@aj��PbLX@q,jx�X@yN�5[��?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 