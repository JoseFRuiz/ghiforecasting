�	`��8m$@`��8m$@!`��8m$@	���h/�?���h/�?!���h/�?"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9`��8m$@cb�qm�h?1" 8��I@A������?I�E��(&@Y���<HO�?r0*	㥛� �]@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�2��A��?!*�fƈB@):\�=셢?19�k�	>>@:Preprocessing2E
Iterator::Root�S����?!e����A@)��e�-�?1yi�9��7@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat���=��?!Q�{�L�/@)�����?1�����-@:Preprocessing2T
Iterator::Root::ParallelMapV2�}�Az��?!�N��L'@)�}�Az��?1�N��L'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip��a�ó?!Λ��!P@)��w�Go�?1>��n�#@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�Y��U��?!0;"N@)�Y��U��?10;"N@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap~�4bf��?!x�O�HC@)��Քd]?1=ˌ ���?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�R����W?!�MEvr�?)�R����W?1�MEvr�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�51.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���h/�?IS�4�J@Q�b�(|�G@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	cb�qm�h?cb�qm�h?!cb�qm�h?      ��!       "	" 8��I@" 8��I@!" 8��I@*      ��!       2	������?������?!������?:	�E��(&@�E��(&@!�E��(&@B      ��!       J	���<HO�?���<HO�?!���<HO�?R      ��!       Z	���<HO�?���<HO�?!���<HO�?b      ��!       JGPUY���h/�?b qS�4�J@y�b�(|�G@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop���~��?!���~��?0"&
CudnnRNNCudnnRNN\<^p���?!f�ܧ �?";
gradients/split_2_grad/concatConcatV2F{B����?!@������?"9
gradients/split_grad/concatConcatV2�"1&P��?!��!K;�?";
gradients/split_1_grad/concatConcatV2,
�~�ӈ?!�5���?"(

concat_1_0ConcatV2
޶�@ʂ?!kD���?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad
�>�j�{?!Î��!�?"$
concatConcatV2�*����w?!�VP�?""
split_1Split^�g%j0v?!|ZM�|�?" 
splitSplit�Ǧ%D�u?!��i��?Q      Y@Y��� �
@a����/X@q�Pܰ
�X@yO�2'�?"�
device�Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�51.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 