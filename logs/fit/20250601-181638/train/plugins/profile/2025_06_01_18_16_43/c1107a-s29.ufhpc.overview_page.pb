�	�^��(@�^��(@!�^��(@      ��!       "q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0�^��(@���r-Z�?1N�@�CJ@Afk}�Ж�?I��dT@r0*	ףp=�T@2E
Iterator::Root���B�?!u墈m�D@)�W�\�?1���&�7@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat�K�uT�?!��?d9@)=�u��?1��SN<�7@:Preprocessing2T
Iterator::Root::ParallelMapV2gҦ�ٌ?!��4K�+1@)gҦ�ٌ?1��4K�+1@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate)�ahur�?!������:@)~�֤��?1!w[�%�-@:Preprocessing2�
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���+҃?!rGE�'@)���+҃?1rGE�'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip8�k����?!�]w�sM@)r1�q�p?1�y��{8@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap76;R}�?!����t<@)�Д�~PW?1@띡��?:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�G��V?!���:0�?)�G��V?1���:0�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"�53.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI;>F��ZK@Q���H0�F@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���r-Z�?���r-Z�?!���r-Z�?      ��!       "	N�@�CJ@N�@�CJ@!N�@�CJ@*      ��!       2	fk}�Ж�?fk}�Ж�?!fk}�Ж�?:	��dT@��dT@!��dT@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q;>F��ZK@y���H0�F@�"P
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop%�࿺��?!%�࿺��?0"&
CudnnRNNCudnnRNNPB���V�?!M䣢-��?";
gradients/split_2_grad/concatConcatV2��Xfڙ�?!D���et�?"9
gradients/split_grad/concatConcatV2¤9���?!ן9����?";
gradients/split_1_grad/concatConcatV2[f��k2�?!pUCg)�?"(

concat_1_0ConcatV2���3G}�?!RT�_\w�?"S
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGradg1���x?!���<���?""
split_1Split�$z	4Vt?!���^��?"$
concatConcatV2�(�&Jt?!Y�����?" 
splitSplitx�s���r?!(���?Q      Y@Y�R��s@aj��PbLX@q.>�L�X@yT09��?"�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
high�53.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�99.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Ampere)(: B 