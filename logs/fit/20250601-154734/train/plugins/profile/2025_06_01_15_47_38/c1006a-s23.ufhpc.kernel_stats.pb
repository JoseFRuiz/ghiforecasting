
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)Z ��*�28��@�GH�QbCudnnRNNh0u  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�@�CH�HXb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �A
�
�void LSTM_elementWise_bp1<float, float, float>(int, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, int, int, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float)*�28��@�.H�4Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �B
�
�void elemWiseRNNcell<float, float, float, (cudnnRNNMode_t)2, (cudnnRNNBiasMode_t)2>(int, int, int, int, int, bool, float const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float) *�28�@�/H�6bCudnnRNNh0u  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2 8��@�*H�/Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)d ��*�28@�TH�]Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
M
ampere_sgemm_32x32_sliced1x4_tnV��*�28��@�JH�NbCudnnRNNhu  �A
�
�void elemWiseRNNcell<float, float, float, (cudnnRNNMode_t)2, (cudnnRNNBiasMode_t)2>(int, int, int, int, int, bool, float const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float) *�28��	@�/H�1bCudnnRNNhu  �B
�
�void LSTM_elementWise_bp1<float, float, float>(int, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, int, int, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float)*�28��@�-H�.Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@�FH�Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�(H�(bgradients/split_1_grad/concathu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align1>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align1::Params)` ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void GENERIC_elementWise_bp2<float, float, float, 4, (cudnnRNNBiasMode_t)2>(int, int, float*, float*, cudnn::reduced_divisor, float*) ��*  28��@�dH�qXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28��@�:H�;bsplit_2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�(H�)bgradients/split_grad/concathu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�(H�(bgradients/split_grad/concathu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�'H�'bgradients/split_1_grad/concathu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2@8�@�HH�Sbsplit_1hu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�#H�$bgradients/split_grad/concathu  �B
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�}@�}H�}b
concat_1_0hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�l@�6H�6b"gradients/transpose_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�l@�6H�6btranspose_0hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�l@�6H�6b$gradients/transpose_9_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�j@�5H�5btranspose_9hu  �B
�
�void GENERIC_elementWise_bp2<float, float, float, 4, (cudnnRNNBiasMode_t)2>(int, int, float*, float*, cudnn::reduced_divisor, float*) ��*  28�h@�hH�hXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` ��*�28�f@�fH�fXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�c@�cH�cXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�`@�.H�1btranspose_5hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�`@�.H�1b$gradients/transpose_6_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�_@�/H�/bgradients/AddNhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/btranspose_6hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/b$gradients/transpose_8_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/btranspose_7hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�.btranspose_8hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/b$gradients/transpose_7_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�\@�-H�.b$gradients/transpose_5_grad/transposehu  �B
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�Z@�ZH�Zb
concat_1_0hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�2`8�Z@�,H�-b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*�28�W@�+H�,b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4::Params)` ��*�2`8�T@�TH�TbCudnnRNNh
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�T@�TH�TXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)Z ��*�208�N@�NH�NbCudnnRNNhu  �A
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�M@�MH�Mb
concat_1_0hu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2@8�J@�JH�Jbsplithu  �B
]
ampere_sgemm_32x32_sliced1x4_nnV��*�28�I@�IH�IXbmodel_2/dense_10/MatMulhu  �A
C
ampere_sgemm_32x128_tn9��*�28�H@�HH�HbCudnnRNNhu  HB
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�G@�GH�Gb$Adam/Adam/update_1/ResourceApplyAdamhu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4::Params)\ ��*�28�G@�GH�GXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
j
ampere_sgemm_32x32_sliced1x4_ntV��*�28�F@�FH�Fb&gradient_tape/model_2/dense_9/MatMul_1hu  �A
�
�void tensorflow::functor::ColumnReduceSimpleKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>) *�28�E@�EH�Eb)gradient_tape/model_2/repeat_vector_2/Sumhu  �B
�
�void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*�28�E@�EH�Eb&gradient_tape/huber_loss/DynamicStitchhu  �B
�
�void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*�28�D@�DH�Db,gradient_tape/model_2/lambda_2/DynamicStitchhu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2 8�D@�DH�Dbsplithu  �B
k
ampere_sgemm_32x32_sliced1x4_tnV��*�28�C@�CH�CXb%gradient_tape/model_2/dense_10/MatMulhu  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)Z ��*�28�C@�CH�CXbmodel_2/dense_9/MatMulhu  �A
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�A@�AH�Ab"Adam/Adam/update/ResourceApplyAdamhu  �B
k
ampere_sgemm_32x32_sliced1x4_ntV��*�28�A@�AH�Ab'gradient_tape/model_2/dense_10/MatMul_1hu  �A
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28�A@�AH�Absplit_1hu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28�A@�AH�Absplithu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�@@�@H�@b$Adam/Adam/update_5/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�?@�?H�?b$Adam/Adam/update_6/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2`8�<@�<H�<b7model_2/dropout_10/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�;@�;H�;b%Adam/Adam/update_17/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2`8�;@�;H�;b7model_2/dropout_11/dropout/random_uniform/RandomUniformhu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x10_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x10_tn_align4::Params)` ��*�28�;@�;H�;Xb$gradient_tape/model_2/dense_9/MatMulh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2 8�;@�;H�;b%Adam/Adam/update_10/ResourceApplyAdamhu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *�28�:@�:H�:b2gradient_tape/model_2/dense_11/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�208�:@�:H�:b7model_2/dropout_12/dropout/random_uniform/RandomUniformhu  �B
�
�void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)/�*�2`8�9@�9H�9Xb model_2/dense_8/Tensordot/MatMulhu  zB
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�9@�9H�9b$Adam/Adam/update_2/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�9@�9H�9b%Adam/Adam/update_11/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�9@�9H�9b%Adam/Adam/update_26/ResourceApplyAdamhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�208�7@�7H�7b*gradient_tape/model_2/lambda_2/BroadcastTohu  �B
�
�void tensorflow::functor::ColumnReduceSimpleKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>) *�28�7@�7H�7bmodel_2/lambda_2/Sumhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�208�6@�6H�6bmodel_2/repeat_vector_2/Tilehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 6, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�6@�6H�6btranspose_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�5@�5H�5b$gradients/transpose_9_grad/transposehu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�5@�5H�5b$Adam/Adam/update_3/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!* 28�5@�5H�5b1gradient_tape/model_2/dense_8/BiasAdd/BiasAddGradhu  �B
�
�std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, true, false, 7, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)� �*28�5@�5H�5Xbmodel_2/dense_11/MatMulhu  �A
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�5@�5H�5btranspose_0hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�2@8�4@�4H�4b3gradient_tape/model_2/permute_2/transpose/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�4@�4H�4btranspose_9hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�4@�4H�4b$Adam/Adam/update_7/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�4@�4H�4b+model_2/batch_normalization_10/moments/meanhu  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2�8�3@�3H�3Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�3@�3H�3b%Adam/Adam/update_21/ResourceApplyAdamhu  �B
�
�void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool) *�28�3@�3H�3bmodel_2/activation_2/Softmaxhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�2@�2H�2b+model_2/batch_normalization_12/moments/meanhu  �B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�2@�2H�2b<gradient_tape/model_2/batch_normalization_10/moments/truedivhu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�2@�2H�2bAdam/gradients/AddN_10hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�2@�2H�2b%Adam/Adam/update_18/ResourceApplyAdamhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�2@�2H�2bBgradient_tape/model_2/batch_normalization_10/moments/BroadcastTo_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�2@�2H�2b/model_2/batch_normalization_10/moments/variancehu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�2@�2H�2bAdam/gradients/AddN_13hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�2@�2H�2bBgradient_tape/model_2/batch_normalization_11/moments/BroadcastTo_1hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�1@�1H�1b$Adam/Adam/update_4/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 6, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�1@�1H�1b$gradients/transpose_4_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�2@8�1@�1H�1bmodel_2/permute_2/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1btranspose_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�1@�1H�1b@gradient_tape/model_2/batch_normalization_10/moments/BroadcastTohu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�1@�1H�1b@gradient_tape/model_2/batch_normalization_11/moments/BroadcastTohu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�1@�1H�1b%Adam/Adam/update_12/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�1@�1H�1b%Adam/Adam/update_16/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1btranspose_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�1@�1H�1b0model_2/batch_normalization_12/AssignMovingAvg_1hu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�1@�1H�1bAdam/gradients/AddN_8hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�1@�1H�1b%Adam/Adam/update_22/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1b$gradients/transpose_1_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0b$gradients/transpose_5_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 6, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�0@�0H�0b$gradients/transpose_1_grad/transposehu  �B
�
�void splitKreduce_kernel<32, 16, int, float, float, float, float, true, false, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*) * 28�0@�0H�0b7gradient_tape/model_2/dense_8/Tensordot/MatMul/MatMul_1hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_13/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b$Adam/Adam/update_8/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_14/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_15/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_24/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b$Adam/Adam/update_9/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0b$gradients/transpose_1_grad/transposehu  �B
�
�void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)  �* 28�0@�0H�0b7gradient_tape/model_2/dense_8/Tensordot/MatMul/MatMul_1hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_19/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_20/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_23/ResourceApplyAdamhu  �B
T
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/bmodel_2/add_4/addhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�/@�/H�/bAdam/gradients/AddN_11hu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/b,gradient_tape/model_2/dropout_10/dropout/Mulhu  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2$8�/@�/H�/Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_7_grad/transposehu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/b model_2/dropout_11/dropout/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�208�/@�/H�/bBgradient_tape/model_2/batch_normalization_12/moments/BroadcastTo_1hu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�28�/@�/H�/b2gradient_tape/model_2/dense_10/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�/@�/H�/b%Adam/Adam/update_25/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 6, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�/@�/H�/btranspose_2hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 6, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�/@�/H�/btranspose_3hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/bBgradient_tape/model_2/batch_normalization_10/batchnorm/mul_1/Mul_1hu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/b model_2/dropout_10/dropout/Mul_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_4_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_6_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_3_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_4_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/btranspose_3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_2_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_3_grad/transposehu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b8model_2/batch_normalization_11/moments/SquaredDifferencehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b+model_2/batch_normalization_11/moments/meanhu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�28�.@�.H�.b1gradient_tape/model_2/dense_9/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b+model_2/batch_normalization_11/moments/meanhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b+model_2/batch_normalization_10/moments/meanhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b/model_2/batch_normalization_10/moments/variancehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 6, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�.@�.H�.b$gradients/transpose_2_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 128, 6, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�.@�.H�.b$gradients/transpose_3_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_8hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_2hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_4hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_2hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.bBgradient_tape/model_2/batch_normalization_10/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.bBgradient_tape/model_2/batch_normalization_11/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_5hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_6hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�.@�.H�.bAdam/gradients/AddN_6hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.bBgradient_tape/model_2/batch_normalization_12/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b+model_2/batch_normalization_12/moments/meanhu  �B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.b:gradient_tape/model_2/batch_normalization_11/moments/mul_1hu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.b,gradient_tape/model_2/dropout_11/dropout/Mulhu  �B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b8gradient_tape/model_2/batch_normalization_10/moments/subhu  �B
V
!Tanh_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.bmodel_2/dense_8/Tanhhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.bBgradient_tape/model_2/batch_normalization_11/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�28�.@�.H�.b7model_2/dropout_13/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_4hu  �B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b.model_2/batch_normalization_11/batchnorm/add_1hu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.b model_2/dropout_12/dropout/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b@gradient_tape/model_2/batch_normalization_10/batchnorm/mul_1/Mulhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b8model_2/batch_normalization_10/moments/SquaredDifferencehu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b/model_2/batch_normalization_11/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b/model_2/batch_normalization_12/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.bBgradient_tape/model_2/batch_normalization_10/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 128, 6, 128, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �*�28�.@�.H�.btranspose_4hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_7hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_2_grad/transposehu  �B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b.model_2/batch_normalization_10/batchnorm/add_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b.model_2/batch_normalization_10/batchnorm/mul_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b.model_2/batch_normalization_11/batchnorm/mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�-@�-H�-bgradients/AddNhu  �B
�
�void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)  �* 28�-@�-H�-b'gradient_tape/model_2/dense_11/MatMul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�-@�-H�-b/model_2/batch_normalization_11/moments/variancehu  �B
W
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-bmodel_2/multiply_2/mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�208�-@�-H�-b@gradient_tape/model_2/batch_normalization_12/moments/BroadcastTohu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-bBgradient_tape/model_2/batch_normalization_11/batchnorm/mul_1/Mul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�-@�-H�-bBgradient_tape/model_2/batch_normalization_12/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�28�-@�-H�-b7model_2/dropout_14/dropout/random_uniform/RandomUniformhu  �B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-b:gradient_tape/model_2/batch_normalization_10/moments/mul_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_8_grad/transposehu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�-@�-H�-bBgradient_tape/model_2/batch_normalization_10/batchnorm/add_1/Sum_1hu  �B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-b:gradient_tape/model_2/batch_normalization_12/moments/mul_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�-@�-H�-bBgradient_tape/model_2/batch_normalization_12/batchnorm/add_1/Sum_1hu  �B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b>gradient_tape/model_2/batch_normalization_11/moments/truediv_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b@gradient_tape/model_2/batch_normalization_11/batchnorm/mul_1/Mulhu  �B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b8gradient_tape/model_2/batch_normalization_11/moments/subhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�-@�-H�-bBgradient_tape/model_2/batch_normalization_11/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�-@�-H�-bBgradient_tape/model_2/batch_normalization_11/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�-@�-H�-b/model_2/batch_normalization_12/moments/variancehu  �B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b<gradient_tape/model_2/batch_normalization_11/moments/truedivhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,bBgradient_tape/model_2/batch_normalization_12/batchnorm/mul_1/Mul_1hu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b(gradient_tape/model_2/multiply_2/mul/Mulhu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b,gradient_tape/model_2/dropout_12/dropout/Mulhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�,@�,H�,bBgradient_tape/model_2/batch_normalization_10/batchnorm/mul_1/Sum_1hu  �B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b>gradient_tape/model_2/batch_normalization_10/moments/truediv_1hu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,bmodel_2/dropout_10/dropout/Mulhu  �B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b8gradient_tape/model_2/batch_normalization_10/moments/Mulhu  �B
k
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b*gradient_tape/model_2/multiply_2/mul/Mul_1hu  �B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b8gradient_tape/model_2/batch_normalization_11/moments/Mulhu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,bmodel_2/dropout_11/dropout/Mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�208�,@�,H�,b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�,@�,H�,bBgradient_tape/model_2/batch_normalization_12/batchnorm/mul_1/Sum_1hu  �B
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�+@�+H�+b'model_2/dropout_10/dropout/GreaterEqualhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b8model_2/batch_normalization_12/moments/SquaredDifferencehu  �B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b8gradient_tape/model_2/batch_normalization_12/moments/Mulhu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+bmodel_2/dropout_12/dropout/Mulhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b.gradient_tape/model_2/dropout_10/dropout/Mul_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b.gradient_tape/model_2/dropout_11/dropout/Mul_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b.model_2/batch_normalization_12/batchnorm/mul_1hu  �B
�
�void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28�+@�+H�+Xb5gradient_tape/model_2/dense_8/Tensordot/MatMul/MatMulhu  �B
a
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�+@�+H�+bmodel_2/dropout_10/dropout/Casthu  �B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b>gradient_tape/model_2/batch_normalization_12/moments/truediv_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b@gradient_tape/model_2/batch_normalization_12/batchnorm/mul_1/Mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*�28�+@�+H�+b-gradients/strided_slice_grad/StridedSliceGradhu  �B
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�+@�+H�+b'model_2/dropout_11/dropout/GreaterEqualhu  �B
�
�void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*�28�+@�+H�+bmodel_2/activation_2/Softmaxhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�*@�*H�*b.gradient_tape/model_2/dropout_12/dropout/Mul_1hu  �B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�*@�*H�*b.model_2/batch_normalization_12/batchnorm/add_1hu  �B
a
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�*@�*H�*bmodel_2/dropout_11/dropout/Casthu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�*@�*H�*bmodel_2/dense_8/BiasAddhu  �B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�*@�*H�*b<gradient_tape/model_2/batch_normalization_12/moments/truedivhu  �B
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�*@�*H�*b'model_2/dropout_12/dropout/GreaterEqualhu  �B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�*@�*H�*b8gradient_tape/model_2/batch_normalization_12/moments/subhu  �B
�
�void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28�*@�*H�*Xb%gradient_tape/model_2/dense_11/MatMulhu  �B
a
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�)@�)H�)bmodel_2/dropout_12/dropout/Casthu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�)@�)H�)b+model_2/batch_normalization_13/moments/meanhu  �B
�
�void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*�28�)@�)H�)b&gradient_tape/model_2/activation_2/Sumhu  �B
�
�void tensorflow::GatherOpKernel<tensorflow::AlignedVector<int, 1>, int, true>(tensorflow::AlignedVector<int, 1> const*, int const*, tensorflow::AlignedVector<int, 1>*, long, long, long, long)*�28�(@�(H�(b$model_2/dense_8/Tensordot/GatherV2_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b+model_2/batch_normalization_14/moments/meanhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�(@�(H�(bmodel_2/lstm_6/zeroshu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b/model_2/batch_normalization_13/moments/variancehu  �B
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type) *�28�(@�(H�(bmodel_2/activation_2/Softmaxhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b/model_2/batch_normalization_14/moments/variancehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�'@�'H�'bdiv_no_nan_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�'@�'H�'bmodel_2/lstm_7/zeroshu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'bBgradient_tape/model_2/batch_normalization_13/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'bBgradient_tape/model_2/batch_normalization_14/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'bBgradient_tape/model_2/batch_normalization_13/batchnorm/add_1/Sum_1hu  �B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b2model_2/batch_normalization_12/AssignMovingAvg/subhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�'@�'H�'bmodel_2/dense_10/BiasAddhu  �B
`
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�'@�'H�'bmodel_2/dropout_14/dropout/Casthu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b,model_2/batch_normalization_11/batchnorm/mulhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b8model_2/batch_normalization_14/moments/SquaredDifferencehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�'@�'H�'b-gradient_tape/huber_loss/weighted_loss/Tile_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�'@�'H�'bBgradient_tape/model_2/batch_normalization_14/moments/BroadcastTo_1hu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b8model_2/batch_normalization_13/moments/SquaredDifferencehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�'@�'H�'bAdam/gradients/zeros_like_6hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�'@�'H�'bhuber_loss/weighted_loss/valuehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�'@�'H�'bBgradient_tape/model_2/batch_normalization_13/moments/BroadcastTo_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b@gradient_tape/model_2/batch_normalization_13/batchnorm/mul_1/Mulhu  �B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b.model_2/batch_normalization_13/batchnorm/add_1hu  �B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�&@�&H�&bAdam/addhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b.model_2/batch_normalization_13/batchnorm/mul_1hu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b,model_2/batch_normalization_12/batchnorm/mulhu  �B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b2model_2/batch_normalization_11/AssignMovingAvg/subhu  �B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b2model_2/batch_normalization_13/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&b
div_no_nanhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&b7gradient_tape/huber_loss/weighted_loss/value/div_no_nanhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_7hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�&@�&H�&bBgradient_tape/model_2/batch_normalization_14/batchnorm/mul_1/Sum_1hu  �B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b8gradient_tape/model_2/batch_normalization_13/moments/Mulhu  �B
q
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b.model_2/batch_normalization_14/batchnorm/add_1hu  �B
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�&@�&H�&b'model_2/dropout_14/dropout/GreaterEqualhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b.model_2/batch_normalization_14/batchnorm/mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_4hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_8hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_5hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b@gradient_tape/model_2/batch_normalization_14/batchnorm/mul_1/Mulhu  �B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b8gradient_tape/model_2/batch_normalization_13/moments/subhu  �B
g
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b&gradient_tape/model_2/activation_2/subhu  �B
y
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b8gradient_tape/model_2/batch_normalization_14/moments/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�&@�&H�&bmodel_2/lstm_8/zeroshu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&bAdam/gradients/AddN_2hu  �B
�
�void tensorflow::GatherOpKernel<tensorflow::AlignedVector<int, 1>, int, true>(tensorflow::AlignedVector<int, 1> const*, int const*, tensorflow::AlignedVector<int, 1>*, long, long, long, long)*�28�&@�&H�&b"model_2/dense_8/Tensordot/GatherV2hu  �B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b,model_2/batch_normalization_11/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�&@�&H�&b@gradient_tape/model_2/batch_normalization_13/moments/BroadcastTohu  �B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&bMulhu  �B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b4model_2/batch_normalization_11/AssignMovingAvg_1/subhu  �B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b4model_2/batch_normalization_13/AssignMovingAvg_1/subhu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&bAdam/gradients/AddN_5hu  �B
T
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bmodel_2/add_5/addhu  �B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b>gradient_tape/model_2/batch_normalization_13/moments/truediv_1hu  �B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b>gradient_tape/model_2/batch_normalization_10/batchnorm/mul/Mulhu  �B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bAdam/Powhu  �B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b2model_2/batch_normalization_10/AssignMovingAvg/subhu  �B
s
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b2model_2/batch_normalization_14/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_4hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b.gradient_tape/model_2/dropout_13/dropout/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�%@�%H�%b@gradient_tape/model_2/batch_normalization_14/moments/BroadcastTohu  �B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b<gradient_tape/model_2/batch_normalization_14/moments/truedivhu  �B

 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b>gradient_tape/model_2/batch_normalization_14/moments/truediv_1hu  �B
y
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b8gradient_tape/model_2/batch_normalization_14/moments/Mulhu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,model_2/batch_normalization_13/batchnorm/mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bmodel_2/dense_10/Reluhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�%@�%H�%bmodel_2/dense_11/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�%@�%H�%bmodel_2/dense_9/BiasAddhu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bmodel_2/dropout_14/dropout/Mulhu  �B
`
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�%@�%H�%bmodel_2/dropout_13/dropout/Casthu  �B
p
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�%@�%H�%b'model_2/dropout_13/dropout/GreaterEqualhu  �B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�%@�%H�%b
LogicalAndhu  �B
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bhuber_loss/mulhu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bmodel_2/dropout_13/dropout/Mulhu  �B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b
Adam/Pow_1hu  �B
O
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bhuber_loss/Subhu  �B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b4model_2/batch_normalization_12/AssignMovingAvg_1/subhu  �B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,model_2/batch_normalization_12/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%b@gradient_tape/model_2/batch_normalization_12/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%b@gradient_tape/model_2/batch_normalization_13/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_tanh_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_tanh_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%b&gradient_tape/model_2/dense_8/TanhGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_3hu  �B
�
�void tensorflow::functor::BlockReduceKernel<int*, int*, 256, tensorflow::functor::Prod<int> >(int*, int*, int, tensorflow::functor::Prod<int>, std::iterator_traits<int*>::value_type)0*�28�%@�%H�%bmodel_2/dense_8/Tensordot/Prodhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAdam/Adam/AssignAddVariableOphu  �B
Y
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bhuber_loss/SelectV2hu  �B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,model_2/batch_normalization_12/batchnorm/addhu  �B
}
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b<gradient_tape/model_2/batch_normalization_13/moments/truedivhu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,gradient_tape/model_2/dropout_13/dropout/Mulhu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,gradient_tape/model_2/dropout_14/dropout/Mulhu  �B
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b&gradient_tape/model_2/activation_2/mulhu  �B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b:gradient_tape/model_2/batch_normalization_14/moments/mul_1hu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b model_2/dropout_13/dropout/Mul_1hu  �B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b,model_2/batch_normalization_14/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�%@�%H�%bAdam/gradients/zeros_likehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAdam/gradients/AddN_9hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_2hu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b model_2/dropout_14/dropout/Mul_1hu  �B
i
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b#gradient_tape/huber_loss/SelectV2_1hu  �B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b,model_2/batch_normalization_13/batchnorm/addhu  �B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b,model_2/batch_normalization_14/batchnorm/addhu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b gradient_tape/huber_loss/Abs/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bBgradient_tape/model_2/batch_normalization_14/batchnorm/mul_1/Mul_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b.model_2/batch_normalization_12/batchnorm/mul_2hu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b,model_2/batch_normalization_14/batchnorm/mulhu  �B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b4model_2/batch_normalization_14/AssignMovingAvg_1/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�$@�$H�$bAdam/gradients/zeros_like_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bmodel_2/dense_9/Reluhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddN_3hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAssignAddVariableOphu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�$@�$H�$bSum_2hu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*�28�$@�$H�$b/gradient_tape/model_2/repeat_vector_2/transposehu  �B
Z
%LessEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�$@�$H�$bhuber_loss/LessEqualhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_14/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b&gradient_tape/model_2/dense_9/ReluGradhu  �B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�$@�$H�$bCasthu  �B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bgradient_tape/huber_loss/Mulhu  �B
{
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b:gradient_tape/model_2/batch_normalization_13/moments/mul_1hu  �B
m
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b,model_2/batch_normalization_10/batchnorm/mulhu  �B
g
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b!gradient_tape/huber_loss/SelectV2hu  �B
c
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b!gradient_tape/huber_loss/Abs/Signhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�$@�$H�$bAdam/gradients/zeros_like_2hu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�$@�$H�$bhuber_loss/weighted_loss/Sumhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bBgradient_tape/model_2/batch_normalization_13/batchnorm/mul_1/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b.model_2/batch_normalization_14/AssignMovingAvghu  �B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�$@�$H�$bCast_2hu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�$@�$H�$b;gradient_tape/model_2/batch_normalization_14/moments/Cast_1hu  �B
i
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b(gradient_tape/model_2/activation_2/mul_1hu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b>gradient_tape/model_2/batch_normalization_14/batchnorm/sub/Neghu  �B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b.model_2/batch_normalization_12/batchnorm/Rsqrthu  �B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b,model_2/batch_normalization_13/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b'gradient_tape/model_2/dense_10/ReluGradhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b.gradient_tape/model_2/dropout_14/dropout/Mul_1hu  �B
Q
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bhuber_loss/sub_1hu  �B
O
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bhuber_loss/Abshu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�$@�$H�$b;gradient_tape/model_2/batch_normalization_10/moments/Cast_1hu  �B
a
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b gradient_tape/huber_loss/truedivhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_10/batchnorm/mul/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bBgradient_tape/model_2/batch_normalization_10/batchnorm/mul_2/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_11/batchnorm/mul/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_13/batchnorm/mul_2/Mulhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b.model_2/batch_normalization_10/batchnorm/mul_2hu  �B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b.model_2/batch_normalization_14/batchnorm/Rsqrthu  �B
U
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bhuber_loss/Squarehu  �B
u
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b4model_2/batch_normalization_10/AssignMovingAvg_1/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_10/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b@gradient_tape/model_2/batch_normalization_11/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddNhu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model_2/batch_normalization_11/moments/Cast_1hu  �B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b9gradient_tape/model_2/batch_normalization_12/moments/Casthu  �B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b>gradient_tape/model_2/batch_normalization_11/batchnorm/mul/Mulhu  �B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b,model_2/batch_normalization_11/batchnorm/addhu  �B
l
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b*huber_loss/weighted_loss/num_elements/Casthu  �B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b2model_2/batch_normalization_10/AssignMovingAvg/mulhu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b4model_2/batch_normalization_13/AssignMovingAvg_1/mulhu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b4model_2/batch_normalization_14/AssignMovingAvg_1/mulhu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bgradient_tape/huber_loss/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b@gradient_tape/model_2/batch_normalization_12/batchnorm/mul/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b@gradient_tape/model_2/batch_normalization_13/batchnorm/mul/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b@gradient_tape/model_2/batch_normalization_14/batchnorm/mul/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bBgradient_tape/model_2/batch_normalization_14/batchnorm/mul_2/Mul_1hu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b.model_2/batch_normalization_14/batchnorm/mul_2hu  �B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b.model_2/batch_normalization_13/batchnorm/Rsqrthu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b.model_2/batch_normalization_11/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b0model_2/batch_normalization_13/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b0model_2/batch_normalization_14/AssignMovingAvg_1hu  �B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b9gradient_tape/model_2/batch_normalization_14/moments/Casthu  �B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b2model_2/batch_normalization_14/AssignMovingAvg/mulhu  �B
_
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#bgradient_tape/huber_loss/Casthu  �B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b9gradient_tape/model_2/batch_normalization_10/moments/Casthu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model_2/batch_normalization_12/moments/Cast_1hu  �B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b2model_2/batch_normalization_11/AssignMovingAvg/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bBgradient_tape/model_2/batch_normalization_12/batchnorm/mul_2/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bBgradient_tape/model_2/batch_normalization_13/batchnorm/mul_2/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b0model_2/batch_normalization_11/AssignMovingAvg_1hu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b gradient_tape/huber_loss/mul/Mulhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b.model_2/batch_normalization_13/batchnorm/mul_2hu  �B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b.model_2/batch_normalization_10/batchnorm/Rsqrthu  �B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b2model_2/batch_normalization_13/AssignMovingAvg/mulhu  �B
o
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b,model_2/batch_normalization_10/batchnorm/addhu  �B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�#@�#H�#b9gradient_tape/model_2/batch_normalization_11/moments/Casthu  �B
�
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�#@�#H�#bKhuber_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  �B
s
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b2model_2/batch_normalization_12/AssignMovingAvg/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b@gradient_tape/model_2/batch_normalization_14/batchnorm/mul_2/Mulhu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b>gradient_tape/model_2/batch_normalization_13/batchnorm/sub/Neghu  �B
m
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b,model_2/batch_normalization_10/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b.model_2/batch_normalization_10/AssignMovingAvghu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b>gradient_tape/model_2/batch_normalization_10/batchnorm/sub/Neghu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b>gradient_tape/model_2/batch_normalization_11/batchnorm/sub/Neghu  �B
q
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b.model_2/batch_normalization_11/batchnorm/Rsqrthu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b.model_2/batch_normalization_12/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b.model_2/batch_normalization_13/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#bAdam/gradients/AddN_12hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"bBgradient_tape/model_2/batch_normalization_11/batchnorm/mul_2/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b@gradient_tape/model_2/batch_normalization_12/batchnorm/mul_2/Mulhu  �B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b>gradient_tape/model_2/batch_normalization_13/batchnorm/mul/Mulhu  �B
o
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b.model_2/batch_normalization_11/batchnorm/mul_2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"bAdam/gradients/AddN_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"bAdam/gradients/AddN_4hu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b4model_2/batch_normalization_11/AssignMovingAvg_1/mulhu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b4model_2/batch_normalization_12/AssignMovingAvg_1/mulhu  �B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b>gradient_tape/model_2/batch_normalization_14/batchnorm/mul/Mulhu  �B

 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b>gradient_tape/model_2/batch_normalization_12/batchnorm/sub/Neghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"b0model_2/batch_normalization_10/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"bAdam/gradients/AddN_7hu  �B
u
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b4model_2/batch_normalization_10/AssignMovingAvg_1/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b@gradient_tape/model_2/batch_normalization_10/batchnorm/mul_2/Mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b@gradient_tape/model_2/batch_normalization_11/batchnorm/mul_2/Mulhu  �B

 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b>gradient_tape/model_2/batch_normalization_12/batchnorm/mul/Mulhu  �B
}
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model_2/batch_normalization_13/moments/Cast_1hu  �B
{
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b9gradient_tape/model_2/batch_normalization_13/moments/Casthu  �B
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�!@�!H�!bAdam/Cast_1hu  �B