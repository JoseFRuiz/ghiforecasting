
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)Z ��*�28��@�FH�QbCudnnRNNh0u  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�@�CH�HXb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �A
�
�void elemWiseRNNcell<float, float, float, (cudnnRNNMode_t)2, (cudnnRNNBiasMode_t)2>(int, int, int, int, int, bool, float const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float) *�28��@�0H�6bCudnnRNNh0u  �B
�
�void LSTM_elementWise_bp1<float, float, float>(int, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, int, int, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float)*�28ݽ@�.H�5Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2 8��@�+H�/Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph0u  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nn_align4::Params)d ��*�28��@�TH�]Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
M
ampere_sgemm_32x32_sliced1x4_tnV��*�28��@�JH�QbCudnnRNNhu  �A
�
�void elemWiseRNNcell<float, float, float, (cudnnRNNMode_t)2, (cudnnRNNBiasMode_t)2>(int, int, int, int, int, bool, float const*, float const*, float const*, float const*, float const*, float const*, float const*, float*, float*, float*, float*, float*, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float) *�28��	@�/H�1bCudnnRNNhu  �B
�
�void LSTM_elementWise_bp1<float, float, float>(int, int, float*, float*, float*, float*, float*, float*, float*, float*, float*, int, int, cudnnRNNClipMode_t, cudnnNanPropagation_t, float, float)*�28��@�,H�0Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@�FH��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�'H�(bgradients/split_1_grad/concathu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align1>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align1::Params)` ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void GENERIC_elementWise_bp2<float, float, float, 4, (cudnnRNNBiasMode_t)2>(int, int, float*, float*, cudnn::reduced_divisor, float*) ��*  28��@�dH�nXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28�@�;H�<bsplit_2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�'H�+bgradients/split_grad/concathu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28�@�(H�(bgradients/split_grad/concathu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�'H�(bgradients/split_grad/concathu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorSlicingOp<Eigen::array<int, 2ul> const, Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, int)*�28��@�'H�(bgradients/split_1_grad/concathu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2@8��@�GH�Rbsplit_1hu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_nt_align4::Params)^ ��*�28��@��H��Xb(gradients/CudnnRNN_grad/CudnnRNNBackproph
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�@�H�b
concat_1_0hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�m@�6H�6b"gradients/transpose_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�l@�6H�6b$gradients/transpose_9_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�l@�6H�6btranspose_0hu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align1::Params)` ��*�28�k@�kH�kXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�j@�5H�5btranspose_9hu  �B
�
�void GENERIC_elementWise_bp2<float, float, float, 4, (cudnnRNNBiasMode_t)2>(int, int, float*, float*, cudnn::reduced_divisor, float*) ��*  28�h@�hH�hXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�b@�bH�bXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�_@�_H�_b
concat_1_0hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�_@�.H�0b$gradients/transpose_7_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�_@�.H�0btranspose_5hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�^@�/H�/bgradients/AddNhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/btranspose_6hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�.b$gradients/transpose_6_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�]@�.H�/btranspose_7hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�\@�.H�.b$gradients/transpose_8_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�\@�.H�.btranspose_8hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�\@�.H�.b$gradients/transpose_5_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�2`8�Z@�,H�.b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*�28�X@�+H�-b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_32x6_tn_align4::Params)` ��*�2`8�W@�WH�WbCudnnRNNh
C
ampere_sgemm_32x128_tn9��*�28�U@�UH�UbCudnnRNNhu  HB
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)^ ��*�28�T@�TH�TXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4::Params)Z ��*�208�N@�NH�NbCudnnRNNhu  �A
�
�void tensorflow::concat_variable_kernel<float, int, true>(tensorflow::GpuDeviceArrayStruct<float const*, 8>, tensorflow::GpuDeviceArrayStruct<int, 8>, int, int, float*) D*�2�8�M@�MH�Mb
concat_1_0hu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nt_align4::Params)\ ��*�28�I@�IH�IXb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �A
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2@8�H@�HH�Hbsplithu  �B
Z
ampere_sgemm_32x32_sliced1x4_nnV��*�28�G@�GH�GXbmodel/dense_2/MatMulhu  �A
h
ampere_sgemm_32x32_sliced1x4_ntV��*�28�F@�FH�Fb$gradient_tape/model/dense_1/MatMul_1hu  �A
�
�void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*�28�E@�EH�Eb&gradient_tape/huber_loss/DynamicStitchhu  �B
�
�void tensorflow::(anonymous namespace)::DynamicStitchKernel<int>(int, int, tensorflow::GpuDeviceArrayStruct<int, 8>, tensorflow::GpuDeviceArrayStruct<int const*, 8>, int*)*�28�E@�EH�Eb(gradient_tape/model/lambda/DynamicStitchhu  �B
�
�void tensorflow::functor::ColumnReduceSimpleKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>) *�28�E@�EH�Eb%gradient_tape/model/repeat_vector/Sumhu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�2 8�D@�DH�Dbsplithu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x6_nn_align4::Params)Z ��*�28�C@�CH�CXbmodel/dense_1/MatMulhu  �A
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�B@�BH�Bb$Adam/Adam/update_1/ResourceApplyAdamhu  �B
h
ampere_sgemm_32x32_sliced1x4_tnV��*�28�B@�BH�BXb"gradient_tape/model/dense_2/MatMulhu  �A
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28�B@�BH�Bbsplithu  �B
�
�void tensorflow::(anonymous namespace)::SplitOpKernel<float>(float const*, int, int, int, tensorflow::GpuDeviceArrayStruct<float*, 8>)*�28�A@�AH�Absplit_1hu  �B
h
ampere_sgemm_32x32_sliced1x4_ntV��*�28�A@�AH�Ab$gradient_tape/model/dense_2/MatMul_1hu  �A
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�@@�@H�@b$Adam/Adam/update_6/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2@8�?@�?H�?b$Adam/Adam/update_5/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�>@�>H�>b"Adam/Adam/update/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2`8�<@�<H�<b4model/dropout_1/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�2`8�;@�;H�;b2model/dropout/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�;@�;H�;b?gradient_tape/model/batch_normalization_1/batchnorm/mul_1/Sum_1hu  �B
�
�void cutlass::Kernel<cutlass_80_tensorop_s1688gemm_64x64_16x10_tn_align4>(cutlass_80_tensorop_s1688gemm_64x64_16x10_tn_align4::Params)` ��*�28�;@�;H�;Xb"gradient_tape/model/dense_1/MatMulh
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�:@�:H�:b%Adam/Adam/update_17/ResourceApplyAdamhu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) *�28�:@�:H�:b/gradient_tape/model/dense_3/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�208�:@�:H�:b4model/dropout_2/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�:@�:H�:b%Adam/Adam/update_11/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�2 8�9@�9H�9b%Adam/Adam/update_10/ResourceApplyAdamhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�8@�8H�8btranspose_0hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�8@�8H�8b%Adam/Adam/update_26/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ColumnReduceSimpleKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>) *�28�8@�8H�8bmodel/lambda/Sumhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�208�7@�7H�7bmodel/repeat_vector/Tilehu  �B
�
�void gemv2T_kernel_val<int, int, float, float, float, float, 128, 16, 2, 2, false, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>, float, float)/�*�2`8�7@�7H�7Xbmodel/dense/Tensordot/MatMulhu  zB
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�7@�7H�7btranspose_9hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�208�6@�6H�6b&gradient_tape/model/lambda/BroadcastTohu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<unsigned int, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorShufflingOp<Eigen::array<int, 3ul> const, Eigen::TensorMap<Eigen::Tensor<unsigned int const, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�5@�5H�5b$gradients/transpose_9_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�5@�5H�5btranspose_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�2@8�4@�4H�4b/gradient_tape/model/permute/transpose/transposehu  �B
�
�std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, float, float, float, float, false, true, true, false, 7, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)� �*28�4@�4H�4Xbmodel/dense_3/MatMulhu  �A
�
�void tensorflow::functor::ColumnReduceMax16ColumnsKernel<float const*, float*, cub::Sum>(float const*, float*, int, int, cub::Sum, std::iterator_traits<float const*>::value_type)�!* 28�4@�4H�4b-gradient_tape/model/dense/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�4@�4H�4b$Adam/Adam/update_7/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�3@�3H�3b$Adam/Adam/update_2/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�3@�3H�3b%Adam/Adam/update_21/ResourceApplyAdamhu  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2�8�2@�2H�2Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�2@�2H�2bAdam/gradients/AddN_13hu  �B
�
�void splitKreduce_kernel<float, float, float, float>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*) *�2�8�2@�2H�2Xb(gradients/CudnnRNN_grad/CudnnRNNBackprophu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�2@�2H�2b$Adam/Adam/update_4/ResourceApplyAdamhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�2@�2H�2b=gradient_tape/model/batch_normalization/moments/BroadcastTo_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�2@�2H�2b?gradient_tape/model/batch_normalization_1/moments/BroadcastTo_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�1@�1H�1b=gradient_tape/model/batch_normalization_1/moments/BroadcastTohu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1b$gradients/transpose_1_grad/transposehu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�1@�1H�1bAdam/gradients/AddN_10hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�2`8�1@�1H�1b;gradient_tape/model/batch_normalization/moments/BroadcastTohu  �B
�
�void tensorflow::(anonymous namespace)::GenerateNormalizedProb<float, float, 4>(float const*, float const*, float const*, float*, int, int, bool) *�28�1@�1H�1bmodel/activation/Softmaxhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�1@�1H�1bAdam/gradients/zeros_like_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1btranspose_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�2@8�1@�1H�1bmodel/permute/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1btranspose_2hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�1@�1H�1b%Adam/Adam/update_22/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�1@�1H�1btranspose_1hu  �B
�
�void splitKreduce_kernel<32, 16, int, float, float, float, float, true, false, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*) * 28�0@�0H�0b3gradient_tape/model/dense/Tensordot/MatMul/MatMul_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0btranspose_3hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b$Adam/Adam/update_8/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0btranspose_4hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_12/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_16/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_18/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0b$gradients/transpose_1_grad/transposehu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b$Adam/Adam/update_3/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_23/ResourceApplyAdamhu  �B
h
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�0@�0H�0b'gradient_tape/model/dropout/dropout/Mulhu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�0@�0H�0bAdam/gradients/AddN_8hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_14/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_15/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_24/ResourceApplyAdamhu  �B
P
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�0@�0H�0bmodel/add/addhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_13/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b%Adam/Adam/update_25/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�0@�0H�0b$Adam/Adam/update_9/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�0@�0H�0b$gradients/transpose_1_grad/transposehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�208�/@�/H�/b?gradient_tape/model/batch_normalization_2/moments/BroadcastTo_1hu  �B
t
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�/@�/H�/b3gradient_tape/model/batch_normalization/moments/subhu  �B
�
�void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)  �* 28�/@�/H�/b3gradient_tape/model/dense/Tensordot/MatMul/MatMul_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/btranspose_5hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_3_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/btranspose_3hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�/@�/H�/b=gradient_tape/model/batch_normalization/batchnorm/mul_1/Mul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�/@�/H�/b?gradient_tape/model/batch_normalization_1/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�/@�/H�/b&model/batch_normalization/moments/meanhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_2_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/b$gradients/transpose_4_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�/@�/H�/btranspose_2hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b(model/batch_normalization_1/moments/meanhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_7_grad/transposehu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b3model/batch_normalization/moments/SquaredDifferencehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_4hu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.bmodel/dropout/dropout/Mul_1hu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.bmodel/dropout_1/dropout/Mul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b=gradient_tape/model/batch_normalization/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_3hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_4hu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�.@�.H�.b%Adam/Adam/update_19/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b&model/batch_normalization/moments/meanhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.b)gradient_tape/model/dropout_1/dropout/Mulhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b)model/batch_normalization/batchnorm/mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�.@�.H�.bAdam/gradients/AddN_6hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�2`8�.@�.H�.bAdam/gradients/AddN_11hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�208�.@�.H�.b=gradient_tape/model/batch_normalization_2/moments/BroadcastTohu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�28�.@�.H�.b/gradient_tape/model/dense_1/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::ApplyAdamKernel<float>(int, float*, float*, float*, float const*, float const*, float const*, float const*, float const*, float const*, float const*, bool) *�28�.@�.H�.b%Adam/Adam/update_20/ResourceApplyAdamhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b=gradient_tape/model/batch_normalization/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b,model/batch_normalization_1/moments/variancehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_5_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_4_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_3_grad/transposehu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b(model/batch_normalization_1/moments/meanhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b5model/batch_normalization_1/moments/SquaredDifferencehu  �B
�
Rvoid tensorflow::BiasGradNHWC_SharedAtomics<float>(int, float const*, float*, int) �*�28�.@�.H�.b/gradient_tape/model/dense_2/BiasAdd/BiasAddGradhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b,model/batch_normalization_1/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b(model/batch_normalization_2/moments/meanhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�28�.@�.H�.b4model/dropout_3/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::FillPhiloxRandomKernelLaunch<tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float> >(unsigned long const*, unsigned long const*, tensorflow::random::PhiloxRandom, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>::ResultElementType*, long, tensorflow::random::UniformDistribution<tensorflow::random::PhiloxRandom, float>)*�28�.@�.H�.b4model/dropout_4/dropout/random_uniform/RandomUniformhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_6hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_8hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_7hu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�.@�.H�.bmodel/dropout_2/dropout/Mul_1hu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�.@�.H�.b;gradient_tape/model/batch_normalization/batchnorm/mul_1/Mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�208�.@�.H�.bgradients/AddNhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�.@�.H�.b(model/batch_normalization_2/moments/meanhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�.@�.H�.b*model/batch_normalization/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�.@�.H�.b*model/batch_normalization/moments/variancehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.b$gradients/transpose_3_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�.@�.H�.btranspose_2hu  �B
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b)model/batch_normalization/batchnorm/add_1hu  �B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b+model/batch_normalization_1/batchnorm/add_1hu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b+model/batch_normalization_1/batchnorm/mul_1hu  �B
R
!Tanh_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-bmodel/dense/Tanhhu  �B
�
�void tensorflow::functor::CleanupSegments<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�-@�-H�-b,model/batch_normalization_2/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�-@�-H�-b?gradient_tape/model/batch_normalization_2/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�-@�-H�-b,model/batch_normalization_2/moments/variancehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_6_grad/transposehu  �B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-b7gradient_tape/model/batch_normalization_1/moments/mul_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�-@�-H�-b?gradient_tape/model/batch_normalization_2/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_8_grad/transposehu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_2_grad/transposehu  �B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-b7gradient_tape/model/batch_normalization_2/moments/mul_1hu  �B
S
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-bmodel/multiply/mulhu  �B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�-@�-H�-b5gradient_tape/model/batch_normalization/moments/mul_1hu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_4_grad/transposehu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�-@�-H�-b=gradient_tape/model/batch_normalization_1/batchnorm/mul_1/Mulhu  �B
�
�void tensorflow::functor::SwapDimension1And2InTensor3UsingTiles<unsigned int, 256, 32, 32, false>(unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*) �!*�28�-@�-H�-b$gradients/transpose_2_grad/transposehu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�-@�-H�-b=gradient_tape/model/batch_normalization/batchnorm/add_1/Sum_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b?gradient_tape/model/batch_normalization_1/batchnorm/mul_1/Mul_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�,@�,H�,b?gradient_tape/model/batch_normalization_1/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�,@�,H�,b?gradient_tape/model/batch_normalization_2/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�,@�,H�,b=gradient_tape/model/batch_normalization/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2@8�,@�,H�,b?gradient_tape/model/batch_normalization_1/batchnorm/mul_1/Sum_1hu  �B
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b9gradient_tape/model/batch_normalization/moments/truediv_1hu  �B
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b;gradient_tape/model/batch_normalization_1/moments/truediv_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b?gradient_tape/model/batch_normalization_2/batchnorm/mul_1/Mul_1hu  �B
e
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b$gradient_tape/model/multiply/mul/Mulhu  �B
g
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b&gradient_tape/model/multiply/mul/Mul_1hu  �B
t
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b3gradient_tape/model/batch_normalization/moments/Mulhu  �B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b5gradient_tape/model/batch_normalization_1/moments/subhu  �B
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�,@�,H�,b9gradient_tape/model/batch_normalization_1/moments/truedivhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b)gradient_tape/model/dropout_2/dropout/Mulhu  �B
Z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,bmodel/dropout/dropout/Mulhu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,bmodel/dropout_1/dropout/Mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long) *�208�,@�,H�,b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>)  �* 28�,@�,H�,b$gradient_tape/model/dense_3/MatMul_1hu  �B
�
�void tensorflow::functor::CleanupSegments<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*  2 8�,@�,H�,b?gradient_tape/model/batch_normalization_2/batchnorm/mul_1/Sum_1hu  �B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�,@�,H�,b5gradient_tape/model/batch_normalization_1/moments/Mulhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�,@�,H�,b5model/batch_normalization_2/moments/SquaredDifferencehu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b)gradient_tape/model/dropout/dropout/Mul_1hu  �B
x
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�2`8�+@�+H�+b7gradient_tape/model/batch_normalization/moments/truedivhu  �B
k
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�+@�+H�+b"model/dropout/dropout/GreaterEqualhu  �B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b5gradient_tape/model/batch_normalization_2/moments/Mulhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b+gradient_tape/model/dropout_1/dropout/Mul_1hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b=gradient_tape/model/batch_normalization_2/batchnorm/mul_1/Mulhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b+model/batch_normalization_2/batchnorm/mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorStridingSlicingOp<Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::DSizes<int, 3> const, Eigen::TensorMap<Eigen::Tensor<float, 3, 1, int>, 16, Eigen::MakePointer> >, Eigen::TensorMap<Eigen::Tensor<float const, 3, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::GpuDevice>, long)*�28�+@�+H�+b-gradients/strided_slice_grad/StridedSliceGradhu  �B
�
�void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28�+@�+H�+Xb1gradient_tape/model/dense/Tensordot/MatMul/MatMulhu  �B
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b9gradient_tape/model/batch_normalization_2/moments/truedivhu  �B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�+@�+H�+b+model/batch_normalization_2/batchnorm/add_1hu  �B
m
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�+@�+H�+b$model/dropout_1/dropout/GreaterEqualhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+b+gradient_tape/model/dropout_2/dropout/Mul_1hu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�+@�+H�+bmodel/dropout_2/dropout/Mulhu  �B
�
�void gemmk1_kernel<int, float, 256, 5, true, false, false, false, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, 0>(cublasGemmk1Params<float, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float, biasType<cublasGemvTensorStridedBatched<float>::value_type, float>::type>)�*�28�*@�*H�*Xb"gradient_tape/model/dense_3/MatMulhu  �B
m
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�*@�*H�*b$model/dropout_2/dropout/GreaterEqualhu  �B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�*@�*H�*b5gradient_tape/model/batch_normalization_2/moments/subhu  �B
^
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�*@�*H�*bmodel/dropout_2/dropout/Casthu  �B
\
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�*@�*H�*bmodel/dropout/dropout/Casthu  �B
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�208�*@�*H�*b;gradient_tape/model/batch_normalization_2/moments/truediv_1hu  �B
�
�void tensorflow::functor::RowReduceKernel<float const*, float*, cub::Max>(float const*, float*, int, int, cub::Max, std::iterator_traits<float const*>::value_type) *�28�*@�*H�*bmodel/activation/Softmaxhu  �B
^
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�2�8�*@�*H�*bmodel/dropout_1/dropout/Casthu  �B
�
�void tensorflow::functor::RowReduceKernel<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, cub::Sum>(cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long>, float*, int, int, cub::Sum, std::iterator_traits<cub::TransformInputIterator<float, tensorflow::(anonymous namespace)::SubtractAndExpFunctor<float, float>, cub::CountingInputIterator<int, long>, long> >::value_type)*�28�*@�*H�*bmodel/activation/Softmaxhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�)@�)H�)bmodel/lstm/zeroshu  �B
�
�void tensorflow::GatherOpKernel<tensorflow::AlignedVector<int, 1>, int, true>(tensorflow::AlignedVector<int, 1> const*, int const*, tensorflow::AlignedVector<int, 1>*, long, long, long, long)*�28�)@�)H�)b model/dense/Tensordot/GatherV2_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�)@�)H�)b(model/batch_normalization_3/moments/meanhu  �B
I
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�(@�(H�(bAdam/Powhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b,model/batch_normalization_3/moments/variancehu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b(model/batch_normalization_4/moments/meanhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�(@�(H�(bmodel/dense/BiasAddhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�(@�(H�(bmodel/lstm_1/zeroshu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�(@�(H�(bAdam/gradients/zeros_like_8hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�(@�(H�(b?gradient_tape/model/batch_normalization_4/moments/BroadcastTo_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b?gradient_tape/model/batch_normalization_3/batchnorm/add_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�(@�(H�(b?gradient_tape/model/batch_normalization_4/batchnorm/add_1/Sum_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�(@�(H�(bdiv_no_nan_1hu  �B
�
�void tensorflow::functor::RowReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)*�28�(@�(H�(b"gradient_tape/model/activation/Sumhu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'b?gradient_tape/model/batch_normalization_3/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, float*, tensorflow::functor::Sum<float> >(float*, float*, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'b?gradient_tape/model/batch_normalization_4/batchnorm/mul_1/Sum_1hu  �B
�
�void tensorflow::functor::ColumnReduceKernel<float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, tensorflow::functor::Sum<float> >(float*, tensorflow::TransformOutputIterator<float, float, tensorflow::functor::DividesBy<float, float>, long>, int, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)�!*  28�'@�'H�'b,model/batch_normalization_4/moments/variancehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�'@�'H�'bAdam/Adam/AssignAddVariableOphu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�'@�'H�'bAdam/gradients/zeros_like_3hu  �B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�'@�'H�'bmodel/dropout_3/dropout/Casthu  �B
Z
%LessEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�'@�'H�'bhuber_loss/LessEqualhu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�'@�'H�'bAdam/gradients/AddN_2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�'@�'H�'b?gradient_tape/model/batch_normalization_3/moments/BroadcastTo_1hu  �B
K
"AddV2_GPU_DT_INT64_DT_INT64_kernel*�28�'@�'H�'bAdam/addhu  �B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b5gradient_tape/model/batch_normalization_3/moments/Mulhu  �B
z
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b9gradient_tape/model/batch_normalization/batchnorm/mul/Mulhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b5model/batch_normalization_4/moments/SquaredDifferencehu  �B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�'@�'H�'b/model/batch_normalization_1/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�'@�'H�'b
div_no_nanhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�'@�'H�'bAdam/gradients/zeros_like_7hu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�&@�&H�&bmodel/dense_1/BiasAddhu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�&@�&H�&bmodel/dense_2/BiasAddhu  �B
v
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b5gradient_tape/model/batch_normalization_4/moments/Mulhu  �B
�
.SquaredDifference_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b5model/batch_normalization_3/moments/SquaredDifferencehu  �B
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b1model/batch_normalization_1/AssignMovingAvg_1/subhu  �B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b/model/batch_normalization_4/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&b7gradient_tape/huber_loss/weighted_loss/value/div_no_nanhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&bAssignAddVariableOp_2hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b=gradient_tape/model/batch_normalization_3/batchnorm/mul_1/Mulhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b+model/batch_normalization_3/batchnorm/mul_1hu  �B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b5gradient_tape/model/batch_normalization_3/moments/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 1ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�&@�&H�&b-gradient_tape/huber_loss/weighted_loss/Tile_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_4hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_6hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&bAssignAddVariableOphu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b=gradient_tape/model/batch_normalization_4/batchnorm/mul_1/Mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�&@�&H�&bAdam/gradients/zeros_like_5hu  �B
]
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&bgradient_tape/huber_loss/Mulhu  �B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b+model/batch_normalization_3/batchnorm/add_1hu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b+model/batch_normalization_4/batchnorm/mul_1hu  �B
O
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&bhuber_loss/mulhu  �B
v
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�&@�&H�&b5gradient_tape/model/batch_normalization_4/moments/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_const_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�&@�&H�&bmodel/lstm_2/zeroshu  �B
]
 Cast_GPU_DT_BOOL_DT_FLOAT_kernel*�28�&@�&H�&bmodel/dropout_4/dropout/Casthu  �B
�	
�	void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float, float>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�&@�&H�&bAdam/gradients/AddN_5hu  �B
�
�void tensorflow::GatherOpKernel<tensorflow::AlignedVector<int, 1>, int, true>(tensorflow::AlignedVector<int, 1> const*, int const*, tensorflow::AlignedVector<int, 1>*, long, long, long, long)*�28�&@�&H�&bmodel/dense/Tensordot/GatherV2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<long const, long const>, Eigen::TensorMap<Eigen::Tensor<long, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<long const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_4hu  �B
R
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bmodel/add_1/addhu  �B
n
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b+model/batch_normalization_4/batchnorm/add_1hu  �B
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b;gradient_tape/model/batch_normalization_3/moments/truediv_1hu  �B
P
%LogicalAnd_GPU_DT_BOOL_DT_BOOL_kernel*�28�%@�%H�%b
LogicalAndhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bAssignAddVariableOp_3hu  �B
�
�void tensorflow::functor::ShuffleInTensor3Simple<unsigned int, 0, 2, 1, false>(int, unsigned int const*, tensorflow::functor::Dimension<3>, unsigned int*)*�28�%@�%H�%b+gradient_tape/model/repeat_vector/transposehu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b+gradient_tape/model/dropout_3/dropout/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�%@�%H�%b=gradient_tape/model/batch_normalization_4/moments/BroadcastTohu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b)model/batch_normalization_3/batchnorm/mulhu  �B
K
 Pow_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b
Adam/Pow_1hu  �B
Q
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bhuber_loss/sub_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bmodel/dense_2/Reluhu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%bmodel/dropout_4/dropout/Mulhu  �B
|
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b;gradient_tape/model/batch_normalization_4/moments/truediv_1hu  �B
m
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�%@�%H�%b$model/dropout_3/dropout/GreaterEqualhu  �B
c
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b"gradient_tape/model/activation/mulhu  �B
z
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b9gradient_tape/model/batch_normalization/batchnorm/sub/Neghu  �B
c
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b"gradient_tape/model/activation/subhu  �B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b/model/batch_normalization_3/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::div_no_nan_op<float, false>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bhuber_loss/weighted_loss/valuehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_max_op<float const, float const, 1>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%bmodel/dense_1/Reluhu  �B
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b9gradient_tape/model/batch_normalization_4/moments/truedivhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�%@�%H�%bAdam/gradients/zeros_likehu  �B
m
(GreaterEqual_GPU_DT_FLOAT_DT_BOOL_kernel*�28�%@�%H�%b$model/dropout_4/dropout/GreaterEqualhu  �B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b/model/batch_normalization_3/AssignMovingAvg/mulhu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b;gradient_tape/model/batch_normalization/batchnorm/mul/Mul_1hu  �B
h
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�%@�%H�%b'model/batch_normalization/batchnorm/mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�%@�%H�%b=gradient_tape/model/batch_normalization_1/batchnorm/RsqrtGradhu  �B
�
�void tensorflow::functor::BlockReduceKernel<int*, int*, 256, tensorflow::functor::Prod<int> >(int*, int*, int, tensorflow::functor::Prod<int>, std::iterator_traits<int*>::value_type)0*�28�%@�%H�%bmodel/dense/Tensordot/Prodhu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b gradient_tape/huber_loss/mul/Mulhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)gradient_tape/model/dropout_3/dropout/Mulhu  �B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b/model/batch_normalization_2/AssignMovingAvg/mulhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b+gradient_tape/model/dropout_4/dropout/Mul_1hu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_2/batchnorm/mulhu  �B
i
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b#gradient_tape/huber_loss/SelectV2_1hu  �B
n
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b-model/batch_normalization/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int) *�28�$@�$H�$bAdam/gradients/zeros_like_2hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b$gradient_tape/model/dense_2/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddN_12hu  �B
j
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b'model/batch_normalization/batchnorm/addhu  �B
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_1/batchnorm/addhu  �B
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_2/batchnorm/addhu  �B
z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�$@�$H�$b8gradient_tape/model/batch_normalization_2/moments/Cast_1hu  �B
z
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b9gradient_tape/model/batch_normalization_3/moments/truedivhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)gradient_tape/model/dropout_4/dropout/Mulhu  �B
n
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b-model/batch_normalization/AssignMovingAvg/mulhu  �B
a
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b gradient_tape/huber_loss/Abs/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b?gradient_tape/model/batch_normalization_3/batchnorm/mul_1/Mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b?gradient_tape/model/batch_normalization_4/batchnorm/mul_1/Mul_1hu  �B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b7gradient_tape/model/batch_normalization_4/moments/mul_1hu  �B
c
!Sign_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b!gradient_tape/huber_loss/Abs/Signhu  �B
O
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bhuber_loss/Subhu  �B
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_1/batchnorm/subhu  �B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b/model/batch_normalization_2/AssignMovingAvg/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_tanh_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_tanh_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b"gradient_tape/model/dense/TanhGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_product_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorConversionOp<float, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_cmp_op<float const, float const, (Eigen::internal::ComparisonName)5>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorCwiseNullaryOp<Eigen::internal::scalar_constant_op<float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b$gradient_tape/model/dense_1/ReluGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddN_3hu  �B
�
Tvoid tensorflow::BiasNHWCKernel<float>(int, float const*, float const*, float*, int)*�28�$@�$H�$bmodel/dense_3/BiasAddhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_1/batchnorm/mulhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 2, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorBroadcastingOp<Eigen::array<int, 2ul> const, Eigen::TensorMap<Eigen::Tensor<float const, 2, 1, int>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, int)*�28�$@�$H�$b=gradient_tape/model/batch_normalization_3/moments/BroadcastTohu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�$@�$H�$bSum_2hu  �B
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_3/batchnorm/addhu  �B
l
"AddV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_4/batchnorm/addhu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bmodel/dropout_3/dropout/Mul_1hu  �B
p
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b/model/batch_normalization/AssignMovingAvg_1/subhu  �B
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b1model/batch_normalization_4/AssignMovingAvg_1/subhu  �B
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_4/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b=gradient_tape/model/batch_normalization_4/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddN_9hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$bAdam/gradients/AddNhu  �B
�
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�$@�$H�$bKhuber_loss/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Casthu  �B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b/model/batch_normalization_1/AssignMovingAvg/mulhu  �B
\
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$bmodel/dropout_3/dropout/Mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b?gradient_tape/model/batch_normalization_1/batchnorm/mul_2/Mul_1hu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_4/batchnorm/mulhu  �B
g
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b!gradient_tape/huber_loss/SelectV2hu  �B
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�$@�$H�$b)model/batch_normalization_3/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�$@�$H�$b;gradient_tape/model/batch_normalization/batchnorm/RsqrtGradhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b+model/batch_normalization_1/batchnorm/mul_2hu  �B
^
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bmodel/dropout_4/dropout/Mul_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b=gradient_tape/model/batch_normalization_3/batchnorm/RsqrtGradhu  �B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_1/AssignMovingAvg_1/mulhu  �B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_2/AssignMovingAvg_1/mulhu  �B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_4/AssignMovingAvg_1/mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization/batchnorm/mul_2/Mul_1hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_1/batchnorm/mul_2/Mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_3/batchnorm/mul_2/Mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_4/batchnorm/mul_2/Mulhu  �B
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model/batch_normalization_3/batchnorm/sub/Neghu  �B
Y
%SelectV2_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bhuber_loss/SelectV2hu  �B
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_2/AssignMovingAvg_1/subhu  �B
j
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b)model/batch_normalization_2/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, int>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_rsqrt_gradient_op<float>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, int>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b=gradient_tape/model/batch_normalization_2/batchnorm/RsqrtGradhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b+model/batch_normalization_1/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b-model/batch_normalization_1/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#bAdam/gradients/AddN_7hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b?gradient_tape/model/batch_normalization_2/batchnorm/mul_2/Mul_1hu  �B
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model/batch_normalization_1/batchnorm/sub/Neghu  �B
a
 Div_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b gradient_tape/huber_loss/truedivhu  �B
e
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b$gradient_tape/model/activation/mul_1hu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model/batch_normalization_1/batchnorm/mul/Mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_3/batchnorm/mul/Mul_1hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_4/batchnorm/mul/Mul_1hu  �B
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b;gradient_tape/model/batch_normalization_4/batchnorm/sub/Neghu  �B
r
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_3/AssignMovingAvg_1/subhu  �B
D
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#bMulhu  �B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b/model/batch_normalization/AssignMovingAvg_1/mulhu  �B
p
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b/model/batch_normalization_4/AssignMovingAvg/mulhu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b?gradient_tape/model/batch_normalization_3/batchnorm/mul_2/Mul_1hu  �B
x
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b7gradient_tape/model/batch_normalization_3/moments/mul_1hu  �B
�
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b?gradient_tape/model/batch_normalization_4/batchnorm/mul_2/Mul_1hu  �B
�
�void tensorflow::functor::BlockReduceKernel<float*, float*, 256, tensorflow::functor::Sum<float> >(float*, float*, int, tensorflow::functor::Sum<float>, std::iterator_traits<float*>::value_type)0*�28�#@�#H�#bhuber_loss/weighted_loss/Sumhu  �B
r
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b1model/batch_normalization_3/AssignMovingAvg_1/mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b=gradient_tape/model/batch_normalization_2/batchnorm/mul_2/Mulhu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b+model/batch_normalization_3/batchnorm/mul_2hu  �B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b+model/batch_normalization_1/batchnorm/Rsqrthu  �B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b+model/batch_normalization_3/batchnorm/Rsqrthu  �B
h
 Sub_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�#@�#H�#b'model/batch_normalization/batchnorm/subhu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b)model/batch_normalization/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b+model/batch_normalization_2/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b-model/batch_normalization_2/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�#@�#H�#b-model/batch_normalization_3/AssignMovingAvg_1hu  �B
_
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"bgradient_tape/huber_loss/Mul_1hu  �B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b+model/batch_normalization_4/batchnorm/Rsqrthu  �B
O
 Abs_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"bhuber_loss/Abshu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model/batch_normalization_4/batchnorm/mul/Mulhu  �B
l
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b)model/batch_normalization/batchnorm/Rsqrthu  �B
U
#Square_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"bhuber_loss/Squarehu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"b+model/batch_normalization/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"bAdam/gradients/AddN_1hu  �B
F
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"bCasthu  �B
M
!Cast_GPU_DT_INT64_DT_FLOAT_kernel*�28�"@�"H�"bAdam/Cast_1hu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model/batch_normalization/batchnorm/mul_2/Mulhu  �B
j
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b)model/batch_normalization/batchnorm/mul_2hu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b+model/batch_normalization_2/batchnorm/mul_2hu  �B
l
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b+model/batch_normalization_4/batchnorm/mul_2hu  �B
n
"Rsqrt_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b+model/batch_normalization_2/batchnorm/Rsqrthu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"b-model/batch_normalization_4/AssignMovingAvg_1hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_sum_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"bAdam/gradients/AddN_4hu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"b+model/batch_normalization_3/AssignMovingAvghu  �B
�
�void Eigen::internal::EigenMetaKernel<Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long>(Eigen::TensorEvaluator<Eigen::TensorAssignOp<Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer>, Eigen::TensorCwiseBinaryOp<Eigen::internal::scalar_difference_op<float const, float const>, Eigen::TensorMap<Eigen::Tensor<float, 1, 1, long>, 16, Eigen::MakePointer> const, Eigen::TensorMap<Eigen::Tensor<float const, 1, 1, long>, 16, Eigen::MakePointer> const> const> const, Eigen::GpuDevice>, long)*�28�"@�"H�"b+model/batch_normalization_4/AssignMovingAvghu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model/batch_normalization_2/batchnorm/mul/Mulhu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b=gradient_tape/model/batch_normalization_2/batchnorm/mul/Mul_1hu  �B
|
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model/batch_normalization_3/batchnorm/mul/Mulhu  �B
|
 Neg_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b;gradient_tape/model/batch_normalization_2/batchnorm/sub/Neghu  �B
x
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"b6gradient_tape/model/batch_normalization/moments/Cast_1hu  �B
H
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"bCast_2hu  �B
x
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"b6gradient_tape/model/batch_normalization_3/moments/Casthu  �B
z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�"@�"H�"b8gradient_tape/model/batch_normalization_4/moments/Cast_1hu  �B
~
 Mul_GPU_DT_FLOAT_DT_FLOAT_kernel*�28�"@�"H�"b=gradient_tape/model/batch_normalization_1/batchnorm/mul/Mul_1hu  �B
_
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!bgradient_tape/huber_loss/Casthu  �B
z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b8gradient_tape/model/batch_normalization_1/moments/Cast_1hu  �B
x
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b6gradient_tape/model/batch_normalization_2/moments/Casthu  �B
v
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b4gradient_tape/model/batch_normalization/moments/Casthu  �B
x
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b6gradient_tape/model/batch_normalization_1/moments/Casthu  �B
z
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b8gradient_tape/model/batch_normalization_3/moments/Cast_1hu  �B
x
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b6gradient_tape/model/batch_normalization_4/moments/Casthu  �B
l
!Cast_GPU_DT_INT32_DT_FLOAT_kernel*�28�!@�!H�!b*huber_loss/weighted_loss/num_elements/Casthu  �B