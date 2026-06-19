/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief 
    Default kernel-level implicit GEMM convolution definitions combine threadblock-scoped 
      matrix multiply-add with the appropriate threadblock-scoped epilogue.  
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/conv/kernel/default_conv2d.h"

#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_fixed_channels.h"
#include "cutlass/conv/threadblock/conv2d_fprop_activation_tile_access_iterator_few_channels.h"

#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_analytic.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_optimized.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_fixed_channels.h"
#include "cutlass/conv/threadblock/conv2d_fprop_filter_tile_access_iterator_few_channels.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename OperatorClass,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized,
  conv::StrideSupport StrideSupport = StrideSupport::kUnity,
  /// Access granularity of A matrix in units of elements
  int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
  /// Access granularity of B matrix in units of elements
  int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value
> struct DefaultConv2dFprop;

/////////////////////////////////////////////////////////////////////////////////////////////////
//                         OpClassTensorOp convolutions 
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage 
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport, 
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  //
  // ThreadMapA 直接来自 GEMM MmaCore，也就是
  // DefaultMmaCore::IteratorThreadMapA。这里没有为 conv2d 重新设计线程分工，
  // 而是复用 GEMM A tile 的搬运方式。
  //
  // implicit GEMM fprop 会把 convolution 写成：
  //
  //   A = activation 的 im2col 逻辑视图，shape = NPQ x RSC
  //   B = filter，                         shape = RSC x K
  //   C = output，                         shape = NPQ x K
  //
  // 因此 ThreadblockShape::kM 对应一块 NPQ 输出位置，ThreadblockShape::kK
  // 对应一块 RSC(filter_r, filter_s, channel) 归约维。MmaCore 认为 A 是
  // row-major M x K；Conv2dFpropActivationTileAccessIteratorOptimized 则把
  // 同一个 A(m,k) 坐标翻译回 activation tensor 的 (n,h,w,c) 地址。
  //
  // 例如 ThreadblockShape = <128,128,64> 且 ElementA=half 时，
  // MmaCore::IteratorThreadMapA 会让 128 个线程覆盖一个 128x64 的 A tile。
  // 对 conv2d 来说，这就是 128 个 NPQ 位置 x 64 个 RSC 元素。后面的
  // IteratorA 使用 ThreadMapA 决定每个线程访问哪些 NPQ 行和哪些 C 通道向量。
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB,
      ThreadMapB,
      AccessTypeB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages 
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kFixedChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFixedChannels<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFixedChannels<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB,
      ThreadMapB,
      AccessTypeB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,//GemmShape<128, 128, 64>
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages//3
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and two stage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kFixedChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFixedChannels<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFixedChannels<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kFewChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFewChannels<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFewChannels<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB,
      ThreadMapB,
      AccessTypeB
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage
/// pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kFewChannels,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorFewChannels<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =

    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorFewChannels<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;

  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and multistage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, layout::TensorNCxHWx<InterleavedK>,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Global,
    MmaPolicy,
    Stages 
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm
/// and 2 stage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA,
        AccessTypeA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm and 2 stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      2, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  // Note GEMM shared memory threadmap is used here because conv global memory
  // layout needs to be mapped to fprop which is similar to the crosswise
  // layout which is used by the interleaved GEMM shared memory threadmap.
  // The Interleaved GEMM global memory layout is similar to the congruous
  // layout.
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and 
/// multistage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {// liangjd

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
    ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
    Stages, MathOperatorTag
  >;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA,
      AccessTypeA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand 
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB,
      AccessTypeB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * AlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    CacheOpB,
    MmaPolicy,
    Stages 
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and 
// multistage pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
    ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
    ElementB, layout::RowMajorInterleaved<InterleavedK>, ElementAccumulator, LayoutC, arch::OpClassTensorOp,
    Stages, MathOperatorTag, true
  >;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      layout::TensorNCxHWx<InterleavedK>,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand 
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      layout::TensorCxRSKx<InterleavedK>,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Global,
    MmaPolicy,
    Stages 
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm
/// and 2 stage pipeline.
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::AlignedArray<ElementA, AlignmentA>;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA,
        AccessTypeA 
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::AlignedArray<ElementB, AlignmentB>;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB,
        AccessTypeB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename detail::DefaultConvEpilogue<
    ArchTag,
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm and 2 stage 
/// pipeline with interleaved layout.
template <
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB,
  int InterleavedK
>
struct DefaultConv2dFprop <
  ElementA,
  layout::TensorNCxHWx<InterleavedK>,
  ElementB,
  layout::TensorCxRSKx<InterleavedK>,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassTensorOp,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
      ElementB, layout::RowMajorInterleaved<InterleavedK>, 
      ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      2, MathOperatorTag, true>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::SmemThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, layout::TensorNCxHWx<InterleavedK>,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::SmemThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, layout::TensorCxRSKx<InterleavedK>,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaTensorOp = typename MmaCore::MmaTensorOp;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultInterleavedConvEpilogue<
    ThreadblockShape,
    WarpMmaTensorOp,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    InterleavedK
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//                            OpClassSimt convolutions
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm, 
/// multi-stage pipeline, and FFMA-based mainloop for SM80

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, LayoutA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, LayoutB,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Always,
    MmaPolicy,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm, 
/// multi-stage pipeline, and FFMA-based mainloop for SM80

template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  int Stages,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  Stages,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      Stages, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA,
      LayoutA,
      ThreadMapA
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB,
      LayoutB,
      ThreadMapB
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmMultistage<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    arch::CacheOperation::Always,
    IteratorB,
    SmemIteratorB,
    arch::CacheOperation::Always,
    MmaPolicy,
    Stages 
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Analytic IteratorAlgorithm, 
/// 2 stage pipeline, and FFMA-based mainloop for SM50
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kAnalytic,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA, LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorAnalytic<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB, LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines a kernel for Conv2dFprop specialization for Optimized IteratorAlgorithm, 
/// 2 stage pipeline, and FFMA-based mainloop for SM50
template <
  typename ElementA,
  typename LayoutA,
  typename ElementB,
  typename LayoutB,
  typename ElementC,
  typename LayoutC,
  typename ElementAccumulator,
  typename ArchTag,
  typename ThreadblockShape,
  typename WarpShape,
  typename InstructionShape,
  typename EpilogueOutputOp,
  typename ThreadblockSwizzle,
  typename MathOperatorTag,
  conv::StrideSupport StrideSupport,
  int AlignmentA,
  int AlignmentB
>
struct DefaultConv2dFprop <
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  arch::OpClassSimt,
  ArchTag,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  EpilogueOutputOp,
  ThreadblockSwizzle,
  2,
  MathOperatorTag,
  IteratorAlgorithm::kOptimized,
  StrideSupport,
  AlignmentA,
  AlignmentB
> {

  // Define the core components from GEMM
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, layout::RowMajor,
      ElementB, layout::ColumnMajor, ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
      2, MathOperatorTag>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using IteratorA =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropActivationTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
        ElementA,
        LayoutA,
        ThreadMapA
      >
    >;

  using SmemIteratorA = typename MmaCore::SmemIteratorA;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using IteratorB =
    cutlass::conv::threadblock::TileIterator<
      cutlass::conv::threadblock::Conv2dFpropFilterTileAccessIteratorOptimized<
        cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
        ElementB,
        LayoutB,
        ThreadMapB
      >
    >;
  
  using SmemIteratorB = typename MmaCore::SmemIteratorB;

  // Warp-level GEMM components
  using WarpMmaSimtOp = typename MmaCore::MmaWarpSimt;
  using MmaPolicy = typename MmaCore::MmaPolicy;

  // Define the Mma
  using Mma = threadblock::ImplicitGemmPipelined<
    ThreadblockShape,
    IteratorA,
    SmemIteratorA,
    IteratorB,
    SmemIteratorB,
    ElementC,
    LayoutC,
    MmaPolicy
  >;

  // Define the epilogue
  using Epilogue = typename epilogue::threadblock::DefaultEpilogueSimt<
    ThreadblockShape,
    WarpMmaSimtOp,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    false,
    layout::NoPermute,
    StrideSupport,
    4
  >::Epilogue;

  // Define the kernel
  using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
    Mma,
    Epilogue,
    ThreadblockSwizzle,
    conv::Operator::kFprop
  >;

};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
