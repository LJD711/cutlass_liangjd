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
    \brief Template for a pipelined Implicit GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad, Deconv)
  typename ConvProblemSize_ = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone    ///! Group mode
>
struct ImplicitGemmConvolution {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check iterator A and B convolution dimension are the same and 
  // set device::ImplicitGemmConvolution::kConvDim
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim, 
    "Convolution on different different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static conv::GroupMode const kGroupMode = GroupMode_;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  //
  //
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout, 
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A;
    TensorRefB ref_B;
    TensorRefC ref_C;
    TensorRefC ref_D;
    typename EpilogueOutputOp::Params output_op;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A,
      TensorRefB const & ref_B,
      TensorRefC const & ref_C,
      TensorRefC const & ref_D,
      typename EpilogueOutputOp::Params const & output_op,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A(ref_A),
      ref_B(ref_B),
      ref_C(ref_C),
      ref_D(ref_D),
      output_op(output_op),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    int swizzle_log_tile;

    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A;
    typename Mma::IteratorA::Element const *ptr_A;
    typename Mma::IteratorB::Params iterator_B;
    typename Mma::IteratorB::Element const *ptr_B;
    typename Epilogue::OutputTileIterator::Params iterator_C;
    typename Epilogue::OutputTileIterator::Element *ptr_C;
    typename Epilogue::OutputTileIterator::Params iterator_D;
    typename Epilogue::OutputTileIterator::Element *ptr_D;
    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations(0) { }

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A(Mma::IteratorA::getParams(args.problem_size, args.ref_A.layout())),
      ptr_A(args.ref_A.data()),
      iterator_B(args.problem_size, args.ref_B.layout()),
      ptr_B(args.ref_B.data()),
      iterator_C(ConvOutputIteratorParameter::layout(args.ref_C), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      ptr_C(args.ref_C.data()),
      iterator_D(ConvOutputIteratorParameter::layout(args.ref_D), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      ptr_D(args.ref_D.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(//计算逻辑CTA网格
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices); //(25, 1, 1) 注意这不是cuda 真正启动的gridDim，而是逻辑上的CTA网格，和 threadblock_swizzle 的实现有关。

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitGemmConvolution() { } 

  /// Executes one ImplicitGEMM
  //
  // 这个 operator() 是每个 CUDA CTA 执行 Implicit GEMM convolution 的入口。
  // host 侧的调用链大致是：
  //
  //   device::ImplicitGemmConvolution::initialize()
  //     -> 根据用户 Arguments 构造 Params
  //
  //   device::ImplicitGemmConvolution::run()
  //     -> Kernel<ImplicitGemmConvolution><<<grid, block, smem_size>>>(params)
  //
  //   cutlass::Kernel()
  //     -> 从 extern __shared__ 得到 SharedStorage
  //     -> op(params, shared_storage)
  //
  // operator() 的两个形参含义如下。
  //
  // 1. Params const &params
  //
  //   一个 CTA 只读的 kernel 运行参数。Params 最初在 host 侧由 Arguments
  //   构造，然后作为 CUDA kernel 参数传到 device。这里用 const reference
  //   避免在 operator() 内再次复制整个 Params。
  //
  //   Params 中最重要的字段包括：
  //
  //   problem_size:
  //     原始 convolution 参数，例如 N/H/W/C、K、R/S、padding、stride、
  //     dilation、groups 和 split_k_slices。
  //
  //   implicit_gemm_problem_size:
  //     convolution 映射后的逻辑 GEMM 尺寸。以 fprop 为例：
  //
  //       GEMM M = N * P * Q
  //       GEMM N = K
  //       GEMM K = R * S * C
  //
  //     A 是 activation 的 NPQ x RSC 逻辑 im2col 视图，B 是 RSC x K
  //     filter，输出 C/D 是 NPQ x K。
  //
  //   grid_tiled_shape:
  //     逻辑 GEMM 被 ThreadblockShape 切分后的 CTA 网格，坐标为
  //     (tile_m, tile_n, tile_k)。其中 tile_k 也用于 split-K slice。
  //
  //   swizzle_log_tile:
  //     ThreadblockSwizzle 使用的 swizzle 参数，用于把 CUDA blockIdx
  //     映射为逻辑 GEMM tile 坐标 threadblock_tile_idx。
  //
  //   gemm_k_iterations:
  //     mainloop 沿 GEMM K/RSC 方向要执行的 threadblock K tile 数量。
  //
  //   gemm_k_iterations_per_channel:
  //     fprop/dgrad 通常为 R*S，表示一个 channel 相关的 filter-position
  //     iteration 数。它作为统一 Mma 接口参数传入；当前
  //     ImplicitGemmMultistage 实现未直接使用该值，r/s/c 的实际推进由
  //     A/B convolution iterator 的 advance() 完成。
  //
  //   iterator_A / iterator_B:
  //     A/B global-memory iterator 的预计算参数，例如 tensor layout、
  //     快速除法器和指针增量。它们不保存实际 tensor 数据。
  //
  //   ptr_A / ptr_B:
  //     activation/filter 的 global-memory 起始地址。
  //
  //   iterator_C / iterator_D:
  //     epilogue output iterator 的布局、stride 和 extent 参数。
  //
  //   ptr_C / ptr_D:
  //     C 是 epilogue 的 source tensor，D 是最终 destination tensor。
  //     常见输出计算为 D = alpha * accumulator + beta * C。
  //
  //   output_op:
  //     epilogue 运算参数，例如 alpha、beta，以及可能的类型转换/激活配置。
  //
  //   semaphore:
  //     serial split-K 使用的 workspace 锁数组。没有 split-K 时不会参与同步。
  //
  //   split_k_mode:
  //     serial/parallel split-K 的执行与输出归约方式。
  //
  // 2. SharedStorage &shared_storage
  //
  //   当前 CTA 独占的 dynamic shared memory。它不是 host 参数，也不是每个
  //   thread 各自一份；同一 CTA 的所有线程引用同一块 shared memory。
  //   cutlass::Kernel() 通过 extern __shared__ 获得起始地址，再转换为
  //   Operator::SharedStorage* 后传入这里。
  //
  //   SharedStorage 是 union：
  //
  //     main_loop: Mma pipeline 保存 A/B tile 的多 stage shared-memory buffer
  //     epilogue:  epilogue 重排 accumulator 和协同写回所需的 shared memory
  //
  //   使用 union 的原因是 mainloop 完成后才执行 epilogue，两阶段生命周期
  //   不重叠，因此可以复用同一块 shared memory；总大小取两者较大值，而
  //   不是二者之和。
  //
  // 数值示例，仅用于理解：
  //
  //   convolution: N=1, P=Q=56, C=64, K=128, R=S=3, groups=1
  //   ThreadblockShape = GemmShape<128, 128, 64>
  //   split_k_slices = 1
  //
  // 则逻辑 GEMM 为 M=3136, N=128, K=576，普通 identity swizzle 下：
  //
  //   grid_tiled_shape.m = ceil(3136 / 128) = 25
  //   grid_tiled_shape.n = ceil(128  / 128) = 1
  //   grid_tiled_shape.k = 1
  //   gemm_k_iterations  = 576 / 64 = 9
  //
  // 每个 CTA 负责一个 128 x 128 输出 tile，并循环 9 次搬运/计算
  // 128 x 64 的 A tile 和 64 x 128 的 B tile。
  CUTLASS_DEVICE
  void operator()(
    Params const &params,          ///< CTA 只读的 convolution/kernel 运行参数
    SharedStorage &shared_storage  ///< 当前 CTA 独占并由所有线程共享的 dynamic smem
  ) {

    // Compute threadblock location
    // ThreadblockSwizzle 把物理 CUDA blockIdx 映射到逻辑 GEMM tile 坐标。
    // threadblock_tile_idx 的三个分量含义是：
    //   m(): 当前 CTA 负责第几个 M/NPQ tile
    //   n(): 当前 CTA 负责第几个 N/output-channel tile
    //   k(): 当前 CTA 负责第几个 split-K slice
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    // 某些 swizzle 会把 grid shape 向上取整，边缘可能产生不对应有效
    // M/N tile 的 CTA，因此这里先做边界检查。
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    // thread_idx 是 CTA 内线程编号，范围为 [0, kThreadCount)。
    int thread_idx = threadIdx.x;

    // A iterator 的 column 对应 implicit GEMM K/RSC 方向。
    // split-K 时，第 k 个 slice 从 k * ThreadblockShape::kK 开始。
    // 非 split-K 时 threadblock_tile_idx.k() 通常为 0。
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;

    // Group convolution 需要把逻辑 K/RSC offset 进一步移动到当前 group
    // 对应的 input-channel 区间。普通非 grouped convolution 不进入此分支。
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode != GroupMode::kDepthwise) {
        int k_per_group = params.problem_size.K / params.problem_size.groups;
        int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
        int channels_per_group = params.problem_size.C / params.problem_size.groups;
        iterator_A_column_offset += group_idx * channels_per_group;
      } else {
        iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    } 

    // Construct iterators to A and B operands
    //
    // 对 fprop：
    //   iterator_A 遍历 activation 的 NPQ x RSC 逻辑矩阵；
    //   iterator_B 遍历 filter 的 RSC x K 逻辑矩阵。
    //
    // A 的 threadblock offset:
    //   row    = tile_m * Shape::kM，定位到当前 NPQ tile
    //   column = split-K/group 修正后的 RSC 起点
    typename Mma::IteratorA iterator_A(
      params.iterator_A,
      params.problem_size,
      params.ptr_A,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,//逻辑M block上的偏移
        iterator_A_column_offset//0 或者 split-K/group 修正后的 RSC 起点
      )
    );
    
    // B 的 threadblock offset:
    //   row    = tile_k * Shape::kK，定位到当前 RSC/split-K 起点
    //   column = tile_n * Shape::kN，定位到当前 output-channel tile
    typename Mma::IteratorB iterator_B(
      params.iterator_B,
      params.problem_size,
      params.ptr_B,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    // warp_idx 是 CTA 内 warp 编号，lane_idx 是 warp 内 lane 编号。
    // 例如 128-thread CTA：warp_idx=0..3，lane_idx=0..31。
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    // Mma 构造时会用 thread/warp/lane id 建立：
    //   global -> shared 的 A/B store iterator
    //   shared -> register 的 warp tile iterator
    // 并把 mainloop buffer 绑定到 shared_storage.main_loop。
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    // 每个线程持有自己负责的 accumulator fragment，位于寄存器中。
    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute threadblock-scoped matrix multiply-add
    // mainloop 重复 gemm_k_iterations 次：
    //   1. 从 global memory 读取 A/B tile
    //   2. cp.async 写入 shared_storage.main_loop
    //   3. warp 从 shared memory 取 fragment
    //   4. Tensor Core MMA 累加到 accumulators
    mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators, params.gemm_k_iterations_per_channel);

    //
    // Epilogue
    //

    // 根据 host 传入的 output_op 参数构造 device epilogue operation。
    EpilogueOutputOp output_op(params.output_op);

    // Construct the semaphore.
    // 同一个输出 (tile_m,tile_n) 的所有 split-K CTA 共用一个 semaphore。
    // block_idx 将二维输出 tile 坐标展平，故不包含 tile_k。
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    
    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    // 输出矩阵 C/D 的 tile 起点。对 fprop，row 对应 NPQ，column 对应 K。
    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D(
      params.iterator_D,
      params.ptr_D,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C(
      params.iterator_C,
      params.ptr_C,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Construct the epilogue
    // mainloop 已经结束，因此 union 中原来的 main_loop 存储可以由
    // shared_storage.epilogue 复用。
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C = iterator_D;
      }

      semaphore.wait(threadblock_tile_idx.k());

    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run efficient epilogue
    // 将各线程寄存器中的 accumulator fragment 协同重排并写入 D：
    //
    //   D = output_op(accumulators, C)
    //
    // 常见 LinearCombination 情况即 D = alpha * accumulator + beta * C。
    epilogue(output_op, iterator_D, accumulators, iterator_C);
  
    //
    // Release the semaphore
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_idx.k() + 1;
      }
      
      semaphore.release(lock);
    }
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
