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
    \brief Templates implementing how threads are mapped to a given tile.

*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/coord.h"
#include "cutlass/predicate_vector.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_view.h"
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {

////////////////////////////////////////////////////////////////////////////////

/// Strip-mines a pitch-linear tile among a given number of threads, first along
/// the contiguous dimension then along the strided dimension.
///
/// The tile must be divisible by the thread count such that all threads may
/// execute the same number of iterations with the same delta to exhaustively
/// cover the tile.
///
/// This class satisfies the "RegularThreadMapping" concept.
///
/// This ThreadMap is used by SIMT kernels and operand E of the sparse tensor
/// kernels.
template <
  typename Shape_,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearStripminedThreadMap {
  
  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal implementation details
  struct Detail {

    static_assert(!(Shape::kContiguous % kElementsPerAccess), "");

    /// Shape of the tile in units of vectors
    using ShapeVec = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert((Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) ||
                      (!(kThreads % ShapeVec::kContiguous)),
                  "Shape must be divisible by number of iterations of each thread.");
  };

  /// Number of iterations by each thread
  using Iterations = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<
          1,
          // Redo the comparison here to work around divide by zero compiler
          // error.  The compiler evaluates both path of platform::conditional.
          (Threads >= Detail::ShapeVec::kContiguous
               ? (Detail::ShapeVec::kStrided + (kThreads / Detail::ShapeVec::kContiguous - 1)) /
                     (kThreads / Detail::ShapeVec::kContiguous)
               : 0)>,
      layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                               Detail::ShapeVec::kStrided>>::type;
  

  /// Interval between accesses along each dimension of the tensor's logical coordinate space
  /// (in units of Elements)
  using Delta = typename platform::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    layout::PitchLinearShape<
      1,
      kThreads / Detail::ShapeVec::kContiguous
    >,
    layout::PitchLinearShape<
      kThreads * kElementsPerAccess,
      1
    >
  >::type;

  /// Shape of the tile in units of vectors
  using StorageShape = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<Shape::kContiguous,
                               Iterations::kStrided*(kThreads / Detail::ShapeVec::kContiguous)>,
      layout::PitchLinearShape<Shape::kContiguous, Shape::kStrided>>::type;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  /// (in units of Elements)
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {
    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * kElementsPerAccess, 
      thread_id / Detail::ShapeVec::kContiguous);
  }
};

/// This ThreadMap is used by GEMV
template <
  typename Shape,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearTilePolicyStripminedThreadContiguous
{
 static_assert((Shape::kContiguous % (Threads * ElementsPerAccess)) == 0,
              "Contiguous shape must divide number of threads");

  using TensorCoord = layout::PitchLinearCoord;

  static int const kThreads = Threads;
  static int const kElementsPerAccess = ElementsPerAccess;

  using Iterations = layout::PitchLinearShape<
                      Shape::kContiguous / (kThreads * kElementsPerAccess),
                      Shape::kStrided>;

  using Delta = layout::PitchLinearShape<1, 1>;

  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id)
  {
    return TensorCoord(thread_id * Iterations::kContiguous * kElementsPerAccess, 0);
  }
};

template <
  typename Shape,
  int Threads,
  int ElementsPerAccess = 1
>
struct PitchLinearTilePolicyStripminedThreadStrided
{
  static_assert((Shape::kStrided % Threads == 0),
                "Strided shape must divide number of threads");

  using TensorCoord = layout::PitchLinearCoord;

  static int const kThreads = Threads;
  static int const kElementsPerAccess = ElementsPerAccess;

  using Iterations = layout::PitchLinearShape<
                      Shape::kContiguous / kElementsPerAccess,
                      Shape::kStrided / kThreads>;

  using Delta = layout::PitchLinearShape<1, 1>;

  using ShapeVec = Shape;

  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id)
  {

    return TensorCoord(0, thread_id * Iterations::kStrided);
  }
};


////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements.
///
/// This ThreadMap is used by tensor core kernels.
//
// 下面以 default_mma_core_sm80.h 中 A operand 的典型实例化为例说明：
//
//   Shape_                 = layout::PitchLinearShape<Shape::kK, Shape::kM>
//                          = layout::PitchLinearShape<64, 128>
//   Threads                = kThreads = 128
//   WarpThreadArrangement_ = layout::PitchLinearShape<8, 4>
//   ElementsPerAccess      = 128 bits / sizeof_bits<half>::value = 8
//
// pitch-linear 坐标有两个维度：
//
//   contiguous: 连续内存方向。对 row-major GEMM A 来说就是 K 方向。
//   strided:    跨行方向。对 row-major GEMM A 来说就是 M 方向。
//
// 所以这个实例描述的是：128 个线程如何覆盖一个 A tile，逻辑尺寸为
// contiguous=K=64、strided=M=128。每个线程一次搬 8 个 half，也就是一个
// 128b 向量。
template <
  // Shape_ 是整个线程块要覆盖的 pitch-linear tile 形状。
  // 当前例子：Shape_ = PitchLinearShape<64, 128>，即 contiguous/K=64，
  // strided/M=128，总元素数为 64 * 128 = 8192 个 half。
  typename Shape_,

  // Threads 是参与搬运这个 tile 的 CTA 线程数。
  // 当前例子：Threads = 128，对应 4 个 warp。
  int Threads,

  // WarpThreadArrangement_ 描述一个 warp 内 32 个 lane 如何排成二维阵列。
  // 当前例子：PitchLinearShape<8, 4>，表示一个 warp 中：
  //   8 个 lane 沿 contiguous/K 方向排开；
  //   4 组 lane 沿 strided/M 方向排开；
  //   8 * 4 = 32 lane，刚好一个 warp。
  typename WarpThreadArrangement_,

  // ElementsPerAccess 是单次向量化访问包含的元素个数。
  // 当前例子：128b / 16b(half) = 8，所以一个 lane 一次读写 8 个 half。
  int ElementsPerAccess = 1
>
struct PitchLinearWarpRakedThreadMap {

  /// Tensor coordinate
  // pitch-linear 坐标类型，包含 contiguous 和 strided 两个分量。
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  // 当前例子：Shape::kContiguous = 64，Shape::kStrided = 128。
  using Shape = Shape_;

  /// Number of threads total
  // 当前例子：kThreads = 128。
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  // 当前例子：kElementsPerAccess = 8。
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  // 一个线程的一次访问覆盖 contiguous 方向的 8 个元素、strided 方向的 1 行。
  // 当前例子：ThreadAccessShape = PitchLinearShape<8, 1>。
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    // 当前例子：WarpThreadArrangement = PitchLinearShape<8, 4>。
    using WarpThreadArrangement = WarpThreadArrangement_;// warp 内 32 个 lane 的排布。

    /// Number of threads per warp
    // 当前例子：kWarpSize = 8 * 4 = 32。
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    // 当前例子：kWarpCount = 128 / 32 = 4。
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    // 把 contiguous/K 方向从“元素数”换算成“向量访问次数”。
    // 当前例子：K=64，每次访问 8 个 half，所以 contiguous access 数是 64/8=8。
    // strided/M 方向不做向量化，仍然是 128 行。
    // 因此 ShapeInAccesses = PitchLinearShape<8, 128>。
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    static_assert(
      !(ShapeInAccesses::kContiguous % WarpThreadArrangement::kContiguous),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    static_assert(
      !(ShapeInAccesses::kStrided % WarpThreadArrangement::kStrided),
      "ShapeInAccesses must be divisible by WarpThreadArrangement.");

    // compute number of warp-level accesses total
    // 一个 warp 的 lane 排布是 <8,4>，所以：
    //   contiguous 方向需要 8 / 8 = 1 轮 warp-level access；
    //   strided/M 方向需要 128 / 4 = 32 轮 warp-level access。
    // 当前例子：WarpAccessIterations = PitchLinearShape<1, 32>。
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    // 优先把多个 warp 分给 strided/M 方向。当前例子有 4 个 warp，strided 方向
    // 有 32 轮可分，所以 4 个 warp 全部分到 strided 方向。
    // 当前例子：kWarpsStrided = min(32, 4) = 4。
    static int const kWarpsStrided =
        (WarpAccessIterations::kStrided >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kStrided);

    // 如果 warp 数量多到 strided 方向放不下，才继续分给 contiguous 方向。
    // 当前例子：kWarpCount=4 <= WarpAccessIterations::kStrided=32，
    // 所以 kWarpsContiguous = 1。
    static int const kWarpsContiguous =
        (kWarpCount > WarpAccessIterations::kStrided
             ? kWarpCount / kWarpsStrided
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    // 当前例子：WarpArrangement = PitchLinearShape<1, 4>。
    // 含义是 4 个 warp 不沿 K/contiguous 分裂，而是沿 M/strided 排开。
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  // 每个 warp 自己还要循环多少次，才能覆盖分配给它的区域。
  // 当前例子：
  //   contiguous iterations = 1 / 1 = 1
  //   strided iterations    = 32 / 4 = 8
  // 因此 Iterations = PitchLinearShape<1, 8>。
  // 注意：contiguous 方向只有 1 轮，所以当前例子每个线程的 K 向量位置固定；
  // strided 方向有 8 轮，所以每个线程会访问 8 个不同的 M/NPQ 行。
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;//一共32轮，有4个warp，所以每个warp 8轮。contiguous 方向只有1轮，strided 方向8轮。

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta between accesses (units of elements, concept: PitchLinearShape)
  // 相邻 iteration 之间的坐标步长，单位是元素，不是向量。
  // 当前例子：
  //   Delta::kContiguous = 8 lane * 8 half/lane = 64
  //   Delta::kStrided    = 4 rows
  // 因此 Delta = PitchLinearShape<64, 4>。
  // 当前 Iterations::kContiguous=1，所以 contiguous delta 实际不会被重复使用；
  // strided loop 会让同一线程访问 M, M+4, M+8, ... 这些行。
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided
  >;//描述的是一个warp内的线程访问 pattern。每次迭代 contiguous 方向跳 64 个元素，strided 方向跳 4 行。

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {//liangjd 输入线程ID，输出这个线程的初始访问坐标，单位是元素。

    // 当前例子：thread_id 0..127。
    //   warp_id = thread_id / 32，取值 0..3。
    //   lane_id = thread_id % 32，取值 0..31。
    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    // warp_footprint 是一个 warp 负责区域的尺寸，单位是“向量访问”。
    // 当前例子：
    //   contiguous = 8 lane * 1 iteration = 8 vectors
    //   strided    = 4 rows * 8 iterations = 32 rows
    // 所以 warp_footprint = (8, 32)。换成元素就是 K=8*8=64，M=32。
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };//warp的访问范围是 contiguous 方向 8 个向量，strided 方向 32 行。

    // This is the offset of a specific warp (in units of vectors)
    // 当前例子 kWarpsContiguous=1，所以所有 warp 在 contiguous/K 方向 offset 都是 0；
    // 4 个 warp 沿 strided/M 方向排布：warp_offset = (0, warp_id)。
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };//描述的是不同 warp 之间的偏移。当前例子 warp_offset=(0,0),(0,1),(0,2),(0,3)。

    // This is the offset of a specific thread within a warp (units of vectors)
    // 当前例子 WarpThreadArrangement=<8,4>：
    //   lane 0..7   -> strided row group 0，contiguous vector 0..7
    //   lane 8..15  -> strided row group 1，contiguous vector 0..7
    //   lane 16..23 -> strided row group 2，contiguous vector 0..7
    //   lane 24..31 -> strided row group 3，contiguous vector 0..7
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };//warp 内线程的偏移。当前例子 lane 0..7 在 contiguous 方向访问 vector 0..7，lane 8..15 也访问 vector 0..7，但在 strided 方向偏移一行，以此类推。

    // This is the offset of a thread within a threadblock tile (units of vectors)
    // 当前例子：
    //   thread_offset_vec.contiguous = lane_id % 8
    //   thread_offset_vec.strided    = warp_id * 32 + lane_id / 8
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;
      //(8,32) *(0, warp_id) + (lane_id % 8, lane_id / 8)
    // This is the offset of a thread within a threadblock tile (units of elements)
    // contiguous 从“向量编号”换回“元素编号”，所以要乘 kElementsPerAccess=8。
    // 当前例子：
    //   thread_offset_base.contiguous = (lane_id % 8) * 8
    //   thread_offset_base.strided    = warp_id * 32 + lane_id / 8
    //
    // 例子：
    //   thread 0  -> warp 0 lane 0  -> (K=0,  M=0)
    //   thread 1  -> warp 0 lane 1  -> (K=8,  M=0)
    //   thread 7  -> warp 0 lane 7  -> (K=56, M=0)
    //   thread 8  -> warp 0 lane 8  -> (K=0,  M=1)
    //   thread 31 -> warp 0 lane31  -> (K=56, M=3)
    //   thread 32 -> warp 1 lane 0  -> (K=0,  M=32)
    //
    // 再结合 Iterations=<1,8> 和 Delta=<64,4>，thread 0 会访问：
    //   (K=0..7, M=0), (K=0..7, M=4), ..., (K=0..7, M=28)
    // thread 1 会访问：
    //   (K=8..15, M=0), (K=8..15, M=4), ..., (K=8..15, M=28)
    // 4 个 warp 合起来覆盖完整的 K=64, M=128 tile。
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Policy defining a warp-raked arrangement in which a shape is partitioned into contiguous
/// elements. Warps are arranged based on a stride.
///
/// This ThreadMap is used by tensor core kernels for NCxHWx layout.
template <
  typename Shape_,
  int Threads,
  typename WarpThreadArrangement_,
  int ElementsPerAccess = 1
>
struct PitchLinearStridedWarpRakedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  using WarpThreadArrangement = WarpThreadArrangement_;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Base ThreadMap
  using BaseThreadMap = PitchLinearWarpRakedThreadMap<
    Shape,
    kThreads,
    WarpThreadArrangement,
    kElementsPerAccess
  >;

  /// Shape of access by each thread
  using ThreadAccessShape = typename BaseThreadMap::ThreadAccessShape;


  struct Detail {

    using WarpThreadArrangement = WarpThreadArrangement_;

    using WarpAccessIterations = typename BaseThreadMap::Detail::WarpAccessIterations;

    static int const kWarpSize = BaseThreadMap::Detail::kWarpSize;

    static int const kWarpCount = BaseThreadMap::Detail::kWarpCount;

    using ShapeInAccesses = typename BaseThreadMap::Detail::ShapeInAccesses;

    // Divide it into the number of warps, first partitioning the contiguous dimension then the
    // stride.
    static int const kWarpsContiguous =
        (WarpAccessIterations::kContiguous >= kWarpCount
             ? kWarpCount
             : WarpAccessIterations::kContiguous);

    static int const kWarpsStrided =
        (kWarpCount > WarpAccessIterations::kContiguous
             ? kWarpCount / kWarpsContiguous
             : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;

  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta between accesses (units of elements, concept: PitchLinearShape)
  using Delta = typename BaseThreadMap::Delta;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided * Iterations::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }


};

////////////////////////////////////////////////////////////////////////////////

/// Transpose the existing ThreadMap.  For example, interleaved layout is like
/// congruous in the global memory and crosswise in the shared memory.  We need
/// to transpose the coordinates between two.

template <typename ThreadMap_, typename WarpThreadArrangement_>
struct TransposePitchLinearThreadMap {
  /// Underlying ThreadMap
  using ThreadMap = ThreadMap_;

  /// Tensor coordinate
  using TensorCoord = typename ThreadMap::TensorCoord;

  /// Tile shape
  using Shape = typename ThreadMap::Shape;

  /// Number of threads total
  static int const kThreads = ThreadMap::kThreads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(!(Shape::kContiguous % kElementsPerAccess),
                  "Shape must be divisible by vector length.");

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement =
        layout::PitchLinearShape<ThreadMap::Detail::kWarpsStrided,
                                 ThreadMap::Detail::kWarpsContiguous>;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations =
      layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
                               ThreadMap::Iterations::kContiguous>;

  static_assert(Iterations::kContiguous == 1,
    "Contiguous iteration has to be one to reuse the same shared store function with those that don't need transpose");

  static_assert(Iterations::kCount, "Number of iterations must be non-zero");

  ///< Delta between accesses (units of elements, concept: PitchLinearShape)
  using Delta =
      layout::PitchLinearShape<Detail::WarpThreadArrangement::kContiguous *
                                   kElementsPerAccess,
                               Detail::WarpThreadArrangement::kStrided>;

  /// Maps thread ID to a coordinate offset within the tensor's logical
  /// coordinate space Note this is slightly different from the one of
  /// PitchLinearWarpRakedThreadMap.
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access
    // (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
        Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
        Detail::WarpThreadArrangement::kStrided * Iterations::kStrided};

    // This is the offset of a specific warp (in units of vectors)
    // Note the order of / and %. Also the 2nd operand is kStrided.
    layout::PitchLinearCoord warp_offset{
        (warp_id / Detail::WarpArrangement::kStrided),
        (warp_id % Detail::WarpArrangement::kStrided)};

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
        lane_id % Detail::WarpThreadArrangement::kContiguous,
        lane_id / Detail::WarpThreadArrangement::kContiguous};

    // This is the offset of a thread within a threadblock tile (units of
    // vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
        warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of
    // elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
        thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
        thread_offset_in_threadblock_tile_vec.strided()};

    return thread_offset_in_threadblock_tile_base;
  }
};

template <typename ThreadMap_>
struct TransposePitchLinearThreadMapSimt {
    /// Underlying ThreadMap
    using ThreadMap = ThreadMap_;

    /// Tensor coordinate
    using TensorCoord = typename ThreadMap::TensorCoord;

    /// Tile shape
    using Shape = typename ThreadMap::Shape;

    /// Number of threads total
    static int const kThreads = ThreadMap::kThreads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;

    static_assert(kElementsPerAccess == 1 , "Simt transpose requires elements per access to be 1");
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
        layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
        ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    static_assert(Iterations::kStrided == 1,
      "Strided iteration has to be one to reuse the same shared store function with those that don't need transpose");

    /// Shape of access by each thread
    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    ///< Delta between accesses (units of elements, concept: PitchLinearShape)
    using Delta =
        layout::PitchLinearShape<ThreadMap::Delta::kStrided,
        ThreadMap::Delta::kContiguous>;


    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space Note this is slightly different from the one of
    /// PitchLinearWarpRakedThreadMap.
    CUTLASS_HOST_DEVICE
        static TensorCoord initial_offset(int thread_id) {

        TensorCoord coord = ThreadMap::initial_offset(thread_id);

        return TensorCoord(
            coord.strided(),
            coord.contiguous()
        );
    }
};

////////////////////////////////////////////////////////////////////////////////


/// Policy defining a warp-striped arrangement.  This partitions a tile into vectorized memory
/// accesses performed by each warp then distributes warps across them. Warps are striped in the
/// strided dimension and raked across the contiguous dimension.
template <
  typename Shape_,                          /// Overall shape to partition in units of elements
  int Threads,                              /// Number of partiticipation threads
  typename WarpThreadArrangement_,          /// Describes the shape of one memory access per warp
  int ElementsPerAccess = 1                 /// Number of elements accessed by each thread per memory operation (i.e. vector size)
>
struct PitchLinearWarpStripedThreadMap {

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract vector length from Layout
  static int const kElementsPerAccess = ElementsPerAccess;

  /// Shape of access by each thread
  using ThreadAccessShape = layout::PitchLinearShape<kElementsPerAccess, 1>;

  /// Internal details made public to facilitate introspection
  struct Detail {

    /// Fixed arrangement of threads within a warp (units of threads).
    using WarpThreadArrangement = WarpThreadArrangement_;

    /// Number of threads per warp
    static int const kWarpSize = WarpThreadArrangement::kCount;

    /// Number of participating warps
    static int const kWarpCount = kThreads / kWarpSize;

    static_assert(
      !(Shape::kContiguous % kElementsPerAccess),
      "Shape must be divisible by vector length.");

    /// Compute the 'shape' of the overall tile in units of vectors
    using ShapeInAccesses = layout::PitchLinearShape<
      Shape::kContiguous / kElementsPerAccess,
      Shape::kStrided
    >;

    // compute number of warp-level accesses total
    using WarpAccessIterations = layout::PitchLinearShape<
      ShapeInAccesses::kContiguous / WarpThreadArrangement::kContiguous,
      ShapeInAccesses::kStrided / WarpThreadArrangement::kStrided
    >;

    // Divide it into the number of warps, first partitioning the strided dimension then the
    // contiguous.
    static int const kWarpsStrided =
      (WarpAccessIterations::kStrided >= kWarpCount
        ? kWarpCount : (kWarpCount / WarpAccessIterations::kStrided));

    static int const kWarpsContiguous =
      (kWarpCount > WarpAccessIterations::kStrided ?
        WarpAccessIterations::kContiguous / kWarpsStrided : 1);

    /// Arrangement of warps within a threadblock-scoped tile
    using WarpArrangement = layout::PitchLinearShape<
      kWarpsContiguous, kWarpsStrided
    >;
  };

  ///< Iterations along each dimension (concept: PitchLinearShape)
  using Iterations = layout::PitchLinearShape<
    Detail::WarpAccessIterations::kContiguous / Detail::kWarpsContiguous,
    Detail::WarpAccessIterations::kStrided / Detail::kWarpsStrided
  >;

  static_assert(Iterations::kCount,
    "Number of iterations must be non-zero");

  ///< Delta between accesses (units of elements, concept: PitchLinearShape)
  using Delta = layout::PitchLinearShape<
    Detail::WarpThreadArrangement::kContiguous * kElementsPerAccess,
    Detail::WarpThreadArrangement::kStrided * Detail::WarpArrangement::kStrided
  >;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    int warp_id = (thread_id / Detail::kWarpSize);
    int lane_id = (thread_id % Detail::kWarpSize);

    //
    // compute warp-level offset
    //

    // This is the shape of the entire area covered by a warp's memory access (in units of vectors)
    layout::PitchLinearCoord warp_footprint{
      Detail::WarpThreadArrangement::kContiguous * Iterations::kContiguous,
      Detail::WarpThreadArrangement::kStrided
    };

    // This is the offset of a specific warp (in units of vectors)
    layout::PitchLinearCoord warp_offset{
      (warp_id % Detail::kWarpsContiguous),
      (warp_id / Detail::kWarpsContiguous)
    };

    // This is the offset of a specific thread within a warp (units of vectors)
    layout::PitchLinearCoord thread_offset_in_warp{
      lane_id % Detail::WarpThreadArrangement::kContiguous,
      lane_id / Detail::WarpThreadArrangement::kContiguous
    };

    // This is the offset of a thread within a threadblock tile (units of vectors)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_vec =
      warp_footprint * warp_offset + thread_offset_in_warp;

    // This is the offset of a thread within a threadblock tile (units of elements)
    layout::PitchLinearCoord thread_offset_in_threadblock_tile_base{
      thread_offset_in_threadblock_tile_vec.contiguous() * kElementsPerAccess,
      thread_offset_in_threadblock_tile_vec.strided()
    };

    return thread_offset_in_threadblock_tile_base;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Strip-mines a pitch-linear tile among a given number of threads, first along the contiguous
/// dimension then along the strided dimension, while each thread access a 2D thread-tile.
///
/// The tile must be divisible by the thread count such that all threads may execute the same
/// number of iterations with the same delta to exhaustively cover the tile.
///
/// This class satisfies the "RegularThreadMapping" concept.
template <
  typename Shape_,
  int Threads,
        typename ThreadTileShape
>
struct PitchLinear2DThreadTileStripminedThreadMap;


template <
  typename Shape_,
  int Threads
>
struct PitchLinear2DThreadTileStripminedThreadMap <Shape_, Threads, cutlass::layout::PitchLinearShape<4, 4>>{

  /// Tensor coordinate
  using TensorCoord = layout::PitchLinearCoord;

  /// Tile shape
  using Shape = Shape_;

  /// Access Shape of each thread
  using ThreadAccessShape = cutlass::layout::PitchLinearShape<4, 4>;
  //using ThreadAccessShape = ThreadTileShape;

  /// Number of threads total
  static int const kThreads = Threads;

  /// Extract length of each access from Layout
  static int const kElementsPerAccess = ThreadAccessShape::kContiguous;

  static_assert(!(kElementsPerAccess % 4) , "kElementsPerAccess, needs to be multiple of 4 (32bits)");

  /// Internal implementation details
  struct Detail {

    static_assert(!(ThreadAccessShape::kContiguous % 4), "ThreadAccessShape, needs to be multiple of 4");

    static_assert(!(Shape::kContiguous % ThreadAccessShape::kContiguous), "");

    static_assert(!((Shape::kContiguous * Shape::kStrided) % (kThreads * ThreadAccessShape::kCount)),
      "Shape must be divisible thread count * accesses per thread.");

    /// Shape of the tile in units of vectors
    using ShapeVec = layout::PitchLinearShape<
      Shape::kContiguous / ThreadAccessShape::kContiguous,
      Shape::kStrided / ThreadAccessShape::kStrided
    >;

    static_assert(
      (Threads < ShapeVec::kContiguous && !(ShapeVec::kContiguous % kThreads)) ||
      (!(kThreads % ShapeVec::kContiguous) && !(ShapeVec::kStrided % (kThreads / ShapeVec::kContiguous))),
      "Shape must be divisible by number of iterations of each thread."
    );
  };

  /// Number of iterations by each thread
  using Iterations = typename platform::conditional<
      Threads >= Detail::ShapeVec::kContiguous,
      layout::PitchLinearShape<
          1,
          // Redo the comparison here to work around divide by zero compiler
          // error.  The compiler evaluates both path of platform::conditional.
          (Threads >= Detail::ShapeVec::kContiguous
               ? Detail::ShapeVec::kStrided /
                     (kThreads / Detail::ShapeVec::kContiguous)
               : 0)>,
      layout::PitchLinearShape<Detail::ShapeVec::kContiguous / kThreads,
                               Detail::ShapeVec::kStrided>>::type;

  /// Interval between accesses along each dimension of the tensor's logical coordinate space
  /// (in units of Elements)
  using Delta = typename platform::conditional<
    Threads >= Detail::ShapeVec::kContiguous,
    layout::PitchLinearShape<
      Shape::kContiguous,
      kThreads * ThreadAccessShape::kStrided / Detail::ShapeVec::kContiguous
    >,
    layout::PitchLinearShape<
      kThreads * ThreadAccessShape::kContiguous,
      1
    >
  >::type;

  /// Maps thread ID to a coordinate offset within the tensor's logical coordinate space
  /// (in units of Elements)
  CUTLASS_HOST_DEVICE
  static TensorCoord initial_offset(int thread_id) {

    return TensorCoord(
      (thread_id % Detail::ShapeVec::kContiguous) * ThreadAccessShape::kContiguous,
      (thread_id / Detail::ShapeVec::kContiguous) * ThreadAccessShape::kStrided);
  }
};

/// Thread Mapping a 2D threadtiled mapping as a transposed Pitchlinear2DThreadTile mapping
template <typename ThreadMap_>
struct TransposePitchLinearThreadMap2DThreadTile {
    /// Underlying ThreadMap
    using ThreadMap = ThreadMap_;

    /// Tensor coordinate
    using TensorCoord = typename ThreadMap::TensorCoord;

    /// Tile shape
    using Shape = typename ThreadMap::Shape;

    /// Number of threads total
    static int const kThreads = ThreadMap::kThreads;

    /// Extract vector length from Layout
    static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;


    static_assert(kElementsPerAccess > 1 , "Simt transpose requires elements per access to be 1");
    ///< Iterations along each dimension (concept: PitchLinearShape)
    using Iterations =
        layout::PitchLinearShape<ThreadMap::Iterations::kStrided,
        ThreadMap::Iterations::kContiguous>;

    static_assert(Iterations::kCount, "Number of iterations must be non-zero");

    /// Shape of access by each thread
    using ThreadAccessShape = typename ThreadMap::ThreadAccessShape;

    ///< Delta between accesses (units of elements, concept: PitchLinearShape)
    using Delta =
        layout::PitchLinearShape<ThreadMap::Delta::kStrided,
        ThreadMap::Delta::kContiguous>;


    /// Maps thread ID to a coordinate offset within the tensor's logical
    /// coordinate space Note this is slightly different from the one of
    /// PitchLinearWarpRakedThreadMap.
    CUTLASS_HOST_DEVICE
        static TensorCoord initial_offset(int thread_id) {

        TensorCoord coord = ThreadMap::initial_offset(thread_id);
        return TensorCoord(
            coord.strided(),
            coord.contiguous()
        );
    }
};


/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace transform
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
