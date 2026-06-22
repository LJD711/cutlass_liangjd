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
    \brief Templates implementing computing the addresses of storing of tiles
   from pitch-linear rank=2 tensors.
*/

#pragma once

#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace transform {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for congruous arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                          Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout =
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;
  static int const kCrosswise = Crosswise;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using StrideIndex = typename Layout::Stride::Index;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Internal details made public to facilitate introspection
  struct Detail {
    /// This iterator is specialized for an access size that is 128 bits in
    /// length.
    static int const kAccessSizeInBits = 128;

    static_assert(sizeof_bits<Element_>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");

    ///< Number of pointers
    static int const kPointerCount =
        (ThreadMap::Iterations::kStrided > 1 ? 2 : 1);
  };

  /// Element type per access
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 private:
  //
  // Data members
  //

  /// Stride value
  StrideIndex stride_;

  /// Internal pointer to first access of tile
  AccessType *pointer_[Detail::kPointerCount];

  /// Internal byte offset
  Index byte_offset_;

  /// Iteration in the contiguous dimension
  int iteration_contiguous_;

  /// Iteration in the strided dimension
  int iteration_strided_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        byte_offset_(0) {
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // This is the offset of a thread within a threadblock tile for a specific
      // pointer (units of elements)
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // initialize pointer
      pointer_[i] = reinterpret_cast<AccessType *>(
          ref.data() + ref.offset(thread_offset_in_threadblock_tile));
    }

    set_iteration_index(0);
  }

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof(Element);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];
    int stride_idx = (iteration_strided_ & ~1);

    int access_offset = stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
                        iteration_contiguous_ * ThreadMap::Delta::kContiguous /
                            ThreadMap::kElementsPerAccess;

    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // Enter here only if (iteration_contiguous_ ==
    // ThreadMap::Iteration::kContiguous)
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // Enter here only if (iteration_strided_ == ThreadMap::Iteration::kStrided)
    // which means we enter the next tile.
    iteration_strided_ = 0;

    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * Shape::kContiguous * Layout::kFactor +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess / Layout::kFactor);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for column-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major congruous TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::RowMajorTensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                  Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile iterator specialized for crosswise arrangements for TensorOps
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
/// 以单元测试
///
///   SM80_Device_Conv2d_Fprop_Optimized_ImplicitGemm_
///   f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_align4
///
/// 的 operand A 为例，这个底层 pitch-linear 偏特化的完整实例化是：
///
///   Shape_       = layout::PitchLinearShape<64, 128>
///   Element_     = cutlass::half_t
///   Layout       = layout::TensorOpMultiplicandCrosswise<16, 64>
///   AdvanceRank  = 1
///   ThreadMap_   = PitchLinearWarpRakedThreadMap<
///                    PitchLinearShape<64, 128>, 128,
///                    PitchLinearShape<8, 4>, 8>
///   Alignment    = 16 bytes
///   Crosswise    = 64 elements
///
/// 注意：test 中的 AlignmentA=4 描述 global-memory activation iterator
/// 每次读取 4 个 half。它不会传入本 shared-memory iterator。这里的
/// Alignment 使用主模板默认值：
///
///   16 bits/half * 8 half/access / 8 = 16 bytes
///
/// 所以 global 的一个 8-half ThreadMap 向量会由两条 8-byte cp.async 写入，
/// 但 shared iterator 的一个 AccessType 仍表示完整的 8 half / 128 bits。
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<Shape_, Element_,
                                layout::TensorOpMultiplicandCrosswise<
                                    sizeof_bits<Element_>::value, Crosswise>,
                                AdvanceRank, ThreadMap_, Alignment> {//liangjd
 public:
  // AdvanceRank 指明 add_tile_offset() 的“前进维”。本实例中外层
  // RowMajorTensorOpMultiplicandCrosswise wrapper 的 AdvanceRank=0（矩阵 column/K
  // 方向），转换到 pitch-linear 坐标后成为 rank=1。这里允许 0 或 1。
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for pitch-linear iterator may along advance along the "
      "contiguous(rank=0) or strided(rank=1) dimension.");

  // 当前 pitch-linear tile 的逻辑形状，单位是标量 Element：
  //   contiguous = 64  -> GEMM A 的 K 方向
  //   strided    = 128 -> GEMM A 的 M/NPQ 方向
  using Shape = Shape_; // PitchLinearShape<64, 128>

  // shared memory 中保存的标量类型。当前为 half_t，16 bits / 2 bytes。
  using Element = Element_; // cutlass::half_t

  // shared-memory 物理布局。它保留逻辑的 pitch-linear 坐标语义，但通过
  // TensorOpMultiplicand 的 XOR swizzle 将 128-bit 向量映射到物理地址，
  // 以降低后续 ldmatrix 读取时的 bank conflict。
  using Layout =
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>; // <16 bits, 64 elements>

  // tile offset 的推进维。当前值为 1。该常量主要保留 iterator 的类型语义；
  // 本类 add_tile_offset() 接收完整的二维 coord，并分别计算两个方向的偏移。
  static int const kAdvanceRank = AdvanceRank; // 1

  // 访问对齐要求，单位是 bytes。当前为 16 bytes，即 128-bit 对齐。
  static int const kAlignment = Alignment; // 16

  // crosswise section 在 contiguous/K 方向包含的标量元素数。当前为 64 half，
  // 恰好等于一个 CTA K tile；不同 pipeline stage 沿这个方向连续排布。
  static int const kCrosswise = Crosswise; // 64

  // 普通索引类型，用于 byte_offset_ 等单个 iterator 内部状态。当前为 int32_t。
  using Index = typename Layout::Index; // int32_t

  // 长索引类型，用于 pointer/tile offset，避免大型张量偏移溢出。当前为 int64_t。
  using LongIndex = typename Layout::LongIndex; // int64_t

  // Layout::Stride 是 Coord<1, int32_t, int64_t>；其 Index 为 int32_t。
  // stride_ 保存的是经过“标量元素 -> 128-bit AccessType”换算后的步长。
  using StrideIndex = typename Layout::Stride::Index; // int32_t

  // 指向 shared-memory A tile 的 TensorRef，包含 Element* 和 swizzled Layout。
  using TensorRef = TensorRef<Element, Layout>;

  // pitch-linear 逻辑坐标：(contiguous, strided)，当前分别对应 (K, M/NPQ)。
  using TensorCoord = typename Layout::TensorCoord; // layout::PitchLinearCoord

  // CTA 内线程如何覆盖 64x128 pitch-linear tile。当前关键常量：
  //   kThreads                  = 128
  //   kElementsPerAccess        = 8 half
  //   Iterations                = <1, 8>
  //   Delta                     = <64, 4>
  //   WarpThreadArrangement     = <8, 4>
  using ThreadMap = ThreadMap_;

  // 当前 Delta::kContiguous=64，kCrosswise=64，64 % 64 == 0。
  // 要求每次 contiguous iteration 的跨度不能落在 crosswise section 中间，
  // 否则下面按 section 计算地址的公式不成立。
  static_assert(!(ThreadMap::Delta::kContiguous % kCrosswise),
                "kCrosswise is the smallest unit in the contiguous dimension "
                "for shared memory swizzling.");

  /// Internal details made public to facilitate introspection
  struct Detail {
    // 本 iterator 的一个逻辑 shared-memory access 固定为 128 bits。
    // 当前就是 8 half = 8 * 16 = 128 bits。
    static int const kAccessSizeInBits = 128; // 16 bytes

    // 当前检查：16 bits/half * 8 half/access == 128 bits。
    // 这也解释了为什么 global AlignmentA=4 不改变本类的 AccessType：
    // 两条 4-half cp.async 最终共同填满一个 8-half shared access。
    static_assert(sizeof_bits<Element_>::value *
                          ThreadMap::kElementsPerAccess ==
                      kAccessSizeInBits,
                  "This iterator requires a policy whose access size is 128bs");

    // 保存几条预计算基础指针。当前 ThreadMap::Iterations::kStrided=8 > 1，
    // 因而 kPointerCount=2。
    //
    // pointer_[0] 预计算当前线程第 0 个 strided 相位的 swizzled 地址；
    // pointer_[1] 预计算向 strided 方向额外偏移 4 行后的地址。
    // get() 用 iteration_strided_ 的最低位在二者间交替，从而避免每次访问
    // 都重新执行完整 layout swizzle。剩余的偶数部分由 stride_idx 处理。
    //
    // Note: TN kblock32 layouts logically only need 1 pointer, but reducing the
    // pointer count has historically hurt performance.
    static int const kPointerCount =
        (ThreadMap::Iterations::kStrided > 1 ? 2 : 1); // 当前为 2
  };

  // 一次完整 shared-memory 访问的数据类型。
  // Layout::kElementsPerAccess = 128 / 16 = 8，所以当前为：
  //
  //   Array<half_t, 8>
  //
  // 大小为 16 bytes。global align4 的两条 8-byte cp.async 会写入这个
  // 16-byte 逻辑 access 的前半和后半。
  using AccessType = Array<Element, Layout::kElementsPerAccess>; // Array<half_t, 8>

 private:
  //
  // Data members
  //

  // 整个 shared-memory leading dimension 中包含多少个 crosswise section。
  //
  // 当前 Stages=3，A 的 shared storage 逻辑形状为：
  //
  //   MatrixShape<M=128, K=64*3=192>
  //
  // RowMajorTensorOpMultiplicandCrosswise::packed() 令 ref.stride(0)=192。
  // 每个 section 包含 kCrosswise=64 个 contiguous 元素，所以：
  //
  //   sections_ = 192 / 64 = 3
  //
  // 当前每个 section 正好对应一个 pipeline stage 的 K=64 tile。
  int sections_;

  // 当前 iterator 的一个 Shape tile 包含多少个 section：
  //
  //   sections_per_stage_ = Shape::kContiguous / kCrosswise
  //                       = 64 / 64
  //                       = 1
  //
  // 泛化情况下，一个 stage 可能包含多个 crosswise section。
  int sections_per_stage_;

  // 经过 Layout factor 和 AccessType 宽度缩放后的 leading-dimension stride。
  // 它不是 byte stride，也不是 half 元素 stride，而是 get()/tile offset 公式
  // 使用的内部 128-bit access 步长：
  //
  //   stride_ = ref.stride(0) * Layout::kFactor /
  //             Layout::kElementsPerAccess
  //           = 192 * 1 / 8
  //           = 24
  //
  // Layout::kFactor=1，Layout::kElementsPerAccess=8。
  StrideIndex stride_;

  // 当前线程的两条预计算 swizzled 基础指针，元素类型是 AccessType*，
  // 因而 pointer + 1 前进 16 bytes，而不是前进一个 half。
  //
  // pointer_[0] 对应 ThreadMap::initial_offset(thread_id)；
  // pointer_[1] 对应该坐标再加 (contiguous=0, strided=4)。
  // 其中 4 来自 WarpThreadArrangement::kStrided。
  AccessType *pointer_[Detail::kPointerCount];

  // 额外的动态 byte offset。它与 pointer_ 分离保存，使 add_pointer_offset()
  // 不需要修改两个基础指针。构造时为 0。
  //
  // 注意单位是 byte；add_pointer_offset() 的输入单位则是 Element。
  Index byte_offset_;

  // 当前 tile 内的 contiguous iteration 编号。当前 Iterations::kContiguous=1，
  // 所以它在有效访问时总为 0；operator++() 每次立即将其回绕。
  int iteration_contiguous_;

  // 当前 tile 内的 strided iteration 编号，范围 0..7。
  //
  // 对 thread 0 来说，八次访问对应 M 行：
  //   0, 4, 8, 12, 16, 20, 24, 28
  //
  // get() 用其最低位选择 pointer_[0]/pointer_[1]，用其余偶数部分计算
  // 更远的 strided 地址增量。
  int iteration_strided_;

 public:
  // 构造当前线程的 shared-memory write iterator。
  //
  // ref：
  //   指向整个三阶段 A shared buffer；当前 layout stride 为 192 half。
  //
  // thread_id：
  //   CTA 内线程编号 0..127，用 ThreadMap::initial_offset() 映射到本线程
  //   在 64x128 pitch-linear tile 中的初始 (K,M) 坐标。
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : sections_(ref.stride(0) / kCrosswise), // 192 / 64 = 3
        sections_per_stage_(Shape::kContiguous / kCrosswise), // 64 / 64 = 1
        // stride_ = kCrosswise x sections_ x kFactor
        stride_(ref.stride(0) * Layout::kFactor /
                Layout::kElementsPerAccess), // 192 * 1 / 8 = 24
        byte_offset_(0) { // 初始无附加 byte offset
    // 当前 ThreadMap 初始坐标公式：
    //
    //   warp_id = thread_id / 32
    //   lane_id = thread_id % 32
    //
    //   thread_offset_base.contiguous = (lane_id % 8) * 8
    //   thread_offset_base.strided    = warp_id * 32 + lane_id / 8
    //
    // 例如：
    //   thread 0  -> (K=0,  M=0)
    //   thread 1  -> (K=8,  M=0)
    //   thread 8  -> (K=0,  M=1)
    //   thread 32 -> (K=0,  M=32)
    layout::PitchLinearCoord thread_offset_base =
        ThreadMap::initial_offset(thread_id);

    CUTLASS_PRAGMA_UNROLL
    // 当前 i=0,1，分别初始化两个 strided 相位的 swizzled 基础地址。
    for (int i = 0; i < Detail::kPointerCount; ++i) {
      // 当前偏移为：
      //
      //   i=0: thread_offset_base + (0, 0)
      //   i=1: thread_offset_base + (0, 4)
      //
      // 单位是 half 元素坐标，不是 byte 或 AccessType。
      layout::PitchLinearCoord thread_offset_in_threadblock_tile =
          thread_offset_base +
          layout::PitchLinearCoord{
              0, ThreadMap::Detail::WarpThreadArrangement::kStrided * i};

      // ref.offset() 调用 TensorOpMultiplicandCrosswise layout，将逻辑 (K,M)
      // 坐标转换成 XOR-swizzled 的标量 half offset。
      //
      // reinterpret_cast 后 pointer 运算单位变成 AccessType=8 half，所以标量
      // offset 要除以 Layout::kElementsPerAccess=8。
      pointer_[i] = reinterpret_cast<AccessType *>(ref.data()) +
                    ref.offset(thread_offset_in_threadblock_tile) /
                        Layout::kElementsPerAccess;
    }

    // 当前初始化为 iteration_contiguous_=0、iteration_strided_=0。
    set_iteration_index(0);
  }

  // 将线性 ThreadMap access 编号拆成 contiguous/strided 两维。
  //
  // 当前 Iterations::kContiguous=1，因此：
  //
  //   iteration_contiguous_ = 0
  //   iteration_strided_    = index
  //
  // index 的有效范围为 0..7。这里的 index 是 shared iterator 的完整
  // 8-half 逻辑 access 编号，不是 global align4 iterator 的 4-half 子访问编号。
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    iteration_contiguous_ = index % ThreadMap::Iterations::kContiguous;
    iteration_strided_ = index / ThreadMap::Iterations::kContiguous;
  }

  // 添加以 Element 为单位的逻辑 pointer offset，但内部保存成 byte。
  // 当前 Element=half，所以 pointer_offset=64 会增加 128 bytes。
  //
  // 它不改变 pointer_[0/1]，只更新 byte_offset_，最终由 get() 叠加。
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    byte_offset_ += pointer_offset * sizeof_bits<Element>::value / 8;
  }

  // 返回当前 ThreadMap iteration 对应的 128-bit shared-memory 地址。
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    // iteration_strided_ 的最低位选择预计算相位：
    //
    //   strided iteration 0,2,4,6 -> pointer_[0]
    //   strided iteration 1,3,5,7 -> pointer_[1]
    //
    // pointer_[1] 本身已经比 pointer_[0] 多表示 4 个 strided 行。
    AccessType *access_ptr = pointer_[iteration_strided_ & 1];

    // 清除最低位，留下需要额外跨越的偶数 iteration 数：
    //
    //   iteration_strided_: 0 1 2 3 4 5 6 7
    //   stride_idx:         0 0 2 2 4 4 6 6
    //
    // 结合 pointer_[0/1] 后，对应行偏移为 0,4,8,12,16,20,24,28。
    int stride_idx = (iteration_strided_ & ~1);

    // access_offset 的单位是 AccessType（每单位 8 half / 16 bytes）。
    //
    // 当前 contiguous iteration 恒为 0，所以第二项为 0。第一项化简为：
    //
    //   access_offset = stride_idx * 4 * 24 / 1
    //                 = stride_idx * 96 AccessType
    //
    // 得到 0,0,192,192,384,384,576,576 个 AccessType 的额外偏移。
    // 每增加 192 AccessType = 192*8=1536 half = 8 行*192 half/行，
    // 正好在每两次 iteration 后再向 M 方向推进 8 行。
    int access_offset =
        stride_idx * ThreadMap::Delta::kStrided * stride_ / Layout::kFactor +
        // kCrosswise elements in the contiguous dimension would span to a
        // shared memory cache line.
        iteration_contiguous_ * (ThreadMap::Delta::kContiguous / kCrosswise) *
            Layout::TileShape::kContiguous;

    // 先按 AccessType 单位叠加 access_offset，再转成 char*，以便精确叠加
    // byte_offset_。当前正常 copy 路径下 byte_offset_ 初始为 0；切换 shared
    // stage 时 add_tile_offset() 会通过它增加或减少 64 half = 128 bytes。
    char *access_byte_ptr =
        reinterpret_cast<char *>(access_ptr + access_offset);

    // 返回当前 16-byte shared access 的最终地址。
    return reinterpret_cast<AccessType *>(access_byte_ptr + byte_offset_);
  }

  // 前进到当前 tile 内的下一个 ThreadMap 逻辑 access。
  // 名字虽然是 operator++，但这里不会切换 pipeline stage；它只遍历
  // Iterations=<1,8> 的内部访问状态。
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    // 当前 kContiguous=1，所以每次 ++ 后都立即进入 contiguous 回绕。
    ++iteration_contiguous_;

    if (iteration_contiguous_ < ThreadMap::Iterations::kContiguous)
      return *this;

    // contiguous iteration 完成，回到 0，并进入下一个 strided iteration。
    iteration_contiguous_ = 0;
    ++iteration_strided_;

    // iteration_strided_ 依次为 0..7；未完成 8 次访问时直接返回。
    if (iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    // 当前 tile 的 8 个逻辑 access 全部遍历完成后回绕到 (0,0)。
    // 真正切换 shared pipeline stage 由 add_tile_offset() 完成。
    iteration_strided_ = 0;

    return *this;
  }

  // 后置递增：返回递增前的 iterator 副本，当前对象按上述规则前进一次。
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    this->operator++();

    return prev;
  }

  // 按 pitch-linear tile 坐标移动 shared-memory 基址。
  //
  // 对 operand A，外层 row-major wrapper 调用：
  //
  //   add_tile_offset(MatrixCoord{0, 1})
  //
  // 会转换成底层：
  //
  //   coord = PitchLinearCoord{1, 0}
  //
  // 当前第一项计算为：
  //
  //   1 * sections_per_stage_(1) * stride_(24) *
  //       ThreadMap::kElementsPerAccess(8) / sections_(3)
  //   = 64 half
  //
  // 所以 shared write iterator 前进一个 stage 时增加 64 half = 128 bytes。
  // 当环形缓冲回绕时 coord.contiguous()=-3，得到 -192 half，正好从
  // stage3 末端回到 stage0。
  //
  // 第二项负责沿 strided tile 方向移动；当前 operand A 的 stage 切换不使用它。
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    add_pointer_offset(coord.contiguous() * sections_per_stage_ * stride_ *
                           ThreadMap::kElementsPerAccess / sections_ +
                       coord.strided() * Shape::kStrided * stride_ *
                           Layout::kElementsPerAccess / Layout::kFactor);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for column-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<
    Shape_, Element_,
    layout::ColumnMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    AdvanceRank, ThreadMap_, Alignment> {
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for column-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 0 : 1), ThreadMap_>;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Tile Iterator specialized for row-major crosswise TensorOp formats.
///
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept
///
/// 这是主模板的一个偏特化。主模板参数顺序是：
///
///   RegularTileAccessIterator<Shape, Element, Layout, AdvanceRank,
///                             ThreadMap, Alignment>
///
/// 下面 template<...> 这一行只是声明“这个偏特化还需要推导哪些参数”，
/// 因此其中的 `int AdvanceRank` 不是主模板的第三个参数。主模板的第三个
/// 参数 Layout 已经被下面的偏特化模式固定为：
///
///   layout::RowMajorTensorOpMultiplicandCrosswise<...>
///
/// 所以如下实例化：
///
///   RegularTileAccessIterator<MatrixShape<M,K>, ElementA, SmemLayoutA, 0,
///                             IteratorThreadMapA>
///
/// 会按下面的方式匹配到本偏特化：
///
///   Shape_        = MatrixShape<M,K>
///   Element_      = ElementA
///   Layout        = RowMajorTensorOpMultiplicandCrosswise<...>  // SmemLayoutA
///   AdvanceRank   = 0
///   ThreadMap_    = IteratorThreadMapA
///   Alignment     = 主模板默认值
///
template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, int Alignment, int Crosswise>
class RegularTileAccessIterator<Shape_, Element_,
                                layout::RowMajorTensorOpMultiplicandCrosswise<
                                    sizeof_bits<Element_>::value, Crosswise>,
                                AdvanceRank, ThreadMap_, Alignment> {//liangjd
 public:
  static_assert(
      AdvanceRank == 0 || AdvanceRank == 1,
      "Specialization for row-major iterator may along advance along the "
      "columns(rank=0) or rows(rank=1) dimension.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, Crosswise>;
  static int const kAdvanceRank = AdvanceRank;
  static int const kAlignment = Alignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = ThreadMap_;

  /// Underlying iterator type
  ///
  /// row-major 矩阵坐标会被转换成 TensorOpMultiplicandCrosswise 使用的
  /// pitch-linear 坐标约定：
  ///
  ///   MatrixCoord(row, column)
  ///      -> PitchLinearCoord(contiguous = column, strided = row)
  ///
  /// 这个 wrapper 自己不计算 XOR swizzle，而是转交给 pitch-linear 版本的
  /// TensorOpMultiplicandCrosswise iterator。后者的 TensorRef 会调用
  /// TensorOpMultiplicandCrosswise::operator()，最终进入
  /// TensorOpMultiplicand::operator() 完成地址置换。
  using UnderlyingIterator = RegularTileAccessIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element,//64 128
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            Crosswise>,
      (kAdvanceRank == 0 ? 1 : 0), ThreadMap_>;//匹配435行的RegularTileAccessIterator

  using AccessType = typename UnderlyingIterator::AccessType;

 private:
  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      // `ref` 携带的是 RowMajorTensorOpMultiplicandCrosswise。这里传入
      // {ref.data(), ref.stride()}，构造 UnderlyingIterator 使用的
      // pitch-linear TensorRef。之后 UnderlyingIterator 初始化每个线程的
      // shared memory 指针时，调用链是：
      //
      //   ThreadMap::initial_offset(thread_id)
      //   -> TensorRef::offset(coord)
      //   -> TensorOpMultiplicandCrosswise::operator()(coord)
      //   -> TensorOpMultiplicand::operator()(coord)  // XOR swizzle
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    // 返回当前 128-bit access 对应的 swizzled shared-memory 地址。
    // 初始地址由上面的 layout 映射算出，后续再由底层 iterator 的
    // iteration state 推进。
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    // row-major tile offset 先转换成 pitch-linear tile offset，
    // 再交给底层 crosswise iterator 处理。
    iterator_.add_tile_offset({coord.column(), coord.row()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  RegularTileAccessIterator operator++(int) {
    RegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};

////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
