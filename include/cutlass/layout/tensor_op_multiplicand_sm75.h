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
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/coord.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/layout/pitch_linear.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace layout {

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
/// This one is the base class of all Ampere/Turing fp16/bf16/int8/int4/int1
/// tensor core kernels.  tf32 TN uses this too.
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicand {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Static constants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  static int const kElementSize = ElementSize;//16
  static int const kElementsPerAccess = kAccessSize / kElementSize;//128/16 = 8 8个单元
  static int const kCrosswise = Crosswise;//64

  /// Contiguous dimension of the tile shape matches one shared memory cache
  /// line - 128B.  For 128bit access size, it equals to 8 accesses.
  static int const kTileShapeContiguous = 128 / (kAccessSize / 8);//8个新元素    kAccessSize / 8 表示一个新元素占多少字节

  /// Number of kblocks to store PartitionShape::kContiguous Elements
  static int const kFactor =
      kTileShapeContiguous * kElementsPerAccess / kCrosswise;//8*8/64 = 1  8个新元素，每个元素是8个单元，一共64个单元，正好是一个kblock

  static_assert(
      (kFactor > 0),
      "kCrosswise should be no large than one shared memory cache line.");

  /// The strided dimension needs to be at least (WarpSize(32) /
  /// kTileShapeContiguous) for a warp to access.  To ensure conflict free
  /// access, it also needs to be at least (kTileShapeContiguous / kFactor).
  /// See comments below
  static int const kTileShapeStride =
      ((kTileShapeContiguous / kFactor) > (32 / kTileShapeContiguous)) // 8/1 > 32/8
          ? (kTileShapeContiguous / kFactor) //8
          : (32 / kTileShapeContiguous); //1

  /// Fundamental tile shape in units of vectors to guarantee bank conflict free
  /// shared memory load/store.
  /// For kFactor = 1, TileShape = <8, 8> 
  /// For kFactor > 1, TileShape = <8, 4>
  using TileShape = PitchLinearShape<kTileShapeContiguous, kTileShapeStride>;//8x8

  /// Fundamental partition shape in units of vectors
  using PartitionShape = PitchLinearShape<4, 4>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

 private:
  //
  // Data members
  //

  /// Stride data member. For GEMM, it equals to kCrosswise x stage.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicand(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicand(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicand packed(TensorCoord const &extent) {
    return TensorOpMultiplicand(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {//liangjd
    // 输入坐标约定：
    //
    //   coord.contiguous() : 逻辑连续维度。例如 RowMajorTensorOpMultiplicandCrosswise
    //                        中对应 K/column 方向。
    //   coord.strided()    : 逻辑非连续维度。例如 row-major A tile 中对应 M/row
    //                        方向。
    //
    // 这个 layout 以 128-bit 向量为基本访问单位。以 fp16 为例，
    // kElementsPerAccess = 128 / 16 = 8，所以连续维度上的元素 [0..7]
    // 属于第 0 个 128-bit 向量，[8..15] 属于第 1 个向量。swizzle 发生在
    // “向量坐标”上，不会打乱同一个 128-bit 向量内部的 8 个 fp16 元素。
    //
    // 地址计算分三层：
    //
    //   1. vec_*：       当前元素属于逻辑张量中的第几个 128-bit 向量。
    //   2. tile_*：      当前向量位于哪个 fundamental shared-memory tile，
    //                    以及在该 tile 内的位置。fp16 + Crosswise=64 时，
    //                    TileShape 通常是 8x8 个 128-bit 向量。
    //   3. partition_*： 每个 tile 再切成 4x4 个 128-bit 向量的小分区。
    //                    下面两个 XOR 会分别置换“分区内的向量位置”和
    //                    “tile 内的分区位置”。
    //
    // 目标：让相邻 strided 行在后续 warp 执行 ldmatrix/ldsm 时落到更分散的
    // shared-memory bank 上。逻辑上矩阵仍然保持外层 wrapper 表达的
    // row-major/column-major 坐标语义，但 shared memory 里的物理地址被置换。
    //
    // 先把标量元素坐标转换成 128-bit 向量坐标。
    //

    // contiguous 方向第几个 128-bit 向量。除以 kElementsPerAccess 是把
    // 标量元素下标转换成向量下标。 coord.contiguous() = 43 coord.strided() = 5
    int vec_contiguous_idx = coord.contiguous() / kElementsPerAccess; //第几个新的元素  5

    // strided 方向第几组。kFactor 表示一个 strided 坐标会被拆到多少个
    // crosswise section 中；除以 kFactor 是得到 section 组号。
    int vec_strided_idx = coord.strided() / kFactor;// 5

    // 当前向量位于第几个 fundamental tile 的 contiguous 方向。
    // TileShape::kContiguous / kFactor 是一个 tile 在 contiguous 方向容纳的
    // 向量数；整除得到 tile 编号。
    int tile_contiguous_idx =
        vec_contiguous_idx / (TileShape::kContiguous / kFactor);// 0

    // 当前向量在该 tile 的 contiguous 方向余数。
    // 第一项 `%` 得到向量在当前 tile 内的本地 contiguous 位置。
    // 第二项处理 kFactor > 1 时同一 strided 坐标拆到不同 crosswise section
    // 的情况，把 coord.strided() % kFactor 折叠进 contiguous 方向。
    int tile_contiguous_residual =
        vec_contiguous_idx % (TileShape::kContiguous / kFactor) +
        ((coord.strided() % kFactor) * (TileShape::kContiguous / kFactor));//5

    // 当前向量在该 tile 的 strided 方向余数。`% TileShape::kStrided` 表示
    // 只关心 tile 内部第几行。
    int tile_strided_residual = vec_strided_idx % TileShape::kStrided;// 5

    // tile 内部再切成 4x4 vector partition。这里的除法得到当前属于第几个
    // partition。一共是8x8，切成4x4，就是2x2个partition。
    int partition_contiguous_idx =
        tile_contiguous_residual / PartitionShape::kContiguous;// 1
    int partition_strided_idx =
        tile_strided_residual / PartitionShape::kStrided; // 1  得出是第几个partition

    // `%` 得到当前向量在 4x4 partition 内部的位置。
    int partition_contiguous_residual =
        tile_contiguous_residual % PartitionShape::kContiguous;
    int partition_strided_residual =
        tile_strided_residual % PartitionShape::kStrided; //在partition内的行号和列号

    //
    // 然后做 shared-memory swizzle。
    //

    // partition 内部的 XOR：用 partition 内第几行
    // partition_strided_residual 去异或低位 contiguous 向量下标。
    // 对 4x4 partition 来说，partition_strided_residual % 4 就是行号，
    // 这样不同行访问同一 logical column 时会被打散到不同 bank。
    int permuted_vec_contiguous_within_partition =
        partition_contiguous_residual ^ (partition_strided_residual % 4);//0 partition 内部位置从 contiguous=1 变成 contiguous=0。

    // partition 级别的 XOR：用 partition_strided_idx 的奇偶性翻转
    // partition_contiguous_idx，使相邻 partition 行在 contiguous 方向交错。
    int permuted_partition_contiguous_within_tile =
        partition_contiguous_idx ^ (partition_strided_idx % 2);//0 partition 本身从 contiguous partition 1 变成 0。

    //
    // 根据 swizzle 后的位置重新合成最终标量元素坐标。
    //

    // swizzle 后的 contiguous 标量元素下标：
    //   1. tile_contiguous_idx * TileShape::kContiguous 定位到第几个 tile；
    //   2. permuted_partition_contiguous_within_tile * PartitionShape::kContiguous
    //      定位到 swizzle 后第几个 4x4 partition；
    //   3. permuted_vec_contiguous_within_partition 定位到 partition 内第几个
    //      128-bit 向量；
    //   4. 乘 kElementsPerAccess 从向量下标转回标量元素下标；
    //   5. 加 coord.contiguous() % kElementsPerAccess，保留向量内部元素偏移。
    int element_contiguous = (tile_contiguous_idx * TileShape::kContiguous +
                              permuted_partition_contiguous_within_tile *
                                  PartitionShape::kContiguous +
                              permuted_vec_contiguous_within_partition) *
                                 kElementsPerAccess +
                             (coord.contiguous() % kElementsPerAccess);// (0 * 8 + 0 * 4 + 0) * 8 + (43 % 8) = 3

    // strided 方向不参与 XOR swizzle；它使用前面按 kFactor 分组后的行号。
    int element_strided = vec_strided_idx;//5

    // 最终线性 offset：
    //
    //   element_contiguous
    //     + element_strided * stride_[0] * kFactor
    //
    // stride_[0] 来自 packed shared-memory TensorRef，是逻辑 leading
    // dimension。kFactor 用来补偿 kFactor > 1 时 strided 方向拆分到多个
    // crosswise section 的情况。
    return element_contiguous + element_strided * stride_[0] * kFactor;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCongruous {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(coord);
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return coord;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(extent);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
/// This one is just for TF32 NT kernel.
template <int Crosswise>
struct TensorOpMultiplicandCongruous<32, Crosswise> {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  /// Fundamental tile shape in units of vectors
  using TileShape = PitchLinearShape<8, 4>;

  /// Partitionshape is the same as TileShape for this layout
  using PartitionShape = PitchLinearShape<8, 4>;

  using PartitionCount =
      PitchLinearShape<TileShape::kContiguous / PartitionShape::kContiguous,
                       TileShape::kStrided / PartitionShape::kStrided>;

  using AccessCount =
      PitchLinearShape<PartitionShape::kContiguous, PartitionShape::kStrided>;

  //
  // Static constants
  //
  static int const kElementSize = 32;
  static int const kElementsPerAccess = kAccessSize / kElementSize;
  static int const kCrosswise = Crosswise;
  static int const kFactor = 1;

 private:
  //
  // Data members
  //

  /// Stride data member.
  Stride stride_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Index ldm = 0) : stride_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCongruous(Stride stride) : stride_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCongruous(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int tc = coord.contiguous() / 32;
    int ts = coord.strided() / 4;

    int c = (coord.contiguous() % 32) / kElementsPerAccess;
    int s = coord.strided() % 4;

    LongIndex offset = (c ^ (2 * s)) * kElementsPerAccess + s * stride_[0] +
                       tc * 32 + ts * stride_[0] * 4 + coord.contiguous() % 4;

    return offset;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return stride_; }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return stride_; }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return extent[1] * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCongruous(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());    
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicand
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCongruous {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCongruous<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

private:

  //
  // Data members
  //

  Base layout_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous(Index ldm = 0): layout_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCongruous(Stride stride): layout_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCongruous packed(TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCongruous(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return layout_.stride();
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return layout_.stride();
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear
/// memory and Crosswise size (in elements).
template <int ElementSize, int Crosswise>
struct TensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicand<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  static int const kCrosswise = Base::kCrosswise;
  static int const kFactor = Base::kFactor;
  using PartitionCount =  typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandCrosswise packed(TensorCoord const &extent) {
    return TensorOpMultiplicandCrosswise(extent[0]);
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(coord);
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return coord;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(extent);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a column-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct ColumnMajorTensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  using PartitionCount = typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  ColumnMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static ColumnMajorTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return ColumnMajorTensorOpMultiplicandCrosswise(extent.row());
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.row(), coord.column()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.contiguous(), coord.strided());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.row(), extent.column()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template mapping a row-major view of pitch-linear memory to
/// TensorOpMultiplicandCrosswise
template <int ElementSize, int Crosswise>
struct RowMajorTensorOpMultiplicandCrosswise {
  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = MatrixCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  using Base = TensorOpMultiplicandCrosswise<ElementSize, Crosswise>;

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = Base::kAccessSize;
  using TileShape = typename Base::TileShape;
  using PartitionShape = typename Base::PartitionShape;

  //
  // Static constants
  //

  static int const kElementSize = Base::kElementSize;
  static int const kElementsPerAccess = Base::kElementsPerAccess;
  using PartitionCount = typename Base::PartitionCount;
  using AccessCount = typename Base::AccessCount;

 private:
  //
  // Data members
  //

  Base layout_;

 public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise(Index ldm = 0) : layout_(ldm) {}

  /// Ctor
  CUTLASS_HOST_DEVICE
  RowMajorTensorOpMultiplicandCrosswise(Stride stride) : layout_(stride) {}

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static RowMajorTensorOpMultiplicandCrosswise packed(
      TensorCoord const &extent) {
    return RowMajorTensorOpMultiplicandCrosswise(extent.column());
  }

  /// Returns the offset of a coordinate in linear memory.
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    return layout_(PitchLinearCoord(coord.column(), coord.row()));
  }

  /// Inverse of layout function, mapping linear offset to logical coordinate
  CUTLASS_HOST_DEVICE
  TensorCoord inverse(LongIndex offset) const {
    PitchLinearCoord coord = layout_.inverse(offset);
    return MatrixCoord(coord.strided(), coord.contiguous());
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const { return layout_.stride(); }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride &stride() { return layout_.stride(); }

  /// Compute the number of contiguous elements needed to store a tensor with
  /// the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return layout_.capacity(PitchLinearCoord(extent.column(), extent.row()));
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandColumnMajorInterleaved {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;

  //static int const kThreadBlockStrided = ThreadBlockStrided;
  static int const kInterleavedK = InterleavedK;
  
private:

  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandColumnMajorInterleaved(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandColumnMajorInterleaved(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandColumnMajorInterleaved packed(TensorCoord const &extent) {
    return TensorOpMultiplicandColumnMajorInterleaved(extent[0] * kInterleavedK);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int const rows_per_smem_cache_line = 128 / kInterleavedK;

    int row_id = coord.strided() / rows_per_smem_cache_line;
    int col_id = (coord.strided() % rows_per_smem_cache_line) * kInterleavedK + coord.contiguous();

    int access_block_id = col_id >> 4;
    int swizzle_access_block_id = access_block_id ^ (row_id & 1);

    int swizzle_col_id = swizzle_access_block_id << 4;

    return row_id * 128 + swizzle_col_id;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return (extent[1] / kInterleavedK) * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

/// Template based on element size (in bits) - defined in terms of pitch-linear memory.
template <int ElementSize, int InterleavedK>
struct TensorOpMultiplicandRowMajorInterleaved {

  /// Logical rank of tensor
  static int const kRank = 2;

  /// Rank of stride vector
  static int const kStrideRank = 1;

  /// Index type used for coordinates
  using Index = int32_t;

  /// Long index type used for offsets
  using LongIndex = int64_t;

  /// Logical coordinate
  using TensorCoord = PitchLinearCoord;

  /// Stride vector
  using Stride = Coord<kStrideRank, Index, LongIndex>;

  //
  // Invariants
  //

  /// This layout is optimized for 128b accesses
  static int const kAccessSize = 128;

  //
  // Static constants
  //

  static int const kElementSize = ElementSize;
  static int const kElementsPerAccess = kAccessSize / kElementSize;

  //static int const kThreadBlockStrided = ThreadBlockStrided;
  static int const kInterleavedK = InterleavedK;
  
private:

  //
  // Data members
  //

  /// Stride data member
  Stride stride_;

public:
  //
  // Methods
  //

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandRowMajorInterleaved(Index ldm = 0): stride_(ldm) { }

  /// Ctor
  CUTLASS_HOST_DEVICE
  TensorOpMultiplicandRowMajorInterleaved(Stride stride): stride_(stride) { }

  /// Helper returns a layout to a tightly packed tensor
  CUTLASS_HOST_DEVICE
  static TensorOpMultiplicandRowMajorInterleaved packed(TensorCoord const &extent) {
    return TensorOpMultiplicandRowMajorInterleaved(extent[1] * kInterleavedK);
  }

  /// Returns the offset of a coordinate in linear memory. 
  /// Assumes coordinate has convention (contiguous, strided)
  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int const rows_per_smem_cache_line = 128 / kInterleavedK;

    int row_id = coord.strided() / rows_per_smem_cache_line;
    int col_id = (coord.strided() % rows_per_smem_cache_line) * kInterleavedK + coord.contiguous();

    int access_block_id = col_id >> 4;
    int swizzle_access_block_id = access_block_id ^ (row_id & 1);

    int swizzle_col_id = swizzle_access_block_id << 4;

    return row_id * 128 + swizzle_col_id;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride stride() const {
    return stride_;
  }

  /// Returns the stride of the layout
  CUTLASS_HOST_DEVICE
  Stride & stride() {
    return stride_;
  }

  /// Compute the number of contiguous elements needed to store a tensor with the given size
  CUTLASS_HOST_DEVICE
  LongIndex capacity(TensorCoord const &extent) const {
    return (extent[0] / kInterleavedK) * stride_[0];
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace layout
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
