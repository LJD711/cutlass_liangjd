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
    \brief Defines iterators used by warp-level matrix multiply operations targeting Tensor Cores.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/arch/memory_sm75.h"
#include "cutlass/gemm/gemm.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/layout/pitch_linear.h"
#include "cutlass/layout/tensor_op_multiplicand_sm75.h"

#include "cutlass/platform/platform.h"
#include "cutlass/fast_math.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace warp {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Operand identity
    Operand Operand,
    /// Data type of A elements
    typename Element_,
    /// Layout of operand
    typename Layout_,
    /// Shape of one matrix production operation (concept: GemmShape)
    typename InstructionShape_,
    /// Delta between *MMA operations (in units of *MMA operations, concept:
    /// MatrixShape)
    int OpDelta_,
    /// Number of threads participating in one matrix operation
    int Threads,
    /// Number of partitions along K dimension
    int PartitionsK_ = 1>
class MmaTensorOpMultiplicandTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                                   64>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, 64>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner), 
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    stride_(ref.stride(0) / Layout::kElementsPerAccess),
    byte_offset_(0),
    k_group_idx_(0) {
      
    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ i);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx =
            (((quad_pair & 1) + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 1));
        access_contiguous_idx = ((quad_quad + ((i & 1) << 1)) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = ((lane_in_quad_pair >> 2) ^ (i >> 2));
        access_contiguous_idx = ((i & 3) ^ lane_in_quad);
        access_strided_idx = lane_id;
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr = 
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = 
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread MMA.TF32 NT TensorOps. It
/// uses LDS.32 to load from shared memory and therefore must be initialized
/// with a TensorRef to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<32, 32>, InstructionShape_,
    OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<32, 32>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual 32bit
    // shared memory load op.  Every one warp of 32bit shared memory load loads
    // 8x4 elements
    static int const kLdsOpInner = Layout::TileShape::kStrided;
    static int const kLdsOpOuter = kThreads / kLdsOpInner;

    static_assert(!(Shape::kContiguous % kLdsOpOuter),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsOpInner),
                  "Shape of warp-level mma must be divisible by 32bit "
                  "fundamental tile size.");

    /// Number of 32 bit shared memory load instructions needed by one MMA instruction
    /// 1688  A 2x2
    /// 1688  B 1x2
    /// 16816 B 1x4
    static int const LdsShapeContiguous =
        InstructionShape::kContiguous / kLdsOpOuter;
    static int const LdsShapeStrided = InstructionShape::kStrided / kLdsOpInner;
    using LdsShape =
        layout::PitchLinearShape<LdsShapeContiguous, LdsShapeStrided>;

    /// Number and arrangement of LDS instructions
    using LdsIterations = layout::PitchLinearShape<
        Shape::kContiguous / LdsShapeContiguous / kLdsOpOuter, 1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

 private:
  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount = Layout::TileShape::kContiguous *
                                   Layout::kElementsPerAccess /
                                   Policy::kLdsOpOuter;

  /// Vectorized access is not used
  static int const kElementsPerAccess = 1;

  /// Pointer type used for accesses
  using AccessType = Element;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

 private:
  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() : stride_(0), byte_offset_(0) {}

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : stride_(ref.stride(0)), byte_offset_(0), k_group_idx_(0) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int access_strided = lane_id % Policy::kLdsOpInner;
      int access_contiguous = (lane_id / Policy::kLdsOpInner) +
                              (access_strided ^ i) * Policy::kLdsOpOuter;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::TileShape::kContiguous * Layout::kElementsPerAccess / 2) {
      if (tile_offset.contiguous() % 2) {
        // Matrix multiply 1688 pointer_[0] <=> pointer_[4] pointer_[1] <=> pointer_[5]
        //           pointer_[2] <=> pointer_[6] pointer_[3] <=> pointer_[7]
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) * stride_ +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {
    Element *fetch_ptr = reinterpret_cast<Element *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsIterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsIterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int ss = 0; ss < Policy::LdsShape::kStrided; ++ss) {
          CUTLASS_PRAGMA_UNROLL
          for (int cc = 0; cc < Policy::LdsShape::kContiguous; ++cc) {
            int access_idx =
                cc + (ss + (c + s * Policy::LdsIterations::kContiguous) *
                               Policy::LdsShape::kStrided) *
                         Policy::LdsShape::kContiguous;
            int access_idx_contiguous = cc + c * Policy::LdsShape::kContiguous;
            int access_idx_strided =
                (ss + s * Policy::LdsShape::kStrided) * Policy::kLdsOpInner;

            AccessType const *source_ptr =
                pointer_[access_idx_contiguous % kPointerCount] +
                Layout::TileShape::kContiguous * Layout::kElementsPerAccess *
                    (access_idx_contiguous / kPointerCount) +
                access_idx_strided * stride_;

            char const *source_byte_ptr =
                reinterpret_cast<char const *>(source_ptr) + byte_offset +
                byte_offset_;

            fetch_ptr[access_idx] =
                *reinterpret_cast<Element const *>(source_byte_ptr);
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
        tile_offset.contiguous() * Shape::kContiguous /
            Layout::kElementsPerAccess +
        tile_offset.strided() * InstructionShape::kStrided * stride_;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps with 64B warp tile
/// the contiguous dimension. This assumes Threadblock contiguous dimension has
/// the same size as the warp tile.  It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// This specialization can be merged into the general one.  Most code is the same.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<16, 32>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = 32;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous / Layout::kFactor;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
    byte_offset_(0),
    k_group_idx_(0) {
      
    int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    //int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 4) {
        // Matrix multiply 1688 A/B
        // Q0 Q1 Q2 Q3 (Q stands for 1 8x128bit block).
        // Four blocks are next to each other in the contiguous dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = quad_pair ^ (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      } else if (Policy::LdsmShape::kContiguous == 2 &&
          kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (((quad_pair & 1) + i * 2) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_pair + (lane_id >> 4 << 3)) / 2;
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (quad_quad + i * 2) ^ (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor) ^ i;
        access_strided_idx = lane_id / Layout::kFactor;
      }

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess / Layout::kFactor +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr = 
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_ / Layout::kFactor;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset = 
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess + 
      tile_offset.strided() * InstructionShape::kStrided * stride_ / Layout::kFactor;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps with 32B warp tile
/// the contiguous dimension. This assumes Threadblock contiguous dimension has
/// the same size as the warp tile.  It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory.
///
/// This specialization can be merged into the general one.  Most code is the same.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCongruous<16, 16>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA || kOperand== Operand::kB,
    "MmaTensorOpMultiplicandIterator may only be instantiated for A or B operands to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = 16;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    static int const kLdsmOpInner = 8;

    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    static_assert(!(Shape::kStrided % kLdsmOpInner),
      "Shape of warp-level mma must be divisible by LDSM's fundamental tile size.");

    /// Shape of one individual LDSM instruction
    static int const LdsmShapeStrided =
        InstructionShape::kStrided / kLdsmOpInner;
    static int const LdsmShapeContiguous = 4 / LdsmShapeStrided;
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    using LdsmIterations = layout::PitchLinearShape<
        Shape::kContiguous / Layout::kElementsPerAccess / LdsmShapeContiguous,
        1>;

    /// Number of groups for each tile
    static int const kGroupsPerTile =
        Shape::kStrided / InstructionShape::kStrided;
  };

private:

  /// Not working on this feature at the moment.
  static_assert(kOpDelta == 1,
    "Alternative arrangements not supported at present.");

  /// Number of internal pointers needed to reference shared memory
  static int const kPointerCount =
      Layout::TileShape::kContiguous / Policy::LdsmShape::kContiguous / Layout::kFactor;

  /// Pointer type used for accesses
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

  /// Internal counter used to jump to next K partition
  int k_group_idx_;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
 using Fragment =
     Array<Element, Shape::kContiguous * InstructionShape::kStrided / kThreads>;

private:

  /// Layout object storing stride values
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  AccessType const *pointer_[kPointerCount];

  /// Byte offset incremented as iterator advances
  Index byte_offset_;

public:

  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(): stride_(0), byte_offset_(0) { }

  /// Constructor from TensorRef
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref,
    int lane_id
  ):
    stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
    byte_offset_(0),
    k_group_idx_(0) {

    //int quad_pair = (lane_id >> 3);
    int quad_quad = (lane_id >> 4);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kPointerCount; ++i) {
      int partition_contiguous_idx = -1;
      int access_contiguous_idx = -1;
      int access_strided_idx = -1;

      if (Policy::LdsmShape::kContiguous == 2 &&
          kOperand == Operand::kA) {
        // Matrix multiply 16816 A
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = lane_in_quad / 2;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * 2 + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      } else if (Policy::LdsmShape::kContiguous == 2 &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 B
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = lane_in_quad / 2;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * 2 + quad_quad) ^
             access_strided_idx);
      } else if (Policy::LdsmShape::kContiguous == 1) {
        // Matrix multiply 16832.SP B
        // Q0
        // Q1
        // Q2
        // Q3
        int factor_in_partition =
            (Layout::PartitionShape::kContiguous * Layout::kFactor /
             Layout::TileShape::kContiguous);

        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor) ^ i);
        access_strided_idx = lane_id / Layout::kFactor;
      } 

      int access_contiguous =
          partition_contiguous_idx * Layout::PartitionShape::kContiguous +
          access_contiguous_idx;

      int access_strided = access_strided_idx;

      pointer_[i] = reinterpret_cast<AccessType const *>(ref.data()) +
                    access_contiguous + access_strided * stride_;
    }
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    byte_offset_ += offset * sizeof(Element);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    int contiguous_offset = tile_offset.contiguous();
    if (Shape::kContiguous ==
        Layout::PartitionShape::kContiguous * Layout::kElementsPerAccess) {
      if (tile_offset.contiguous() % 2) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kPointerCount / 2; ++i) {
          AccessType const *tmp_pointer = pointer_[i];
          pointer_[i] = pointer_[i + kPointerCount / 2];
          pointer_[i + kPointerCount / 2] = tmp_pointer;
        }
      }
      contiguous_offset = (tile_offset.contiguous() >> 1) << 1;
    }

    int offset = (tile_offset.strided() * InstructionShape::kStrided) *
                     stride_ * Layout::kElementsPerAccess / Layout::kFactor +
                 contiguous_offset * Shape::kContiguous;

    add_pointer_offset(offset);

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    add_tile_offset({0, 1});

    if (kPartitionsK > 1) {
      ++k_group_idx_;
      // Jump to next stage
      if (k_group_idx_ == Policy::kGroupsPerTile) {
        k_group_idx_ = 0;
        add_tile_offset(
            {0, ((kPartitionsK - 1) * Policy::kGroupsPerTile)});
      }
    }

    return *this;
  }

  /// Advances the iterator along the opposite of the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {
    byte_offset_ -= stride_ * InstructionShape::kStrided * sizeof(Element) *
                    Layout::kElementsPerAccess;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    load_with_byte_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      Index byte_offset) const {

    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
      reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {

      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {

        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        AccessType const *source_ptr =
            pointer_[c % kPointerCount] +
            Layout::TileShape::kContiguous * (c / kPointerCount) +
            Policy::kLdsmOpInner * Policy::LdsmShape::kStrided * s * stride_ / Layout::kFactor;

        char const *source_byte_ptr = reinterpret_cast<char const *>(source_ptr) + byte_offset + byte_offset_;

        cutlass::arch::ldsm<layout::ColumnMajor, Policy::LdsmShape::kCount>(
          fetch_ptr[access_idx],
          source_byte_ptr
        );
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    Index pointer_offset =
      tile_offset.contiguous() * Shape::kContiguous / Layout::kElementsPerAccess +
      tile_offset.strided() * InstructionShape::kStrided * stride_ / Layout::kFactor;

    byte_offset += sizeof(AccessType) * pointer_offset;

    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // no op
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Congruous may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// MBlock or NBlock size
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.contiguous(), tile_offset.strided()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to load from shared
/// memory and therefore must be initialized with a TensorRef to shared memory. 
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:

  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for RowMajor Congruous may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCongruous<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCongruous<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

private:

  /// Underlying tile iterator
  Base iterator_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(
    TensorRef const &ref, 
    int lane_id
  ): iterator_({ref.data(), ref.stride()}, lane_id) {
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {

    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator++() {

    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator--() {

    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {

    iterator_.load(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
      frag,
      {tile_offset.strided(), tile_offset.contiguous()},
      byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
// 这个偏特化是 warp 从 TensorOpMultiplicandCrosswise shared-memory layout
// 读取 A/B fragment 的底层 pitch-linear iterator。它不负责 global->shared
// copy；它负责：
//
//   swizzled shared memory
//        -> ldmatrix/ldsm
//        -> 每个 lane 的寄存器 Fragment
//
// 当前讨论的 SM80 F16 Conv2d Fprop 实例中，外层 matrix-layout wrapper 会把
// MatrixShape 转成这里的 PitchLinearShape。A/B 的具体实例分别为：
//
//   A:
//     Shape_            = PitchLinearShape<64, 64>
//     Operand_          = Operand::kA
//     Element_          = half_t
//     Layout            = TensorOpMultiplicandCrosswise<16, 64>
//     InstructionShape_ = PitchLinearShape<16, 16>
//     OpDelta_          = 1
//     PartitionsK_      = 1
//
//   B:
//     Shape_            = PitchLinearShape<64, 64>
//     Operand_          = Operand::kB
//     Element_          = half_t
//     Layout            = TensorOpMultiplicandCrosswise<16, 64>
//     InstructionShape_ = PitchLinearShape<16, 8>
//     OpDelta_          = 1
//     PartitionsK_      = 1
//
// 对 A/B 而言，contiguous 都是 GEMM K 方向；strided 分别是 M/N 方向。
template <
    /// Size of the matrix to load (concept: PitchLinearShape)
    // warp iterator 一次逻辑 load 覆盖的完整 pitch-linear tile。
    // 当前 A/B 都是 <contiguous=64, strided=64>。
    typename Shape_,
    /// Identifies A or B multiplicand
    // 标识这是 A 还是 B。后面的 lane 地址公式会因 operand 不同选择不同分支。
    Operand Operand_,
    /// Data type of elements
    // shared memory 中元素类型。当前 Element_=half_t，sizeof_bits=16。
    typename Element_,
    /// Shape of one matrix product operation (concept: PitchLinearShape)
    // 单条 mma.sync 指令对应的 pitch-linear operand 形状。
    // 当前 A=<K=16,M=16>，B=<K=16,N=8>。
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    // 相邻 MMA instruction tile 的间隔。当前 DefaultMmaTensorOp policy 给出 1。
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    // crosswise K-block 的标量元素数。当前为 64 个 half，即 128 bytes。
    int Crosswise,
    /// Number of partitions along K dimension
    // warp-level split-K partition 数。当前 WarpCount::kK=1。
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    // 精确匹配 32-thread TensorOp + pitch-linear crosswise layout 的偏特化。
    Shape_, Operand_, Element_,
    cutlass::layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                                   Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {//liangjd
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  // 当前 A/B: Shape::kContiguous=64, Shape::kStrided=64。
  using Shape = Shape_;

  /// Operand tag
  // 编译期常量。当前分别为 Operand::kA 或 Operand::kB。
  static Operand const kOperand = Operand_;

  // 本偏特化只服务 MMA 的 A/B multiplicand；不能实例化为 accumulator/C iterator。
  static_assert(kOperand == Operand::kA || kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator may only be instantiated for "
                "A or B operands to warp-level Mma.");

  /// Element type
  // 当前 Element=half_t，一个元素 16 bit。
  using Element = Element_;

  /// Element number when the layout crosses
  // 当前 kCrosswise=64，表示 layout 以 K=64 个元素为一个 crosswise section。
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  // 当前 Layout=TensorOpMultiplicandCrosswise<16,64>。
  // 该 layout 以 128-bit vector 为基本访问单位，并对 vector 地址做 XOR
  // swizzle，以便后续 ldmatrix 更少产生 shared-memory bank conflict。
  using Layout = cutlass::layout::TensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: GemmShape)
  // 当前 A=<16,16>；B=<16,8>。这个类后面的地址公式主要使用
  // InstructionShape::kContiguous，当前 A/B 都等于 K=16。
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  // 当前 kOpDelta=1，即相邻 MMA instruction tile 紧邻排列。
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  // ldmatrix 和 mma.sync 都由完整 warp 的 32 个 lane 协同执行。
  static int const kThreads = 32;

  /// Number of partitions along K dimension
  // 当前 kPartitionsK=1，因此一个 warp 独占其 WarpShape::kK=64。
  static int const kPartitionsK = PartitionsK_;

  /// TensorRef type for loading element from a tensor
  // TensorRef 保存 shared-memory 基地址以及 crosswise layout/stride。
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  // 当前为 int32_t，用于 byte offset、pointer offset 等较小索引。
  using Index = typename TensorRef::Index;

  /// Long Index type
  // 当前为 int64_t，用于通用的大范围元素偏移接口。
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  // Layout::Stride 的标量类型。当前为 int32_t。
  // 注意 stride_ 后面保存的是 AccessType 数量，不是 byte 数。
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  // 当前为 PitchLinearCoord(contiguous, strided)。
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    // warp tile 的 contiguous/K=64 必须能被 instruction contiguous/K=16 整除。
    // 当前 64 % 16 == 0。
    static_assert(
        !(Shape::kContiguous % InstructionShape::kContiguous),
        "Shape of warp-level Mma must be divisible by operator shape.");

    // Determine number of elements along outer dimension per individual LDSM op
    // Layout::kElementsPerAccess = 128 bits / 16 bits = 8 half。
    // 一个 AccessType/shared-memory vector 正好是 8 half = 16 bytes。
    static int const kLdsmOpOuter = Layout::kElementsPerAccess;
    // ldmatrix 的基本 strided 高度固定按 8 行组织。
    static int const kLdsmOpInner = 8;

    // 当前 64 % 8 == 0，warp tile 的 K 维可以完整分成 128-bit vectors。
    static_assert(!(Shape::kContiguous % kLdsmOpOuter),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    // 当前 64 % 8 == 0，warp tile 的 M/N 维可以完整分成 8-row blocks。
    static_assert(!(Shape::kStrided % kLdsmOpInner),
                  "Shape of warp-level mma must be divisible by LDSM's "
                  "fundamental tile size.");

    /// Shape of one individual LDSM instruction
    // 单个 MMA operand 的 contiguous/K=16 含两个 128-bit vectors：
    //   LdsmShapeContiguous = 16 / 8 = 2。
    static int const LdsmShapeContiguous =
        InstructionShape::kContiguous / kLdsmOpOuter;
    // 当前：
    //   4 / LdsmShapeContiguous * 8 = 4 / 2 * 8 = 16
    //   Shape::kStrided = 64
    //   16 > 64 为 false
    // 所以 LdsmShapeStrided = 4 / 2 = 2。
    //
    // 得到 LdsmShape=<2,2>，kCount=4，对应一次 ldmatrix.x4/ldsm<...,4>。
    static int const LdsmShapeStrided =
        ((4 / LdsmShapeContiguous * kLdsmOpInner) > Shape::kStrided)
            ? (Shape::kStrided / kLdsmOpInner)
            : (4 / LdsmShapeContiguous);
    using LdsmShape =
        layout::PitchLinearShape<LdsmShapeContiguous, LdsmShapeStrided>;

    /// Number and arrangement of LDSM instructions
    // 当前：
    //   contiguous iterations = 1
    //   strided iterations    = 64 / 8 / 2 = 4
    // 所以每个 lane 为一个 K-group 执行 4 次 ldmatrix.x4。
    // 每次 ldmatrix.x4 给每个 lane 4 个 uint32=16 bytes，四次共 64 bytes，
    // 正好是 Fragment 的 32 half。
    using LdsmIterations =
        layout::PitchLinearShape<1, Shape::kStrided / kLdsmOpInner /
                                        LdsmShape::kStrided>;

    /// Number of K-groups inside one crosswise tile.
    // Layout::TileShape::kContiguous=8（单位是 128-bit vectors），
    // Layout::kFactor=1，LdsmShape::kContiguous=2，因此：
    //
    //   kGroupsPerTile = 8 / 1 / 2 = 4。
    //
    // 当前这 4 组就是 K[0..15]、K[16..31]、K[32..47]、K[48..63]。
    static int const kGroupsPerTile = Layout::TileShape::kContiguous /
                                      Layout::kFactor / LdsmShape::kContiguous;
  };

 private:
  /// Not working on this feature at the moment.
  // 当前实现的 XOR 地址推进只支持紧邻 MMA tile，因此要求 kOpDelta==1。
  static_assert(kOpDelta == 1,
                "Alternative arrangements not supported at present.");

  /// Pointer type used for accesses
  // 当前 AccessType=Array<half_t,8>，大小 16 bytes。
  // pointer_ 的 +1 单位是一个 128-bit vector，而不是一个 half 或一个 byte。
  using AccessType = Array<Element, Layout::kElementsPerAccess>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  // 当前 A/B：
  //
  //   Fragment elements = 64 * 16 / 32 = 32 half
  //   Fragment bytes    = 32 * 2       = 64 bytes
  //
  // 32 个 lane 合计 32*32=1024 half，正好覆盖一个 64x16 的 warp K-group。
  using Fragment = Array<Element, Shape::kStrided *
                                      InstructionShape::kContiguous / kThreads>;

 private:

  /// Total number of sections.  The memory is divided into stages.  One stage
  /// can store one tile.  Stage is divided into sections.  Interleaved layout
  /// can have multiple sections in a stage.  The rest layout only has one section
  /// in a stage.
  // sections_ = ref.stride(0) / kCrosswise。
  // 当前 shared storage 有 Stages=3，packed stride=K*Stages=64*3=192，
  // 所以 sections_=192/64=3，恰好对应 stage0/stage1/stage2。
  int sections_;

  /// Layout object storing stride values
  // stride_ 的单位是 AccessType（当前每个 AccessType=16 bytes）。
  // 当前 stride_=192 * factor1 / elements_per_access8 = 24 AccessType。
  // 换算成字节为 24*16=384 bytes，等价于逻辑 192 half。
  StrideIndex stride_;

  /// Shared memory base pointers - not advanced
  // 指向 shared-memory buffer 的固定基地址。普通 K-group 迭代主要修改
  // byte_offset_；跨 section/stage 或 strided tile 时才移动 pointer_。
  AccessType const *pointer_;

  /// Byte offset incremented as iterator advances
  // lane 私有的动态字节偏移，包含 lane 初始地址和 K-group XOR swizzle。
  Index byte_offset_;

  /// Internal counter used to determine when to increment byte offset and when
  /// to XOR it
  // 当前逻辑 K-group 编号。当前范围为 [0,4)，到 4 后回到 0 并跨到下一 tile。
  int k_group_idx_;

 public:
  /// Default ctor constructs null iterator
  // 空 iterator 只用于满足类型/容器构造需求；在调用 load() 前必须用有效
  // shared-memory TensorRef 重新构造。这里将所有地址状态清零。
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator()
      : pointer_(nullptr),  // 没有绑定 shared-memory 基地址
        sections_(0),      // 尚不知道 shared buffer 中有多少 section/stage
        stride_(0),        // 尚未计算以 AccessType 为单位的 stride
        byte_offset_(0),   // lane 动态字节偏移归零
        k_group_idx_(0) {} // 从逻辑 K-group0 开始

  /// Constructor from TensorRef
  // ref 必须指向 shared memory；lane_id 范围为 [0,31]。
  // 构造函数的任务是给每个 lane 计算其第一条 ldmatrix 地址。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      // ref.data() 是 Element*，这里转成 AccessType*=Array<half,8>*。
      // 所以 pointer_+1 当前前进 8 half = 16 bytes。
      : pointer_(reinterpret_cast<AccessType const *>(ref.data())),
        // 一个 crosswise section 含 kCrosswise 个标量元素。
        // 当前 ref.stride(0)=192 half，kCrosswise=64，因此 sections_=3。
        sections_(ref.stride(0) / kCrosswise),
        // stride_ = kCrosswise x sections_ x kFactor
        // 将逻辑 element stride 转成 AccessType stride：
        //   192 elements * factor1 / 8 elements-per-access = 24 AccessType。
        stride_(ref.stride(0) * Layout::kFactor / Layout::kElementsPerAccess),
        // 构造时尚未加入 lane 地址和 K-group swizzle。
        byte_offset_(0),
        // 初始逻辑位置为 K-group0。
        k_group_idx_(0) {
    // Warp level iterator at most use double buffer to hide latency.  If there
    // are more than 2 sections, every stage should have more than 1 section.
    // 当前 sections_=3，来自三个 shared-memory pipeline stages。pointer_ 保持
    // buffer 基址，section/stage 的切换由 pointer_、byte_offset_ 共同完成。

    // Turing silicon requires all 32 threads in a warp provide valid addresses
    // even for LDSM.1 and LDSM.2
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 750))
    // 对 SM75，ldmatrix.x1/x2 只有部分 lane 的地址在语义上有用，但硬件仍要求
    // 全部 32 lane 给出有效地址。取模让多余 lane 复用有效 lane 的地址。
    // 当前目标是 SM80，因此该预处理不会编译进当前 device 路径。
    lane_id = lane_id % (Policy::LdsmShape::kCount * Policy::kLdsmOpInner);
#endif

    // 下列变量是同一个 lane_id 的不同粒度分解，用位运算避免除法/取模：
    //
    //   quad_quad          = lane / 16，范围 0..1
    //   quad_pair          = lane / 8， 范围 0..3
    //   lane_in_pair       = lane % 2， 范围 0..1
    //   lane_in_quad       = lane % 4， 范围 0..3
    //   lane_in_quad_pair  = lane % 8， 范围 0..7
    //   lane_in_quad_quad  = lane % 16，范围 0..15
    //
    // 不同 Tensor Core instruction/layout factor 使用不同组合，得到无 bank
    // conflict 的 ldmatrix lane 地址。
    int quad_quad = (lane_id >> 4);
    int quad_pair = (lane_id >> 3);
    int lane_in_pair = (lane_id & 1);
    int lane_in_quad = (lane_id & 3);
    int lane_in_quad_pair = (lane_id & 7);
    int lane_in_quad_quad = (lane_id & 15);

    // 三个坐标先置为 -1，随后必须由与 Layout::kFactor 匹配的合法分支赋值：
    //
    //   partition_contiguous_idx: contiguous 方向第几个 4-vector partition
    //   access_contiguous_idx:    partition 内第几个 128-bit vector
    //   access_strided_idx:       strided 方向对应的 lane row
    int partition_contiguous_idx = -1;
    int access_contiguous_idx = -1;
    int access_strided_idx = -1;

    if (Layout::kFactor == 8) {
      // kFactor=8 常见于更小 crosswise block/interleaved 数据类型。
      // 当前 PartitionShape::kContiguous=4、TileShape::kContiguous=8 时：
      //   factor_in_partition = 4 * 8 / 8 = 4。
      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // ldmatrix fragment 完全沿 strided 方向展开时：
        // lane%8 决定 partition，lane%4 与 lane/factor 的 XOR 决定 vector，
        // lane/factor 决定 strided row。
        partition_contiguous_idx = lane_in_quad_pair / factor_in_partition;
        access_contiguous_idx = ((lane_in_quad) ^ (lane_id / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
    } else if (Layout::kFactor == 4) {
      // Super Integer matrix multiply Interleaved-32
      // 当前通用常量下 factor_in_partition = 4 * 4 / 8 = 2。
      int factor_in_partition =
          (Layout::PartitionShape::kContiguous * Layout::kFactor /
           Layout::TileShape::kContiguous);

      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Integer matrix multiply 8816 A/B：ldsm shape 全部沿 strided 展开。
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_contiguous_idx = ((lane_in_pair * factor_in_partition) ^
                                 (lane_in_quad_quad / Layout::kFactor));
        access_strided_idx = lane_id / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Integer matrix multiply 16832 A：A/B 的 lane-to-row 关系不同，
        // 所以 operand tag 会参与编译期分支。
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_quad / Layout::kFactor;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + quad_quad) ^
             access_strided_idx);
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Integer matrix multiply 16832 B：高 16 lanes 通过 quad_quad*2
        // 选择另外两组 strided rows。
        partition_contiguous_idx = lane_in_quad / factor_in_partition;
        access_strided_idx = lane_in_quad_pair / Layout::kFactor + quad_quad * 2;
        access_contiguous_idx =
            ((lane_in_pair * factor_in_partition + ((lane_id & 8) >> 3)) ^
             access_strided_idx);
      }
    } else if (Layout::kFactor == 2) {
      // Super Matrix multiply kBlock = 32
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // Matrix multiply 1688 A/B。
        // 每个 partition 由 lane%2 选择；lane%8/2 选择 contiguous vector；
        // lane/2 选择 16 个 strided rows。
        // (Q stands for 1 8x128bit block).
        // Q0
        // Q1
        // Q2
        // Q3
        // Four blocks are next to each other in the strided dimension.
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx = (lane_in_quad_pair / Layout::kFactor);
        access_strided_idx = lane_id / Layout::kFactor;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // Matrix multiply 16816 | 1688.TF32 A。
        // quad_quad 与 partition 内 vector 做 XOR，匹配 A 的 row-major view。
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_quad ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = (lane_in_quad_quad / Layout::kFactor);
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // Matrix multiply 16816 | 1688.TF32 B。
        // B 使用 column-major view，因此 quad_pair 的最低位参与 XOR。
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            ((quad_pair & 1) ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx =
            (lane_in_quad_pair + (lane_id >> 4 << 3)) / Layout::kFactor;
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B：ldsm fragment 完全沿 contiguous 展开。
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_id % Layout::kFactor);
        access_contiguous_idx =
            (quad_pair ^ (lane_in_quad_pair / Layout::kFactor));
        access_strided_idx = lane_in_quad_pair / Layout::kFactor;
      }
    } else if (Layout::kFactor == 1) {
      // 当前 SM80 F16 Crosswise<16,64> 正好进入本分支：
      //   TileShape=<8,8>, kElementsPerAccess=8, kCrosswise=64
      //   kFactor=8*8/64=1。
      // Super Matrix multiply kBlock = 64
      if (Policy::LdsmShape::kStrided == Policy::LdsmShape::kCount) {
        // 适用于 ldsm shape 完全沿 strided 展开的其它 instruction。
        // 当前 LdsmShape=<2,2>、kCount=4，2!=4，因此 F16 m16n8k16
        // 不进入这个子分支。
        // Q0
        // Q1
        // Q2
        // Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = lane_in_quad;
        access_strided_idx = lane_id;
      }
      else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kA) {
        // 当前 A 进入这里，因为 LdsmShape::kStrided=2、kCount/2=2。
        //
        //   partition = (lane % 8) / 4，范围 0..1
        //   vector    = (lane / 16) XOR (lane % 4)，范围 0..3
        //   row       = lane % 16，范围 0..15
        //
        // Q0 Q2
        // Q1 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_quad ^ lane_in_quad);
        access_strided_idx = lane_in_quad_quad;
      } else if (Policy::LdsmShape::kStrided ==
                     (Policy::LdsmShape::kCount / 2) &&
                 kOperand == Operand::kB) {
        // 当前 B 进入这里：
        //
        //   partition = (lane % 8) / 4，范围 0..1
        //   vector    = ((lane / 8) & 1) XOR (lane % 4)
        //   row       = lane%8 + (lane/16)*8，范围 0..15
        //
        // Q0 Q1
        // Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = ((quad_pair & 1) ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair + (lane_id >> 4 << 3);
      }
      else if (Policy::LdsmShape::kContiguous == Policy::LdsmShape::kCount) {
        // Matrix multiply 16832.SP B：当前 F16 m16n8k16 不进入该分支。
        // Q0 Q1 Q2 Q3
        partition_contiguous_idx = (lane_in_quad_pair >> 2);
        access_contiguous_idx = (quad_pair ^ lane_in_quad);
        access_strided_idx = lane_in_quad_pair;
      }
    }

    // PartitionShape::kContiguous=4。把 partition 编号和 partition 内 vector
    // 编号合成 TileShape contiguous vector 编号。当前结果范围为 0..7。
    int access_contiguous =
        partition_contiguous_idx * Layout::PartitionShape::kContiguous +
        access_contiguous_idx;

    // 当前 A/B 的 strided row 都映射到范围 0..15。
    int access_strided = access_strided_idx;

    // access_contiguous/access_strided 的单位都是 128-bit AccessType vectors。
    // 当前 stride_=24 AccessType，sizeof_bits<Element>*elements/8=16 bytes：
    //
    //   byte_offset_ = (access_contiguous + access_strided*24) * 16 bytes。
    //
    // 部分 lane 的当前实例结果：
    //
    //   lane   A(access_c,row,byte)     B(access_c,row,byte)
    //     0      (0, 0,    0)             (0, 0,    0)
    //     8      (0, 8, 3072)             (1, 0,   16)
    //    16      (1, 0,   16)             (0, 8, 3072)
    //    31      (6,15, 5856)             (6,15, 5856)
    //
    // 这些地址是 ldmatrix 每个 lane 提供的 shared-memory 起点，不表示每个
    // lane 独立读取完整矩阵；32 lane 会协同完成矩阵 fragment 装载。
    byte_offset_ = (access_contiguous + access_strided * stride_) *
                   sizeof_bits<Element>::value * Layout::kElementsPerAccess / 8;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  // offset 的单位是 Element，不是 AccessType，也不是 byte。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    // 当前 Element=half，所以一个 element offset 转成 2 bytes。
    // 这里只修改 lane 私有 byte_offset_，shared 基址 pointer_ 不变。
    byte_offset_ += offset * sizeof_bits<Element>::value / 8;

    // 返回自身引用，允许 iterator.add_pointer_offset(x).load(...) 式链式调用。
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  // tile_offset 的两个分量不是标量元素坐标：
  //
  //   contiguous: 以一个 MMA K-group/InstructionShape::kContiguous 为单位
  //   strided:    以一个完整 warp tile 的 Shape::kStrided 为单位
  //
  // 当前一个 contiguous tile=K16，一个 strided tile=M/N64。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    // 一个 crosswise tile 当前含 kGroupsPerTile=4 个 K-group。
    // whole_tiles 是跨过的完整 K64/crosswise blocks 数。
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    // k_groups_delta 是完整 block 内剩余的 K16 group 数，当前范围通常 0..3。
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;

    // K-group 在 crosswise layout 中通过 XOR 选择物理地址，而不是简单相加。
    // 当前每个 delta 的 XOR 基本步长为：
    //
    //   16 bits * 8 elements/access * 2 contiguous-vectors / 8
    //   = 32 bytes。
    //
    // 所以 k_groups_delta=1/2/3 分别 XOR 32/64/96 bytes。
    byte_offset_ ^= k_groups_delta * sizeof_bits<Element>::value *
                    Layout::kElementsPerAccess *
                    Policy::LdsmShape::kContiguous / 8;
    // pointer_ 的单位是 AccessType。当前：
    //
    //   strided tile +1:
    //     1 * stride24 * ShapeStrided64 / factor1
    //     = 1536 AccessType = 24576 bytes
    //     = 64 rows * 192 half/row * 2 bytes
    //
    //   whole contiguous tile +1:
    //     stride24 / sections3 = 8 AccessType = 128 bytes
    //     = 一个 K64 half block。
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  // 这个版本专门正确处理负 contiguous offset。C++ 整数除法向 0 截断，
  // 因此先把负余数规范化到 [0,kGroupsPerTile)，再修正 whole_tiles。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {

    // 先按普通除法拆成完整 K64 blocks 和残余 K16 groups。
    int whole_tiles = tile_offset.contiguous() / Policy::kGroupsPerTile;
    int k_groups_delta = tile_offset.contiguous() % Policy::kGroupsPerTile;
    if (k_groups_delta < 0) {
      // 例如当前 -1/4 得到 whole=0、remainder=-1；规范化成：
      //   whole=-1, remainder=3
      // 即“退一个 K64 block，再前进到该 block 的 group3”。
      whole_tiles -= 1;
      k_groups_delta += Policy::kGroupsPerTile;
    }

    if ((Policy::kGroupsPerTile / kPartitionsK) >= 2) {
      // 有至少 2 组时，delta 的 bit0 控制第一级 32-byte XOR。
      // 当前 groups/partition=4，因此此分支生效。
      byte_offset_ ^= (k_groups_delta & 1) * Policy::LdsmShape::kContiguous *
                      sizeof_bits<Element>::value *
                      Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) >= 4) {
      // 有至少 4 组时，delta 的 bit1 控制第二级 64-byte XOR。
      // 加入当前 k_group_idx_ 的低 bit，是为了从任意当前 K-group 相对移动
      // 时仍得到正确 swizzled 地址。当前本分支也生效。
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 1)) & 2) * 
                      Policy::LdsmShape::kContiguous *
                      sizeof_bits<Element>::value *
                      Layout::kElementsPerAccess / 8;
    }
    if ((Policy::kGroupsPerTile / kPartitionsK) == 8) {
      // 只有 8 K-groups/tile 时才需要第三级 128-byte XOR。
      // 当前 groups=4，因此编译器会删除本分支。
      byte_offset_ ^= ((k_groups_delta + (k_group_idx_ & 3)) & 4) * 
                      Policy::LdsmShape::kContiguous *
                      sizeof_bits<Element>::value *
                      Layout::kElementsPerAccess / 8;
    }

    // 更新逻辑 K-group。它可能暂时超过当前 tile 的 group 数。
    k_group_idx_ += k_groups_delta;
    // 超出的 group 数折算成额外 whole tile 数。
    whole_tiles += k_group_idx_ / (Policy::kGroupsPerTile / kPartitionsK);
    // 将组号重新约束到当前 partition 的合法范围。当前是 [0,4)。
    k_group_idx_ = k_group_idx_ % (Policy::kGroupsPerTile / kPartitionsK);

    // 与正向版本一样，strided 方向按完整 warp tile 移动；contiguous
    // whole_tiles 按完整 crosswise K block 移动。
    pointer_ +=
        tile_offset.strided() * stride_ * Shape::kStrided / Layout::kFactor +
        whole_tiles * stride_ / sections_;
    return *this;
  }

  /// Advances the iterator along the advance dimension
  // 前进一个 MMA K-group。当前即 K16：group0->1->2->3->下一 K64 tile。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {

    // 下面列出的 ^1/^3/^7 或 ^2/^6 是不同 kblock/instruction 下的
    // crosswise XOR 序列。数字乘以各自基本 byte step 后修改 byte_offset_。
    // Integer matrix multiply 16832 Interleaved-32
    //   NONE
    // Integer matrix multiply 16816 Interleaved-32 || Integer matrix multiply 16816 kblock=32

    // Integer matrix multiply 8816  Interleaved-32
    //   ^1 ^1
    // Matrix multiply 1684.TF32 kblock=16 || Integer matrix multiply 16816 kblock=64
    // Matrix multiply 1688 kblock=32 || Integer matrix multiply 8816 kblock=64
    //   ^1 ^3 ^1 ^3
    // Matrix multiply 1688 kblock=64
    //   ^1 ^3 ^1 ^7 ^1 ^3 ^1 ^7

    // Matrix multiply 16816 kblock=32 | 1688.TF32 kblock=16 || Integer matrix multiply 16832 kblock=64
    //   ^2 ^2
    // Matrix multiply 16816 kblock=64 | 1688.TF32 kblock=32 || Integer matrix multiply 16832 kblock=128
    //   ^2 ^6 ^2 ^6

    if ((Policy::kGroupsPerTile / kPartitionsK) > 1) {
      // 当前 groups/partition=4，所以 mask=1。
      // 若 groups=8，mask=3；若 groups=2，mask=0。
      int mask = ((Policy::kGroupsPerTile / kPartitionsK) == 8)
                     ? 3
                     : (((Policy::kGroupsPerTile / kPartitionsK) == 4) ? 1 : 0);

      // 当前 groups=4、基本步长=32 bytes，K-group 地址序列为：
      //
      //   group0 -> group1: XOR 1*32 = 32 bytes
      //   group1 -> group2: XOR 3*32 = 96 bytes
      //   group2 -> group3: XOR 1*32 = 32 bytes
      //   group3 -> wrap:   XOR 3*32 = 96 bytes
      //
      // 相对初始 lane byte offset，四组物理 XOR 状态为：
      //   base, base^32, base^64, base^96。
      if (((k_group_idx_ & mask) % 2) == 0)
        byte_offset_ ^= 1 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 1)
        byte_offset_ ^= 3 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
      else if ((k_group_idx_ & mask) == 3)
        byte_offset_ ^= 7 * Policy::LdsmShape::kContiguous *
                        sizeof_bits<Element>::value *
                        Layout::kElementsPerAccess / 8;
    }

    // 逻辑 K-group 前进一步。
    k_group_idx_++;

    if (k_group_idx_ == (Policy::kGroupsPerTile / kPartitionsK)) {
      // 当前从 group3 前进后 k_group_idx_=4，于是回绕到 group0。
      k_group_idx_ = 0;
      // XOR 序列已经回到本 tile 的基础状态；再跨过 4 个 K16 groups，
      // 即 pointer_ 前进一个完整 K64/crosswise block（当前 128 bytes）。
      add_tile_offset({Policy::kGroupsPerTile, 0});
    }

    return *this;
  }

  /// Advances the iterator along the advance dimension
  // 该偏特化没有实现逐组 operator--。调用会触发 assert；需要负向移动时
  // 应使用 add_tile_offset_negative()。
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() { assert(0); }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  // 复合赋值只是 add_tile_offset() 的语法包装。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  // 负 offset 通过坐标取负后复用 add_tile_offset()。当 contiguous offset
  // 不是 kGroupsPerTile 的整数倍时，优先使用专门的 negative 版本更直观。
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  // 无额外偏移的主 load 接口，最终进入下面的 byte-offset 版本。
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { load_with_byte_offset(frag, 0); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      // 当前 frag=Array<half,32>，总大小 64 bytes/lane。
      Fragment &frag,
      /// loads a tile with a linear offset in units of bytes
      // 调用者额外提供的 byte offset；会与 lane 自身 byte_offset_ 相加。
      Index byte_offset) const {
    // 当前 LdsmShape::kCount=4，所以 fetch_ptr 指向 4-uint32=16-byte chunks。
    // 64-byte Fragment 可视为 fetch_ptr[0..3] 四个 chunk。
    Array<unsigned, Policy::LdsmShape::kCount> *fetch_ptr =
        reinterpret_cast<Array<unsigned, Policy::LdsmShape::kCount> *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    // 当前 LdsmIterations::kStrided=4，因此 s=0..3。
    for (int s = 0; s < Policy::LdsmIterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      // 当前 LdsmIterations::kContiguous=1，因此 c 只有 0。
      for (int c = 0; c < Policy::LdsmIterations::kContiguous; ++c) {
        // 当前 access_idx=s，依次填充 Fragment 的四个 16-byte chunks。
        int access_idx = c + s * Policy::LdsmIterations::kContiguous;

        // source_ptr 单位是 AccessType。当前 c=0、factor=1、
        // kLdsmOpInner=8、LdsmShapeStrided=2、stride=24：
        //
        //   source_ptr = pointer_ + 8 * 2 * s * 24
        //              = pointer_ + 384*s AccessType
        //              = pointer_ + 6144*s bytes。
        //
        // 6144 bytes = 16 logical rows * 192 half/row * 2 bytes，
        // 所以四次 s 迭代分别读取 strided rows 0、16、32、48 对应的片段。
        AccessType const *source_ptr =
            pointer_ + Policy::LdsmShape::kContiguous * c +
            Policy::kLdsmOpInner / Layout::kFactor *
                Policy::LdsmShape::kStrided * s * stride_;

        // 将固定 AccessType 基址转成 byte pointer，再叠加：
        //   1. 调用者给出的 byte_offset
        //   2. lane 初始地址和当前 K-group swizzle 的 byte_offset_。
        char const *source_byte_ptr =
            reinterpret_cast<char const *>(source_ptr) + byte_offset +
            byte_offset_;

        // 32 个 lane 协作执行 shared-memory matrix load。
        // 当前 count=4，对应 ldmatrix.x4：每个 lane 得到 4 个 uint32=16 bytes。
        // 外层 s 循环执行四次，最终每 lane 装入 64 bytes=32 half。
        cutlass::arch::ldsm<layout::RowMajor, Policy::LdsmShape::kCount>(
            fetch_ptr[access_idx], source_byte_ptr);
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      // pointer_offset 的单位是 Element。当前 half_t 每个元素 2 bytes。
      Index pointer_offset) const {
    load_with_byte_offset(frag, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      // tile_offset 是额外逻辑 tile 偏移，不会修改 iterator 内部状态。
      TensorCoord const &tile_offset) const {
    load_with_byte_offset(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      // pointer_offset 仍以 Element 为单位，先转换成 bytes。
      Index pointer_offset) const {
    load_with_byte_offset(frag, tile_offset, pointer_offset * sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      // 初始额外 byte offset，后面再加 tile_offset 对应的 bytes。
      Index byte_offset) const {
    // pointer_offset 的单位是 AccessType：
    //
    //   contiguous +1:
    //     InstructionContiguous16 / ElementsPerAccess8
    //     = 2 AccessType = 32 bytes
    //
    //   strided +1:
    //     ShapeStrided64 * stride24
    //     = 1536 AccessType = 24576 bytes。
    Index pointer_offset = tile_offset.contiguous() *
                               InstructionShape::kContiguous /
                               Layout::kElementsPerAccess +
                           tile_offset.strided() * Shape::kStrided * stride_;

    // 当前 sizeof_bits<AccessType>=128，所以每个 pointer_offset 单位是16 bytes。
    byte_offset += sizeof_bits<AccessType>::value * pointer_offset / 8;

    // 复用真正执行 ldmatrix 的 byte-offset load，不改变 iterator 本身。
    load_with_byte_offset(frag, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    // 当前 modulus = kGroupsPerTile4 / PartitionsK1 = 4。
    // 例如传入 0/1/2/3 保持不变，传入 4 回到 0。
    //
    // 重要：这里只同步逻辑计数器，不修改 pointer_ 或 byte_offset_，也不会
    // 执行 load。调用者必须保证当前地址状态本来就对应所声明的 K-group。
    k_group_idx_ = k_group % (Policy::kGroupsPerTile / kPartitionsK);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kB,
                "MmaTensorOpMultiplicandIterator for ColumnMajor Crosswise may "
                "only be instantiated for B operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// KBlock size
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kRow,
                               InstructionShape::kColumn>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.row(), tile_offset.column()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.row(), tile_offset.column()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.contiguous(), tile_offset.strided()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It uses LDSM to
/// load from shared memory and therefore must be initialized with a TensorRef
/// to shared memory.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Identifies A or B multiplicand
    Operand Operand_,
    /// Data type of elements
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions)
    int OpDelta_,
    /// Element number when the layout crosses (in units of elements)
    int Crosswise,
    /// Number of partitions along K dimension
    int PartitionsK_>
class MmaTensorOpMultiplicandTileIterator<
    Shape_, Operand_, Element_,
    cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
        sizeof_bits<Element_>::value, Crosswise>,
    InstructionShape_, OpDelta_, 32, PartitionsK_> {
 public:
  /// Shape of tile to load (concept: PitchLinearShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand_;

  static_assert(kOperand == Operand::kA,
                "MmaTensorOpMultiplicandIterator for RowMajor Crosswise may "
                "only be instantiated for A operand to warp-level Mma.");

  /// Element type
  using Element = Element_;

  /// Element number when the layout crosses
  static int const kCrosswise = Crosswise;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<Element_>::value, kCrosswise>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept:
  /// MatrixShape)
  static int const kOpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Underlying tile iterator implementation
  using Base = MmaTensorOpMultiplicandTileIterator<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, kOperand, Element,
      layout::TensorOpMultiplicandCrosswise<sizeof_bits<Element_>::value,
                                            kCrosswise>,
      layout::PitchLinearShape<InstructionShape::kColumn,
                               InstructionShape::kRow>,
      kOpDelta, kThreads, PartitionsK_>;

 public:
  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = typename Base::Fragment;

 private:
  /// Underlying tile iterator
  Base iterator_;

 public:
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator() {}

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator(TensorRef const &ref, int lane_id)
      : iterator_({ref.data(), ref.stride()}, lane_id) {}

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_pointer_offset(LongIndex offset) {
    iterator_.add_pointer_offset(offset);

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &add_tile_offset_negative(
      TensorCoord const &tile_offset) {
    iterator_.add_tile_offset_negative({tile_offset.column(), tile_offset.row()});

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator++() {
    ++iterator_;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator--() {
    --iterator_;

    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator+=(
      TensorCoord const &tile_offset) {
    add_tile_offset(PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of
  ///< the tensor
  CUTLASS_DEVICE
  MmaTensorOpMultiplicandTileIterator &operator-=(
      TensorCoord const &tile_offset) {
    add_tile_offset(-PitchLinearCoord(tile_offset.column(), tile_offset.row()));
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const { iterator_.load(frag); }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index pointer_offset) const {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a linear offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    assert(0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load_with_byte_offset(
      /// fragment to load from the tensor
      Fragment &frag,
      /// loads a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// loads a tile with a logical offset AND a pointer offset
      Index byte_offset) const {
    iterator_.load_with_byte_offset(
        frag, {tile_offset.strided(), tile_offset.contiguous()}, byte_offset);
  }

  /// Notify the iterator which k-group it is currently pointing to.
  ///
  /// This does not advance the iterator. Rather, it overrides its internal
  /// tracking with constant-valued k-group index to enable the compiler to
  /// fold constants and achieve more efficient code.
  ///
  /// This is used by some nontrivial permuted layouts.
  CUTLASS_DEVICE
  void set_kgroup_index(int k_group) {
    iterator_.set_kgroup_index(k_group); 
  }
};

////////////////////////////////////////////////////////////////////////////////

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Layout of operand in memory
    typename Layout_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator;

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, cutlass::layout::RowMajor, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout.
///
/// This iterator is not tested.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, cutlass::layout::AffineRankN<2>, InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::RowMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible =
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<
    Element, 
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;

            frag[mma_accum_start + row * kElementsPerAccess + col] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///
template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element type
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_>
class MmaTensorOpAccumulatorTileIterator<Shape_, Element_,
                                         cutlass::layout::ColumnMajor,
                                         InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajor;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static bool const kDivisible = 
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN);

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<
      (Shape::kRow + InstructionShape::kM - 1) / InstructionShape::kM,
      (Shape::kColumn + InstructionShape::kN - 1) / InstructionShape::kN
    >;
  };

private:

  // Assume accumulator tile is an arrangement of 8-by-8 tiles replicated over the entire
  // shape, with each quad mapped to one row and each thread mapped to 1/4 of the elements
  // of that row. The accumulators within one row are assumed to be consecutive.
 static int const kElementsPerAccess = InstructionShape::kN / 4;
 static int const kRowsPerTile = 8;
 static int const kAccumulatorRows = InstructionShape::kM / kRowsPerTile;

public:

  //
  // Derived quantities
  //

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, 
    Policy::MmaIterations::kCount * InstructionShape::kMN / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;

            frag[idx] = offset_ref.at({accum_m, accum_n});
          }
        }
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        
        int mma_accum_start = kAccumulatorRows * kElementsPerAccess * 
          (mma_n * Policy::MmaIterations::kRow + mma_m);

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < kAccumulatorRows; ++row) {
          CUTLASS_PRAGMA_UNROLL
          for (int col = 0; col < kElementsPerAccess; ++col) {
            int accum_m = mma_m * InstructionShape::kM * OpDelta::kRow +
                          row * kRowsPerTile;
            int accum_n = mma_n * InstructionShape::kN * OpDelta::kColumn + col;
            int idx = mma_accum_start + row * kElementsPerAccess + col;
            
            offset_ref.at({accum_m, accum_n}) = frag[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element typ
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Interleaved N
    int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, cutlass::layout::ColumnMajorInterleaved<InterleavedN>,
    InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = Element_;

  /// Layout of source tile
  using Layout = cutlass::layout::ColumnMajorInterleaved<InterleavedN>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");

    static_assert(platform::is_same<TensorCoord, MatrixCoord>::value,
      "Layouts must be defined for logical MatrixCoord coordinate space.");

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / InstructionShape::kM,
                                      Shape::kColumn / InstructionShape::kN>;
  };

private:

  static int const kElementsPerAccess = 2;

public:

  //
  // Derived quantities
  //

  using AccessType = Array<Element, kElementsPerAccess>;

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<Element, Shape::kCount / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref, 
    int lane_id
  ):
    ref_(ref) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    MatrixCoord lane_offset(quad, lane_in_quad * kElementsPerAccess);

    ref_.add_coord_offset(lane_offset);
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(TensorCoord const &tile_offset) {

    ref_.add_coord_offset(tile_offset * make_Coord(Shape::kRow, Shape::kColumn));

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag, 0);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType* frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
          offset_ref.offset(TensorCoord(accum_m, accum_n)));

        frag_ptr[idx] = access_ptr[0];
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kRow;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                 offset_ref.offset(TensorCoord(accum_m, accum_n)));

        access_ptr[0] = frag_ptr[idx];               
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

/// This tile iterator is specialized for 32-thread TensorOps. It is used to load or store
/// accumulators from memory and is agnostic to layout. It could be faster if it assumed row-major
/// accumulator layout.
///
/// Satisfies:
///   ReadableRandomAccessContiguousTileIteratorConcept |
///   WriteableRandomAccessContiguousTileIteratorConcept
///

template <
    /// Size of the matrix to load (concept: MatrixShape)
    typename Shape_,
    /// Element typ
    typename Element_,
    /// Shape of one matrix product operation (concept: MatrixShape)
    typename InstructionShape_,
    /// Interval between adjacent *MMA instructions (in units of MMA
    /// instructions, concept: MatrixShape)
    typename OpDelta_,
    /// Interleaved N
    int InterleavedN>
class MmaTensorOpAccumulatorTileIterator<
    Shape_, Element_, cutlass::layout::TensorNCxHWx<InterleavedN>,
    InstructionShape_, OpDelta_> {
 public:

  /// Shape of tile to load (concept: MatrixShape)
  using Shape = Shape_;

  /// Operand tag
  static Operand const kOperand = Operand::kC;

  /// Element type
  using Element = int8_t;

  /// Layout of source tile
  using Layout = cutlass::layout::TensorNCxHWx<InterleavedN>;

  /// Shape of one matrix product operation (concept: MatrixShape)
  using InstructionShape = InstructionShape_;

  /// Delta between *MMA operations (in units of *MMA operations, concept: MatrixShape)
  using OpDelta = OpDelta_;

  /// Number of participating threads
  static int const kThreads = 32;

  /// TensorRef type for loading element from a tensor
  using TensorRef = TensorRef<Element, Layout>;

  /// Index type
  using Index = typename TensorRef::Index;

  /// Long Index type
  using LongIndex = typename TensorRef::LongIndex;

  /// Long Index type
  using StrideIndex = typename TensorRef::Layout::Stride::Index;

  /// Coordinate for an element in the tensor
  using TensorCoord = typename TensorRef::TensorCoord;

  /// Internal structure of iterator - made public to enable introspection
  struct Policy {
    static_assert(
        !(Shape::kRow % InstructionShape::kM) &&
            !(Shape::kColumn % InstructionShape::kN),
        "Shape of warp-level Mma must be divisible by operator shape.");

    /// Number of elements in strided dimension that each STG writes
    static int const kStridedPerSTG = 8;

    /// Factor to calculate reorder index to pack accumulator.
    static int const kPackedFactor = Shape::kColumn / 32;

    /// Number of mma operations performed
    using MmaIterations = MatrixShape<Shape::kRow / kStridedPerSTG,
                                      Shape::kColumn / InterleavedN>;
  };

private:

  static int const kElementsPerAccess = InterleavedN / 4;

public:

  //
  // Derived quantities
  //

  struct alignas((kElementsPerAccess * sizeof_bits<Element>::value / 8)) AccessType {
      Array<Element, kElementsPerAccess> storage;
  };

  /// Fragment object holding a thread's part of a tile
  using Fragment = Array<int32_t, Shape::kCount / kThreads>;

private:

  /// Reference to output tensor
  TensorRef ref_;

  /// Row offset index globally
  LongIndex global_offset_row_;

  /// Column offset index globally
  LongIndex global_offset_col_;

  /// Output tensor size
  TensorCoord extent_;

  /// Alpha 
  float alpha_;

  /// Beta
  float beta_;

public:
  
  /// Default ctor constructs null iterator
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator() { }

  /// Constructor from TensorRef
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator(
    TensorRef const &ref,
    int const lane_id,
    TensorCoord extent,
    float alpha = 1.0f,
    float beta = 0.0f
  ):
    ref_(ref),
    extent_(extent),
    alpha_(alpha),
    beta_(beta) {

    int quad = (lane_id >> 2);
    int lane_in_quad = (lane_id & 3);

    global_offset_row_ = quad;

    global_offset_col_ = lane_in_quad * kElementsPerAccess;
  }

  /// Adds a pointer offset to internal pointer(s) to advance through memory
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_pointer_offset(LongIndex offset) {
    ref_.add_pointer_offset(offset);
    return *this;
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole tiles
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator &add_tile_offset(MatrixCoord const &tile_offset) {

    global_offset_row_ += tile_offset.row() * Shape::kRow;

    global_offset_col_ += tile_offset.column() * Shape::kColumn;

    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator++() {
    // deliberate no-op
    return *this;
  }

  /// Advances the iterator along the advance dimension
  CUTLASS_HOST_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator--() {
    // deliberate no-op
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator+=(TensorCoord const &tile_offset) {
    add_tile_offset(tile_offset);
    return *this;
  }

  ///< advances in units of whole tiles along the logical coordinate space of the tensor
  CUTLASS_DEVICE
  MmaTensorOpAccumulatorTileIterator & operator-=(TensorCoord const &tile_offset) {
    add_tile_offset(-tile_offset);
    return *this;
  }

  /// Loads a fragment from memory at the location pointed to by the iterator.
  CUTLASS_HOST_DEVICE
  void load(Fragment &frag) const {
    load_with_pointer_offset(frag);
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_pointer_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index pointer_offset) const {               ///< loads a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    AccessType* frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int mma_n = 0; mma_n < Policy::MmaIterations::kN; ++mma_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kM; ++mma_m) {
        int accum_m = mma_m * InstructionShape::kM;
        int accum_n = mma_n * InstructionShape::kN;

        int idx = mma_m + mma_n * Policy::MmaIterations::kM;

        AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                 accum_m * offset_ref.stride(0) + accum_n);

        frag_ptr[idx] = access_ptr[0];
      }
    }
  }

  /// Loads a fragment from memory with additional logical offset
  CUTLASS_DEVICE
  void load_with_byte_offset(
    Fragment &frag,                             ///< fragment to load from the tensor
    Index byte_offset) const {                  ///< loads a tile with a linear offset

    load_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset) const {     ///< loads a tile with a logical offset in units of whole tiles

    load(frag, tile_offset, 0);
  }

  /// Loads a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void load(
    Fragment &frag,                             ///< fragment to load from the tensor
    TensorCoord const &tile_offset,             ///< loads a tile with a logical offset in units of whole tiles
    Index pointer_offset) const {               ///< loads a tile with a logical offset AND a pointer offset

    load_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }

  /// Stores a fragment to memory
  CUTLASS_HOST_DEVICE
  void store(Fragment const &frag) const {
    store_with_pointer_offset(frag, 0);
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_pointer_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index pointer_offset) const {               ///< store a tile with a linear offset
  
    TensorRef offset_ref(ref_);
    offset_ref.add_pointer_offset(pointer_offset);

    Array<float, Shape::kCount / kThreads> output_frag_f;
    Array<Element, Shape::kCount / kThreads> output_frag;

    LongIndex pq = extent_.h() * extent_.w();

    LongIndex extent_row = extent_.n() * pq;
    LongIndex extent_col = extent_.c();

    LongIndex k_major = (global_offset_col_ / InterleavedN) * pq;
    Index k_minor = global_offset_col_ % InterleavedN;
    LongIndex k_offset = k_major * InterleavedN + k_minor;
    LongIndex k_offset_delta = pq * InterleavedN;

    LongIndex stride_n = pq * extent_.c();

    Index n;
    LongIndex pq_rem;

    unsigned int pq_mul, pq_shr;
    find_divisor(pq_mul, pq_shr, pq);

    if(beta_ == 0.0f) {
      CUTLASS_PRAGMA_UNROLL
      for(int i = 0; i < int(frag.size()); ++i) {
        output_frag_f[i] = frag[i];
      }

      if(InstructionShape::kM == Policy::kStridedPerSTG) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < int(frag.size()); ++i) {
          output_frag[i] = (Element)(output_frag_f[i] * alpha_);
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < int(frag.size()); ++i) {
          int map_i = (i / (16 * Policy::kPackedFactor)) * (16 * Policy::kPackedFactor)
                    + (i % (8 * Policy::kPackedFactor)) / 2 * 4
                    + (i % (8 * Policy::kPackedFactor)) % 2
                    + (i / (8 * Policy::kPackedFactor)) % 2 * 2;
          output_frag[i] = (Element)(output_frag_f[map_i] * alpha_);
        }
      }

      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
       
          int accum_n = mma_n * InterleavedN;

          int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;
         
          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            access_ptr[0] = frag_ptr[idx];
          }
        }
      }
    } else {
      if(InstructionShape::kM == Policy::kStridedPerSTG) {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < int(frag.size()); ++i) {
          output_frag_f[i] = frag[i];
        }
      } else {
        CUTLASS_PRAGMA_UNROLL
        for(int i = 0; i < int(frag.size()); ++i) {
          int map_i = (i / (16 * Policy::kPackedFactor)) * (16 * Policy::kPackedFactor)
                    + (i % (8 * Policy::kPackedFactor)) / 2 * 4
                    + (i % (8 * Policy::kPackedFactor)) % 2
                    + (i / (8 * Policy::kPackedFactor)) % 2 * 2;
          output_frag_f[i] = frag[map_i];
        }
      }

      AccessType const *frag_ptr = reinterpret_cast<AccessType const*>(&output_frag);

      Array<Element, kElementsPerAccess> ref_frag;
      AccessType *ref_frag_ptr = reinterpret_cast<AccessType *>(&ref_frag);

      CUTLASS_PRAGMA_UNROLL
      for (int mma_m = 0; mma_m < Policy::MmaIterations::kRow; ++mma_m) {
        int accum_m = mma_m * Policy::kStridedPerSTG;

        fast_divmod(n, pq_rem, global_offset_row_ + accum_m, pq, pq_mul, pq_shr);
        LongIndex offset_m = n * stride_n + k_offset + pq_rem * InterleavedN;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_n = 0; mma_n < Policy::MmaIterations::kColumn; ++mma_n) {
       
          int accum_n = mma_n * InterleavedN;

          int idx = mma_n + mma_m * Policy::MmaIterations::kColumn;
         
          if((global_offset_row_ + accum_m < extent_row) && (global_offset_col_ + accum_n < extent_col)) {
            AccessType* access_ptr = reinterpret_cast<AccessType *>(offset_ref.data() +
                                                                    offset_m + mma_n * k_offset_delta);

            ref_frag_ptr[0] = access_ptr[0];

            CUTLASS_PRAGMA_UNROLL
            for(int i = 0; i < kElementsPerAccess; ++i) {
              output_frag[idx * kElementsPerAccess + i] = Element(alpha_ * output_frag_f[idx * kElementsPerAccess + i]
                                                                + beta_ * ref_frag[i]);
            }

            access_ptr[0] = frag_ptr[idx];
          }
        }
      }
    }
  }

  /// Stores a fragment to memory with additional pointer offset
  CUTLASS_DEVICE
  void store_with_byte_offset(
    Fragment const &frag,                       ///< fragment to store from the tensor
    Index byte_offset) const {                  ///< store a tile with a linear offset

    store_with_pointer_offset(byte_offset / sizeof(Element));
  }

  /// Stores a fragment to memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
    Fragment &frag,                             ///< fragment to store to the tensor
    TensorCoord const &tile_offset) const {     ///< stores a tile with a logical offset in units of whole tiles

    store(frag, tile_offset, 0);
  }

  /// Stores a fragment from memory with logical offset in units of whole tiles.
  CUTLASS_DEVICE
  void store(
      /// fragment to store to the tensor
      Fragment const &frag,
      /// stores a tile with a logical offset in units of whole tiles
      TensorCoord const &tile_offset,
      /// stores a tile with a logical offset AND a pointer offset
      Index pointer_offset) const {
    store_with_pointer_offset(frag, ref_.offset(tile_offset) + pointer_offset);
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace warp
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
