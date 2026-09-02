/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//

//
#pragma once

#include "cutlass/conv/collective/builders/sm100_common.inl"
#include "cutlass/conv/collective/builders/sm90_gmma_builder.inl"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::conv::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  class ArchTag,
  conv::Operator ConvOp,
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNKL,    // (MmaAtomShapeM, MmaAtomShapeN, TileK, optional: TileL)
  class ClusterShape_MNK, // Static cluster shape or dynamic (int, int, _1)
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    ArchTag,
    arch::OpClassTensorOp,//固定类型不需要推导
    ConvOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNKL,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      (cute::is_same_v<ArchTag, arch::Sm100> 
      || cute::is_same_v<ArchTag, arch::Sm103>
      ) &&
      (cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized1SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelImplicitTmaWarpSpecialized2SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs1SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs2SmSm100> ||
       cute::is_same_v<KernelScheduleType, KernelScheduleAuto>) &&
      ((sizeof(ElementA) * AlignmentA) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0) &&
      ((sizeof(ElementB) * AlignmentB) % cutlass::gemm::collective::detail::tma_alignment_bytes == 0)>> {
      //只有条件成立时，enable_if_t<条件> 才是有效类型，特化才能参与匹配。
private:
  // For fprop, majorA = K,  major B = K;
  // For wgrad, majorA = MN, major B = MN;
  // For dgrad, majorA = K,  major B = MN;
  static constexpr cute::UMMA::Major UmmaMajorA =
    (ConvOp == conv::Operator::kWgrad) ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
  static constexpr cute::UMMA::Major UmmaMajorB =
    (ConvOp == conv::Operator::kFprop) ? cute::UMMA::Major::K : cute::UMMA::Major::MN;

  // For fp32 types, map to tf32 MMA value type
  using ElementAMma = cute::conditional_t<cute::is_same_v<ElementA, float>, tfloat32_t, ElementA>;//half
  using ElementBMma = cute::conditional_t<cute::is_same_v<ElementB, float>, tfloat32_t, ElementB>;//half

  using TileShape_MNK = decltype(cute::take<0,3>(TileShape_MNKL{})); // (MmaAtomShapeM, MmaAtomShapeN, TileK) Shape<_64, _64, Shape<_64>>
  //这里的第三项 Shape<_64> 不是普通的 _64，表示 K 维本身是一个嵌套 shape；take<0, 3> 只裁剪最外层，不会把嵌套的 Shape<_64> 展平
  static constexpr auto
  get_tiled_mma_schedule() {
    if constexpr (cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs1SmSm100>) {
      return KernelImplicitTmaWarpSpecialized1SmSm100{};
    }
    else if constexpr (cute::is_same_v<KernelScheduleType, KernelStridedDgradTmaWs2SmSm100>) {
      return KernelImplicitTmaWarpSpecialized2SmSm100{};
    }
    else {
      return KernelScheduleType{};
    }
  }

  using TiledMmaSchedule = decltype(get_tiled_mma_schedule());
  using TiledMma = decltype(detail::sm100_make_tiled_mma<ElementAMma, ElementBMma, ElementAccumulator,
                                                         TileShape_MNK, ClusterShape_MNK,
                                                         UmmaMajorA, UmmaMajorB, TiledMmaSchedule>());

  using AtomThrID = typename TiledMma::AtomThrID;

  // ((MMA_TILE_M,MMA_TILE_K), MMA_M, MMA_K)
  using MmaShapeA_MK = decltype(partition_shape_A(TiledMma{}, make_shape(cute::size<0>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));
  // ((MMA_TILE_N,MMA_TILE_K), MMA_N, MMA_K)
  using MmaShapeB_NK = decltype(partition_shape_B(TiledMma{}, make_shape(cute::size<1>(TileShape_MNK{}),
                                                                         cute::size<2>(TileShape_MNK{}))));

  static constexpr auto
  get_tma_atom_A() {
    if constexpr (cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs1SmSm100> ||
                  cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs2SmSm100>) {
      static_assert(ConvOp == conv::Operator::kDgrad, "Operator+Schedule mismatch");
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
    else {
      return cutlass::conv::collective::detail::sm100_cluster_shape_to_im2col_tma_atom_A(ClusterShape_MNK{}, AtomThrID{});
    }
  }

  static constexpr auto
  get_tma_atom_B() {
    if constexpr (cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs1SmSm100> ||
                  cute::is_same_v<KernelScheduleType,KernelStridedDgradTmaWs2SmSm100>) {
      static_assert(ConvOp == conv::Operator::kDgrad, "Operator+Schedule mismatch");
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
    else if constexpr (ConvOp == conv::Operator::kWgrad) {
      return cutlass::conv::collective::detail::sm100_cluster_shape_to_im2col_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
    else {
      return cutlass::gemm::collective::detail::sm100_cluster_shape_to_tma_atom_B(ClusterShape_MNK{}, AtomThrID{});
    }
  }

  // For wgrad kernel, tensor A uses tma tiled mode and tensor B uses tma im2col mode.
  using GmemTiledCopyA = decltype(get_tma_atom_A());//返回 TMA_copy_atom_op，后续需要make_tma_copy
  using GmemTiledCopyB = decltype(get_tma_atom_B());

  using BlockTileA_M = decltype(cute::size<0,0>(MmaShapeA_MK{}) * cute::size<1>(MmaShapeA_MK{}));
  using BlockTileA_K = decltype(cute::size<0,1>(MmaShapeA_MK{}) * cute::size<2>(MmaShapeA_MK{}));
  using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorA, ElementAMma, BlockTileA_M, BlockTileA_K>());//记录SMEM中A矩阵的布局，主要是为了后续的TMA_LOAD和UMMA操作

  using BlockTileB_N = decltype(cute::size<0,0>(MmaShapeB_NK{}) * cute::size<1>(MmaShapeB_NK{}));
  using BlockTileB_K = decltype(cute::size<0,1>(MmaShapeB_NK{}) * cute::size<2>(MmaShapeB_NK{}));
  using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
      UmmaMajorB, ElementBMma, BlockTileB_N, BlockTileB_K>());

  // 下面先统计 kernel 中除 mainloop A/B 多级缓冲和 epilogue tensor storage 之外的固定 SMEM 开销，
  // 再用架构可用 SMEM 减去这些开销，推导 mainloop 能容纳多少级 A/B tile。

  // UMMA warp（producer）与 epilogue warp（consumer）通过该 pipeline 交接 TMEM 中的累加器 tile。
  // 固定使用双缓冲，使 UMMA 可以生产下一块累加器，同时 epilogue 消费上一块。
  static constexpr uint32_t AccumulatorPipelineStageCount = 2;

  // CLC（Cluster Launch Control）调度器结果的 pipeline 深度；这里一次只缓存一个调度结果。
  static constexpr uint32_t SchedulerPipelineStageCount = 1;

  // 一个 tile scheduler CLCResponse 的大小（byte）；当前 SM100 implicit-GEMM scheduler 的响应为 16B。
  static constexpr uint32_t CLCResponseSize = 16;

  // 累加器 pipeline 每一级使用 full/empty barrier；这里只统计其 SharedStorage（不包含 TMEM 数据本身）。
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);

  // CLC fetch pipeline 的共享存储，每一级包含用于调度结果交接的 full/empty cluster barrier。
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);

  // 用于约束 mainloop load 与 epilogue load 两个 warp-group 执行顺序的 barrier 存储：
  // sequence depth = 1，length = 2。
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);

  // 保存 scheduler 返回的 CLCResponse；每个 scheduler pipeline stage 对应一个响应槽位。
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;

  // 释放 TMEM 前用于 cluster 内同步的 barrier。
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);

  // 保存 TMEM 分配结果（base pointer）；每个槽位使用一个 32-bit TMEM 地址。
  static constexpr auto TmemBasePtrsStorage = SchedulerPipelineStageCount * sizeof(uint32_t);

  // 汇总 kernel 级固定 SMEM 开销。CollectiveMainloop/CollectiveEpilogue 自己的 SharedStorage 不在这里重复计算。
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               TmemDeallocStorage +
                                                               CLCResponseStorage +
                                                               TmemBasePtrsStorage);

  // 从当前架构的 SMEM 总容量中扣除上述 kernel 固定开销，得到可供 collective 使用的容量。
  static constexpr int ReducedSmemCapacityBytes = detail::sm100_reduced_smem_capacity_bytes<ArchTag, KernelSmemCarveout>();

  // 一层 mainloop SMEM 缓冲所覆盖的逻辑 tile：(A 的 M, B 的 N, 公共 K)。
  // 单级容量由 A 的 M*K 个元素与 B 的 N*K 个元素组成；helper 使用 sizeof_bits_v 和
  // bits_to_bytes 计算实际字节数，因此也能正确处理 sub-byte 类型。
  using SmemTileShape = cute::Shape<BlockTileA_M, BlockTileB_N, BlockTileA_K>;

  // 决定 A/B mainloop pipeline 的 stage 数：
  //   * StageCountType 显式给出 stage 数时，直接使用该值；
  //   * 使用自动模式时，以 ReducedSmemCapacityBytes（还会扣除 StageCountType 指定的 carveout）
  //     除以每一级 A/B tile 加 pipeline barrier 的字节数，得到最多可容纳的 stage 数。
  static constexpr int PipelineStages = detail::compute_stage_count_or_override<
      ReducedSmemCapacityBytes, ElementAMma, ElementBMma, SmemTileShape>(StageCountType{});

  // 根据全局内存 layout tag 推导卷积空间维数：TensorNWC/NHWC/NDHWC 分别对应 1D/2D/3D。
  // helper 同时要求 A、B 使用相同的 layout tag。
  constexpr static int NumSpatialDimensions = detail::gmem_layout_tags_to_spatial_dims<GmemLayoutA, GmemLayoutB>();

  using DispatchPolicy = cutlass::conv::MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<
      ConvOp,
      PipelineStages,
      NumSpatialDimensions,
      SchedulerPipelineStageCount,
      AccumulatorPipelineStageCount,
      ClusterShape_MNK,
      ArchTag>;

public:
  using CollectiveOp = cutlass::conv::collective::CollectiveConv<
      DispatchPolicy,
      TileShape_MNKL,
      ElementA,
      ElementB,
      TiledMma,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyA, SmemLayoutAtomA>,
      detail::Sm100ImplicitGemmTileTraits<GmemTiledCopyB, SmemLayoutAtomB>
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::conv::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
