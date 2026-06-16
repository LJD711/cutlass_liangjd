/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


#pragma once

namespace cutlass::gemm::collective::detail {

// ============================================================================
// Sm100DenseGemmTmaUmmaCarveout
// ============================================================================
// 作用：计算 SM100 Dense GEMM kernel 中，除 A/B data buffer 和 Epilogue 之外
//      所有辅助数据结构（pipeline barrier、scheduler、TMEM 管理等）占用的 SMEM。
//
// 背景：
//   - SM100 每个 SM 的 SMEM 总容量 = 232,448 bytes (≈227 KiB)
//   - 这些容量需要分配给：
//     (1) Mainloop A/B staging buffer（主要部分，由 PipelineStages 决定）
//     (2) Epilogue 的输入/输出 buffer
//     (3) 本结构体统计的各种辅助开销（称为 "Carveout"）
//   - compute_stage_count_or_override() 中使用的 "Effective SMEM" =
//     总容量(232448) - KernelSmemCarveout - EpilogueSharedStorage
//
// 三种 "Stage" 概念的区分：
//   - AccumulatorPipelineStageCount：UMMA accumulator pipeline 深度（TMEM 双缓冲），
//     通常是 2，控制 MMA warpgroup 对 TMEM accumulator 的读写同步。
//   - SchedulerPipelineStageCount：Tile Scheduler 通信队列深度，通常是 1-3，
//     控制 scheduler 向 kernel 推送工作 tile 的并发度。
//   - PipelineStages（Mainloop TMA data stage）：由 SMEM 容量动态计算，
//     表示 A/B shared memory 中同时缓存的 Ktile 数量。
// ============================================================================

template<
  class ClusterShape_MNK,           // Cluster 形状，如 Shape<_2,_1,_1> 表示 2×1×1 的 2SM cluster
  int AccumulatorPipelineStageCount,// UMMA accumulator pipeline stage 数（通常是 2，TMEM 双缓冲）
  int SchedulerPipelineStageCount,  // Tile scheduler pipeline stage 数（通常是 1-3）
  int CLCResponseSize,              // CLC (Cluster Launch Control) response 大小（通常是 16 B）
  bool IsArrayOfPointersGemm,       // 是否为 pointer-array / grouped GEMM
  int NumTensorMaps=2               // 每个 tensor 的 TMA descriptor 数量（grouped gemm 时用）
>
struct Sm100DenseGemmTmaUmmaCarveout {

  // ==========================================================================
  // 1. AccumulatorPipelineStorage
  // ==========================================================================
  // 类型：PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage
  // 作用：UMMA (Tensor Core) accumulator pipeline 的 shared memory。
  //       用于 MMA warpgroup 与 TMEM (Tensor Memory) 之间的 producer-consumer 同步。
  //       UMMA 运算的 accumulator 存储在 TMEM 中，需要 pipeline 来管理读写时序。
  // 大小估算：
  //   PipelineUmmaAsync<N> 内部使用 PipelineAsync<N>，其 SharedStorage 包含：
  //     full_barrier_[N]  (ClusterBarrier, 8B each)
  //     empty_barrier_[N] (ClusterBarrier, 8B each)
  //   sizeof = 2 × N × 8 = 16N bytes
  //   当 AccumulatorPipelineStageCount = 2 时：32 B
  // ==========================================================================
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);

  // ==========================================================================
  // 2. NumCLCResponses / CLCPipelineStorage
  // ==========================================================================
  // 作用：CLC (Cluster Launch Control) Tile Scheduler 的 fetch pipeline。
  //       Producer (scheduler) 向 consumer (TMA/MMA/Epilogue warps) 分发工作 tile。
  //       通过 full/empty barrier 实现 request/response 的同步。
  // NumCLCResponses：
  //   - 普通 dense GEMM：1 份 response（scheduler 直接和 kernel 通信）
  //   - ArrayOfPointersGemm / Grouped GEMM：2 份 response
  //     （一份给 TMA updater warps，一份给 MMA/Epilogue warps）
  // 大小估算：
  //   PipelineCLCFetchAsync<N> 的 SharedStorage 包含：
  //     full_barrier_[N]  (ClusterTransactionBarrier, 8B)
  //     empty_barrier_[N] (ClusterBarrier, 8B)
  //   sizeof = 2 × N × 8 = 16N bytes
  //   当 SchedulerPipelineStageCount = 3, NumCLCResponses = 1 时：48 B
  //   当 SchedulerPipelineStageCount = 1, NumCLCResponses = 1 时：16 B
  // ==========================================================================
  static constexpr int NumCLCResponses = (IsArrayOfPointersGemm ? 2 : 1);
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage) * NumCLCResponses;

  // ==========================================================================
  // 3. LoadOrderBarrierStorage
  // ==========================================================================
  // 类型：OrderedSequenceBarrier<1,2>::SharedStorage
  // 作用：控制 A 和 B 两个 tensor 的 TMA load 执行顺序。
  //       SequenceDepth=1 表示一个 sequence，SequenceLength=2 表示 A 和 B 两步。
  //       确保 A tile 的 TMA load 在 B tile 之前完成（或反之，取决于配置），
  //       防止 A/B 数据竞争或乱序到达导致的同步问题。
  // 大小估算：
  //   barrier_[1][2] = 2 × ClusterBarrier(8B) = 16 B
  // ==========================================================================
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);

  // ==========================================================================
  // 4. CLCResponseStorage
  // ==========================================================================
  // 作用：Tile scheduler 返回给 kernel 的 response buffer。
  //       每个 response 包含一个工作 tile 的信息（tile 坐标、边界 predicate、
  //       K-slice 偏移等），kernel 根据这些信息执行对应的 MMA 运算。
  // SchedulerPipelineStageCount：
  //   决定同时 in-flight 的 scheduler request 数量。
  //   Stage=3 表示 kernel 可以同时持有 3 个未完成的 tile 请求。
  // CLCResponseSize：
  //   定义在 sm100_common.inl 中，SM100 的 CLCResponse = { uint32_t data[4] } = 16 B。
  // NumCLCResponses：
  //   同上文，普通 GEMM 为 1，Grouped GEMM 为 2。
  // 大小估算：SchedulerPipelineStageCount × 16 × NumCLCResponses
  //   Stage=3, NumCLCResponses=1 时：48 B
  //   Stage=1, NumCLCResponses=1 时：16 B
  // ==========================================================================
  static constexpr auto CLCResponseStorage = SchedulerPipelineStageCount * detail::CLCResponseSize * NumCLCResponses;

  // ==========================================================================
  // 5. CLCThrottlePipelineStorage
  // ==========================================================================
  // 类型：PipelineAsync<SchedulerPipelineStageCount>::SharedStorage
  // 作用：Scheduler throttle control pipeline，用于控制 scheduler 向 kernel
  //       推送工作的速率。防止 scheduler 过快地分发 tile 导致 kernel 的
  //       shared memory 或 register pressure 被压垮。
  // 大小估算：
  //   PipelineAsync<N> 的 SharedStorage 包含：
  //     full_barrier_[N]  (ClusterBarrier, 8B)
  //     empty_barrier_[N] (ClusterBarrier, 8B)
  //   sizeof = 2 × N × 8 = 16N bytes
  //   Stage=3 时：48 B；Stage=1 时：16 B
  // ==========================================================================
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);

  // ==========================================================================
  // 6. TmemDeallocStorage
  // ==========================================================================
  // 类型：sizeof(cutlass::arch::ClusterBarrier)
  // 作用：Tensor Memory deallocation 的同步 barrier。
  //       UMMA 运算完成后，accumulator 数据需要从 TMEM 写回 SMEM/global memory。
  //       这个 barrier 用于同步 cluster 内所有 CTA，确保 TMEM 数据已被消费
  //       且可以安全 dealloc，避免数据被提前覆盖。
  // 大小估算：sizeof(ClusterBarrier) = sizeof(uint64_t) = 8 B
  // ==========================================================================
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);

  // ==========================================================================
  // 7. TmemBasePtrsStorage
  // ==========================================================================
  // 作用：每个 accumulator pipeline stage 对应的 TMEM base pointer。
  //       UMMA 的 accumulator 存储在 TMEM (Tensor Memory) 中，每个 stage
  //       需要记录其 TMEM 区域的基地址，供 tcgen05.mma 指令引用。
  // 大小估算：SchedulerPipelineStageCount × sizeof(uint32_t) = Stage × 4 bytes
  //   Stage=3 时：12 B；Stage=1 时：4 B
  // ==========================================================================
  static constexpr auto TmemBasePtrsStorage = SchedulerPipelineStageCount * sizeof(uint32_t);

  // ==========================================================================
  // 8. TensorMapStorage
  // ==========================================================================
  // 作用：仅 ArrayOfPointersGemm / Grouped GEMM 使用。
  //       普通 GEMM 的 TMA descriptor 在 host 端创建后直接传给 kernel，
  //       通过 constant memory 或 params 传入，不占 SMEM。
  //       但 Grouped GEMM 中每个 sub-problem 的 global address 不同，
  //       需要在 device 端动态更新 TMA descriptor，因此 descriptor
  //       需要存放在 SMEM 中供 TMA 指令引用。
  // 大小估算：
  //   cute::TmaDescriptor = 64 B（CUDA TMA descriptor 标准大小）
  //   NumTensorMaps 默认 = 2（A 和 B 两个 tensor）
  //   乘以 5：CUTLASS 预分配了 5 个 tensormap 变体（支持不同 layout/stride）
  //   64 × 2 × 5 = 640 B（仅 grouped GEMM 时）
  // ==========================================================================
  static constexpr auto TensorMapStorage =
    IsArrayOfPointersGemm ? sizeof(cute::TmaDescriptor) * NumTensorMaps * 5 /* We have five tensormaps smem */ :
    0;

  // ==========================================================================
  // 9. TensorMapReadyPipelineStorage
  // ==========================================================================
  // 作用：仅 ArrayOfPointersGemm / Grouped GEMM 使用。
  //       用于同步 tensormap 在 SMEM 中已更新完毕、可供 TMA load 指令使用。
  //       Producer (tensormap updater warp) 更新 descriptor 后 arrive，
  //       Consumer (TMA load warp) 等待 ready 后开始发射 TMA。
  // 大小估算：
  //   PipelineAsync<SchedulerPipelineStageCount>
  //   Stage=3 时：48 B；Stage=1 时：16 B
  // ==========================================================================
  static constexpr auto TensorMapReadyPipelineStorage =
    IsArrayOfPointersGemm ? sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage) :
    0;

  // ==========================================================================
  // KernelSmemCarveout: 上述所有辅助开销的总和
  // ==========================================================================
  // 作用：从 SMEM 总容量中扣除这些开销后，剩余部分才是可用于
  //       Mainloop A/B staging buffer 和 Epilogue buffer 的空间。
  //
  // 典型值（非 grouped GEMM，AccumulatorStage=2, SchedulerStage=3）：
  //   32 + 48 + 16 + 48 + 48 + 8 + 12 = 212 B（约 0.09% 的 SMEM）
  //
  // 典型值（Conv builder，AccumulatorStage=2, SchedulerStage=1）：
  //   32 + 16 + 16 + 16 + 16 + 8 + 4 = 108 B（约 0.05% 的 SMEM）
  //
  // 典型值（Grouped GEMM，AccumulatorStage=2, SchedulerStage=3）：
  //   32 + 96 + 16 + 96 + 48 + 8 + 12 + 640 + 48 = 996 B（约 0.4% 的 SMEM）
  //
  // 注意：Carveout 本身对 stage 数限制很小，真正的瓶颈是 A/B tile 的 data size。
  // ==========================================================================
  static constexpr auto KernelSmemCarveout = static_cast<int>( AccumulatorPipelineStorage +
                                                               CLCPipelineStorage +
                                                               LoadOrderBarrierStorage +
                                                               TmemDeallocStorage +
                                                               CLCThrottlePipelineStorage +
                                                               CLCResponseStorage +
                                                               TmemBasePtrsStorage +
                                                               TensorMapStorage +
                                                               TensorMapReadyPipelineStorage
                                                              );
};

// ============================================================================
// Sm100SparseGemmTmaUmmaCarveout
// ============================================================================
// 作用：Sparse GEMM（2:4 structured sparsity）版本的 carveout 计算。
//       与 Dense 版本结构类似，但做了 16-byte 对齐的 round_up 处理，
//       且少了 TensorMapStorage（sparse GEMM 目前不支持 grouped/array 模式）。
//
// PipelineStorage：将 barrier 相关存储按 16B 对齐打包
// OtherStorage：将 response/pointer 按 16B 对齐打包
// ============================================================================

template<class ClusterShape_MNK, int AccumulatorPipelineStageCount, int SchedulerPipelineStageCount, int CLCResponseSize>
struct Sm100SparseGemmTmaUmmaCarveout {

  // LoadOrderBarrier = OrderedSequenceBarrier<1,2>
  // 作用同 dense 版本，控制 A/B TMA load 顺序。大小：16 B
  static constexpr auto LoadOrderBarrierStorage = sizeof(typename cutlass::OrderedSequenceBarrier<1,2>::SharedStorage);

  // CLCPipelineStorage = PipelineCLCFetchAsync
  // 作用同 dense 版本，Tile Scheduler fetch pipeline。大小：16×Stage B
  static constexpr auto CLCPipelineStorage = sizeof(typename cutlass::PipelineCLCFetchAsync<SchedulerPipelineStageCount, ClusterShape_MNK>::SharedStorage);

  // AccumulatorPipeline = PipelineUmmaAsync
  // 作用同 dense 版本，UMMA accumulator pipeline。大小：16×AccumulatorStage B
  static constexpr auto AccumulatorPipelineStorage = sizeof(typename cutlass::PipelineUmmaAsync<AccumulatorPipelineStageCount>::SharedStorage);

  // CLC Throttle pipeline storage
  // 作用同 dense 版本，Scheduler throttle control。大小：16×Stage B
  static constexpr auto CLCThrottlePipelineStorage = sizeof(typename cutlass::PipelineAsync<SchedulerPipelineStageCount>::SharedStorage);

  // Tmem dealloc
  // 作用同 dense 版本，TMEM dealloc barrier。大小：8 B
  static constexpr auto TmemDeallocStorage = sizeof(cutlass::arch::ClusterBarrier);

  // PipelineStorage：将上述 barrier 相关存储按 16-byte 对齐后求和，再 16-byte 对齐
  // 目的：确保 barrier 在 SMEM 中的对齐，防止硬件访问未对齐地址。
  static constexpr auto PipelineStorage = static_cast<int>(cutlass::round_up(
                                                      cutlass::round_up(LoadOrderBarrierStorage, 16) +
                                                      cutlass::round_up(CLCPipelineStorage, 16) +
                                                      cutlass::round_up(AccumulatorPipelineStorage, 16) +
                                                      cutlass::round_up(CLCThrottlePipelineStorage, 16) +
                                                      cutlass::round_up(TmemDeallocStorage, 16),
                                                    16));

  // CLC (scheduler) response
  // 作用同 dense 版本，scheduler response buffer。大小：Stage × CLCResponseSize
  static constexpr auto CLCQueryResponseStorage = SchedulerPipelineStageCount * CLCResponseSize;

  // Tmem ptr storage
  // 作用同 dense 版本，TMEM base pointer。注意 sparse 版本只存一个 uint32_t
  //（因为 sparse 的 accumulator pipeline 管理方式略有不同）。
  static constexpr auto TmemBasePtrsStorage = sizeof(uint32_t);

  // OtherStorage：将 response/pointer 按 16-byte 对齐后求和，再 16-byte 对齐
  static constexpr auto OtherStorage = static_cast<int>(cutlass::round_up(
                                                   cutlass::round_up(CLCQueryResponseStorage, 16) +
                                                   cutlass::round_up(TmemBasePtrsStorage, 16),
                                                 16));

  // KernelSmemCarveout = PipelineStorage + OtherStorage
  // 作用同 dense 版本，统计 sparse GEMM 中所有非 data/epilogue 的 SMEM 开销。
  static constexpr auto KernelSmemCarveout = static_cast<int>( PipelineStorage +
                                                               OtherStorage);
};
} // namespace cutlass::gemm::collective::detail
