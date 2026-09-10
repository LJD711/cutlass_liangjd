 SM100 Conv2d：CLC Pipeline 流程与 PTX 汇总

## 0. 文档范围

主要源码：

- kernel：[`sm100_implicit_gemm_tma_warpspecialized.hpp`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp)
- scheduler：[`sm100_tile_scheduler.hpp`](include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp)
- CLC pipeline：[`sm100_pipeline.hpp`](include/cutlass/pipeline/sm100_pipeline.hpp)
- mbarrier PTX：[`barrier.h`](include/cutlass/arch/barrier.h)

本例使用：

```cpp
using CLCPipeline = PipelineCLCFetchAsync<1, ClusterShape>;
```

即只有一个 response stage，每次复用 `stage 0` 时通过 phase 翻转区分新旧轮次。

---

## 1. CLC pipeline 传递什么

CLC 是 Cluster Launch Control。

正在运行的 persistent cluster 通过：

```ptx
clusterlaunchcontrol.try_cancel
```

尝试取消一个尚未启动的 cluster，并接手它原本对应的工作。

CLC 返回一个 opaque 16B response：

```cpp
struct CLCResponse {
  uint32_t data[4];
};
```

response 中包含：

- 是否成功取消 cluster；
- 成功时，被取消 cluster 的首 CTA ID。

这里的调度粒度是 **cluster**，不是 cluster 内每个 CTA 分别调度：

| 层级 | 实际执行者 |
| --- | --- |
| 认领一份新工作 | 一个正在运行的 active cluster |
| 生产 CLC query | 该 cluster 的 CTA0 Sched warp |
| 发射 `try_cancel` PTX | 上述 warp 中 `elect_one_sync()` 选出的一个 lane |

一次成功 query 会取消并接手一个尚未启动的完整 cluster。返回 `first_ctaid`
只是用被取消 cluster 的首 CTA 坐标表示这份 cluster work；请求方 cluster 中各 CTA
再结合自己的 `block_id_in_cluster_` 得到各自的 tile。cluster shape 为 `1x1x1`
时，cluster 中只有一个 CTA，此时才表面上等价于 CTA 级调度。

首个 work `W0` 不经过 CLC，它直接来自当前 cluster 的 `blockIdx`：

```cpp
WorkTileInfo initial_work_tile_info(...) {
  return swizzle_and_rasterize(
    blockIdx.x, blockIdx.y, blockIdx.z,
    /*valid=*/true,
    /*cluster_offset_m=*/0,
    /*cluster_offset_n=*/0);
}
```

后续 `W1、W2、...` 才来自 CLC query。

---

## 2. full barrier、empty barrier 和 response slot

[`PipelineCLCFetchAsync::SharedStorage`](include/cutlass/pipeline/sm100_pipeline.hpp#L921-L931) 每个 stage 包含：

```cpp
using FullBarrier  = cutlass::arch::ClusterTransactionBarrier;
using EmptyBarrier = cutlass::arch::ClusterBarrier;

FullBarrier  full_barrier_[Stages];
EmptyBarrier empty_barrier_[Stages];
```

kernel 另有相同 stage 数量的 response slot：

```cpp
CLCResponse clc_response[SchedulerPipelineStageCount];
```

三者的关系是：

```text
empty_barrier[n]
    表示 response[n] 已经被所有 consumer 读取
    ready 后 producer 才能覆盖该 slot

full_barrier[n]
    表示本轮 16B CLC response 已经到达
    ready 后 consumer 才能读取该 slot

clc_response[n]
    保存 CLC 返回的 opaque b128 数据
```

因此依赖关系为：

```text
所有 consumer 读完旧 response
        │
        ▼
empty[n] ready
        │
        ▼
producer 发出新 CLC query
        │
        ▼
硬件写入 16B response，并完成 transaction
        │
        ▼
full[n] ready
        │
        ▼
所有 consumer 可以读取新 response
```

---

## 3. 完整 CLC 流程

```text
CTA0 的 Sched warp                         Cluster 内所有参与角色
        │                                            │
        │ 1. producer_acquire(stage)                 │
        │    等 empty barrier                        │
        │                                            │
        │ 2. 给每个 CTA 的 full barrier 登记：       │
        │    arrival = 1                             │
        │    expected transaction = 16B              │
        │                                            │
        │ 3. 一个 elected lane 发 CLC query          │
        │    try_cancel 尚未启动的 cluster            │
        │                                            │
        │        CLC 硬件异步执行                     │
        │             │                              │
        │             ├─ multicast 16B response      │
        │             └─ complete_tx(16B)            │
        │                                            │
        │                         full barrier ready  │
        │                                            │
        │                         4. consumer_wait()  │
        │                         5. 读取 response    │
        │                         6. 解析 valid/ctaid │
        │                         7. consumer_release │
        │                                            │
        │<----- 所有 consumer 向 CTA0 empty arrive --│
        │                                            │
        │    empty barrier ready                     │
        │                                            │
        └────────── stage 可以用于下一次 query ───────┘
```

只有 CTA0 会进入 scheduler 主循环，判断条件来自：

```cpp
is_participant.sched =
  (warp_category == WarpCategory::Sched) &&
  is_first_cta_in_cluster;
```

因此其他 CTA 的 Sched warp 不会各自调用 `advance_to_next_work()`。其他 CTA 的
MainloopLoad、MMA、Epilogue 等 consumer warp 只等待并读取 multicast 给本 CTA
的同一份 response。

参与读取 response 的角色包括：

- CTA0 的 Sched warp；
- cluster 内所有 MainloopLoad warp；
- cluster 内所有 MMA warp；
- cluster 内所有 Epilogue warp；
- 需要加载 C/Aux 时的 EpilogueLoad warp。

因此 [`consumer_arv_count`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L482-L498) 按所有参与线程计算：

```cpp
consumer_arv_count =
    NumSchedThreads
  + cluster_size *
      (NumMainloopLoadThreads
       + NumEpilogueThreads
       + NumMMAThreads);

if (is_epi_load_needed) {
  consumer_arv_count += cluster_size * NumEpilogueLoadThreads;
}
```

少任何一组 consumer arrival，CTA0 的 `empty_barrier` 都不会 ready，scheduler 也就不能复用 response slot。

---

## 4. Producer：`advance_to_next_work()`

实现位于 [`sm100_tile_scheduler.hpp:438-456`](include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L438-L456)：

```cpp
PipelineState<Stages>
advance_to_next_work(
    Pipeline& clc_pipeline,
    PipelineState<Stages> producer_state) const {

  uint32_t mbarrier_addr =
    clc_pipeline.producer_get_barrier(producer_state);

  clc_pipeline.producer_acquire(producer_state);

  if (cute::elect_one_sync()) {
    issue_clc_query(
      producer_state,
      mbarrier_addr,
      clc_response_ptr_);
  }

  ++producer_state;
  return producer_state;
}
```

调用顺序：

```text
producer_get_barrier()
    取得 full_barrier[stage] 的 SMEM 地址
        │
        ▼
producer_acquire()
    等 empty_barrier[stage]
    给 cluster 内每个 CTA 的 full barrier 登记 16B
        │
        ▼
elect_one_sync()
    只让一个 lane 发 query
        │
        ▼
issue_clc_query()
    尝试取消未启动 cluster
    将 response multicast 给所有 CTA
        │
        ▼
++producer_state
    推进 stage/phase
```

其中前两项“为每个 CTA 登记 barrier”和“只发一条 query”由 CTA0 Sched warp
以两种不同的 lane 分工完成。假设 `cluster_size = 4`：

```text
CTA0 Sched warp 执行 producer_acquire()
  lane 0 -> CTA0 full_barrier.arrive_expect_tx(16B)
  lane 1 -> CTA1 full_barrier.arrive_expect_tx(16B)
  lane 2 -> CTA2 full_barrier.arrive_expect_tx(16B)
  lane 3 -> CTA3 full_barrier.arrive_expect_tx(16B)
  lane 4..31 -> pred=false，不操作

随后同一个 warp 执行 elect_one_sync()
  -> 仅一个 elected lane
  -> 仅发出一条 clusterlaunchcontrol.try_cancel
```

所以 `mapa + mbarrier.arrive.expect_tx` 表示 CTA0 的 warp 在远程“映射”所有 CTA
的本地 full barrier，并不表示每个 CTA 都发一条 query。

这里没有显式调用：

```cpp
clc_pipeline.producer_commit(...);
```

因为 `clusterlaunchcontrol.try_cancel` 自带：

```text
.mbarrier::complete_tx::bytes
```

response 到达时由硬件完成 full barrier 的 16B transaction。

---

## 5. Consumer：`fetch_next_work()`

实现位于 [`sm100_tile_scheduler.hpp:458-477`](include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L458-L477)：

```cpp
auto fetch_next_work(
    WorkTileInfo current_work,
    TileSchedulerPipeline& pipeline,
    TileSchedulerPipelineState consumer_state) {

  pipeline.consumer_wait(consumer_state);

  uint32_t smem_addr = cute::cast_smem_ptr_to_uint(
    &clc_response_ptr_[consumer_state.index()]);

  auto next_work =
    work_tile_info_from_clc_response(smem_addr);

  pipeline.consumer_release(consumer_state);

  next_work = swizzle_and_rasterize(
    next_work.M_idx,
    next_work.N_idx,
    next_work.L_idx,
    next_work.is_valid(),
    block_id_in_cluster_.x,
    block_id_in_cluster_.y);

  return cute::make_tuple(next_work, true);
}
```

执行顺序：

```text
consumer_wait(full[stage])
        │
        ▼
ld.shared.b128 response
        │
        ▼
query_cancel.is_canceled
        │
        ├─ false → next_work.valid = false
        │
        └─ true  → get_first_ctaid(x,y,z)
                        │
                        ▼
                  临时写入 M_idx/N_idx/L_idx
                        │
                        ▼
                  swizzle_and_rasterize()
                        │
                        ▼
                  当前 CTA 的最终 tile 坐标

读取 response 后：
consumer_release()
    → 向 CTA0 的 empty_barrier[stage] arrive
```

CLC 返回的是被取消 cluster 的首 CTA ID，不一定等于最终 `WorkTileInfo`：

- cluster 大于 `1x1x1` 时，要加当前 CTA 的 cluster 内 offset；
- 启用 swizzle/raster-order 时，还要执行坐标变换。

---

## 6. `operator()` 中的 query/fetch 循环

CTA0 Sched warp 的核心循环位于
[`sm100_implicit_gemm_tma_warpspecialized.hpp:628-661`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L628-L661)：

```cpp
bool requires_clc_query = true;

do {
  if (requires_clc_query) {
    producer_state = scheduler.advance_to_next_work(
      clc_pipeline, producer_state);
  }

  auto [next_work, increment_pipe] =
    scheduler.fetch_next_work(
      work_tile_info,
      clc_pipeline,
      consumer_state);

  requires_clc_query = increment_pipe;

  if (increment_pipe) {
    ++consumer_state;
  }

  work_tile_info = next_work;

} while (work_tile_info.is_valid());

clc_pipeline.producer_tail(producer_state);
```

基础 `PersistentTileSchedulerSm100` 每次 fetch 都消费一个 CLC response，因此：

```cpp
increment_pipe == true;
```

Stream-K 可能只在当前 work unit 内更新 `K_idx/k_tile_count`，没有消费新的 CLC response，此时：

```cpp
increment_pipe == false;
```

所以不能无条件推进 CLC state，也不能无条件发下一次 query。

当 query 返回 `valid=false` 时，表示没有更多可取消的 cluster。各角色完成当前 work 后退出；Sched warp 最后通过 `producer_tail()` 等所有未回收 stage 的 empty barrier。

---

## 7. PTX 指令汇总

### 7.1 实际执行路径

| 阶段 | C++ 调用 | 关键 PTX | 作用 |
| --- | --- | --- | --- |
| barrier 初始化 | `FullBarrier::init()` / `EmptyBarrier::init()` | `mbarrier.init.shared::cta.b64` | 初始化 full/empty barrier 及 arrival count |
| 发布初始化 | `fence_barrier_init()` | `fence.mbarrier_init.release.cluster` | 让 barrier 初始化可在 cluster 范围发布 |
| 等 stage 可覆盖 | `producer_acquire()` | 循环 `mbarrier.try_wait.parity.shared::cta.b64` | 等 `empty_barrier[stage]` ready |
| 映射远端 CTA barrier | `arrive_and_expect_tx()` | `mapa.shared::cluster.u32` | CTA0 Sched warp 的 lane `r` 得到 cluster rank `r` CTA 的 full barrier 地址 |
| 登记 response transaction | `arrive_and_expect_tx()` | `mbarrier.arrive.expect_tx.shared::cluster.b64` | CTA0 Sched warp 的前 `cluster_size` 个 lane 各为一个 CTA 登记 arrival 和 16B expected transaction |
| 选择一个 query lane | `elect_one_sync()` | `elect.sync` | 仅在 CTA0 Sched warp 内选择一个线程 |
| 发 CLC query | `issue_clc_query()` | `clusterlaunchcontrol.try_cancel.async...b128` | 每个请求方 active cluster 只发一条 query；取消一个未启动 cluster，并向本 cluster 所有 CTA multicast response |
| 等 response 到达 | `consumer_wait()` | 循环 `mbarrier.try_wait.parity.shared::cta.b64` | 等本 CTA 的 `full_barrier[stage]` ready |
| 读取 response | `work_tile_info_from_clc_response()` | `ld.shared.b128` | 将 opaque 16B response 读入寄存器 |
| 判断取消结果 | 同上 | `clusterlaunchcontrol.query_cancel.is_canceled.pred.b128` | 判断 query 是否成功取消 cluster |
| 生成 valid | 同上 | `selp.u32` | predicate 转换为 `valid=1/0` |
| 提取 CTA ID | 同上 | `clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128` | 成功时提取首 CTA 的 x/y/z 坐标 |
| async proxy fence | `fence_view_async_shared()` | `fence.proxy.async.shared::cta` | 处理 async proxy 与普通 SMEM 访问的可见性 |
| consumer 释放 stage | `consumer_release()` | `mapa.shared::cluster.u32` + `mbarrier.arrive.shared::cluster.b64` | 向 CTA0 的 empty barrier 报告一个 consumer 已读完 |
| producer 收尾 | `producer_tail()` | `mbarrier.test_wait.parity`；未完成时循环 `mbarrier.try_wait.parity` | 等所有 stage 被完整释放 |

### 7.2 核心 PTX：发出 query

源码：[`sm100_tile_scheduler.hpp:392-407`](include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L392-L407)

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta
  .mbarrier::complete_tx::bytes
  .multicast::cluster::all.b128
  [result_addr], [full_mbarrier_addr];
```

这里：

- `result_addr` 指向 `clc_response[stage]`；
- `full_mbarrier_addr` 指向 `full_barrier[stage]`；
- `.b128` 表示 response 为 16B；
- `.multicast::cluster::all` 表示所有 CTA 收到相同 response；
- `.complete_tx::bytes` 表示写完 response 后完成 barrier transaction。

这条 PTX 只由请求方 active cluster 的 CTA0 Sched warp 中一个 elected lane 发射。
它取消的是同一 grid 中另一个尚未启动的完整 cluster；multicast 的接收方则是当前
请求方 cluster 的所有 CTA。

### 7.3 核心 PTX：等待 barrier

源码：[`barrier.h:410-427`](include/cutlass/arch/barrier.h#L410-L427)

```ptx
LAB_WAIT:
  mbarrier.try_wait.parity.shared::cta.b64
    p_ready, [barrier_addr], phase, ticks;
  @p_ready bra DONE;
  bra LAB_WAIT;
DONE:
```

同一实现分别用于：

- producer 等 empty barrier；
- consumer 等 full barrier。

### 7.4 核心 PTX：为所有 CTA 登记 16B

源码：[`barrier.h:606-620`](include/cutlass/arch/barrier.h#L606-L620)

```ptx
setp.eq.u32 p, pred, 1;
@p mapa.shared::cluster.u32
     remote_addr, local_full_barrier, cta_id;
@p mbarrier.arrive.expect_tx.shared::cluster.b64
     _, [remote_addr], 16;
```

更准确地说，是 **CTA0 的 Sched warp** 的前 `cluster_size` 个 lane 各负责一个
CTA。`lane_idx` 被用作目标 `cta_id`，`mapa.shared::cluster` 把相同 SMEM offset
映射到该 CTA 的 full barrier；其他 CTA 不会重复执行一次 query。

### 7.5 核心 PTX：解析 response

源码：[`sm100_tile_scheduler.hpp:409-435`](include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp#L409-L435)

```ptx
.reg .pred p1;
.reg .b128 clc_result;

ld.shared.b128 clc_result, [result_addr];

clusterlaunchcontrol.query_cancel.is_canceled.pred.b128
  p1, clc_result;

selp.u32 valid, 1, 0, p1;

@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
  {M_idx, N_idx, L_idx, _}, clc_result;
```

失败时 `p1=false`，`get_first_ctaid` 不执行，调用方只能使用 `valid=false`，不能读取无效的 M/N/L。

### 7.6 核心 PTX：释放 response slot

源码：[`barrier.h:484-501`](include/cutlass/arch/barrier.h#L484-L501)

```ptx
mapa.shared::cluster.u32
  remote_addr, local_empty_barrier, producer_cta_id;

mbarrier.arrive.shared::cluster.b64
  _, [remote_addr];
```

本例：

```cpp
producer_cta_id = 0;
```

所以所有 consumer 都向 CTA0 的 `empty_barrier[stage]` arrive。