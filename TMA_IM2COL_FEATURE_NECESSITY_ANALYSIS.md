# TMA-im2col 硬件特性必要性分析

> **决策建议：建议加入 TMA-im2col，并保留非 TMA 软件回退路径。**  
> 如果目标芯片需要在 3×3、带 padding/stride 等主流卷积上充分释放 SM100 级 Tensor Core 的吞吐，TMA-im2col 更接近一项“架构使能特性”，而不是普通的可选优化。现有 NCU 数据对这一方向提供了强证据；但由于 Kernel 14 与 Kernel 29 的卷积形状、tile、MMA 指令和 CTA 组织不同，当前数据不能把全部性能差异严格归因于 TMA-im2col，最终立项前仍应完成同构 A/B 实验与 PPA 评估。

## 1. 分析对象与范围

- NCU 报告：[conv2d_NHWC_512_gpu0.ncu-rep](./conv2d_NHWC_512_gpu0.ncu-rep)
- GPU：NVIDIA B300 SXM6 AC，Compute Capability 10.3
- 解析工具：NVIDIA Nsight Compute 2026.2.1
- 对比对象：报告中的 Kernel ID 14 与 Kernel ID 29

两条 kernel 都执行 FP16 输入、FP32 累加的 NHWC Conv2D，但不是严格同形状 A/B：

| 项目 | Kernel 14 | Kernel 29 |
|---|---:|---:|
| 卷积形状 | N=512，H×W=160×160，C=256，K=256，R×S=3×3，stride=2 | N=512，H×W=40×40，C=128，K=128，R×S=3×3，stride=1 |
| 输出 | 512×80×80×256 | 512×40×40×128 |
| Kernel 路径 | SM100 CUTLASS，TMA-im2col，2CTA multicast | SM80 风格 cuDNN implicit GEMM，`LDGSTS` |
| 计算 tile | 256×256×64 | 128×64×32 |
| Tensor 指令 | `HMMA.2CTA` / UTCHMMA | `HMMA.16816.F32` |
| CTA | block 256 threads，2SM/2CTA | block 128 threads，单 CTA |

因此，本报告把它们视为两种**代表性数据供给架构**的端到端对比，而不是只切换一个硬件开关的因果实验。尤其不能直接用 0.831 ms 与 0.594 ms 判断快慢，因为两者工作量不同。

## 2. 结论摘要

1. **Kernel 29 的主要问题在片上供数路径，而不是 HBM。** 它的 DRAM 吞吐仅为 8.39%，但 L1/TEX 已到 76.99%，shared-load 管线达到 66.17%，全部 shared 管线达到 74.04%，Tensor 管线却只有 70.58%。这组指标共同说明：数据已较多命中片上缓存，但从缓存搬到 shared memory、再交给 Tensor Core 的过程占用了大量管线资源。

2. **Kernel 14 的 TMA 路径显著减轻了这类片上压力。** 它的 L1/TEX 为 48.22%，shared-load 管线仅为 7.03%，Tensor 管线达到 96.43%，SM 总吞吐达到 96.77%。TMA-im2col、multicast 和 2CTA pipeline 让 Tensor Core 在较低 occupancy 下仍能接近满载。

3. **“Kernel 29 没有使用 TMA”不是唯一差异，但很可能是关键架构差异之一。** Kernel 14 同时使用了更先进的 MMA、较大 tile、2SM 协作和不同的调度方式，所以不能把测得的 4.13× 运算速率差异全部记到 TMA-im2col 名下；然而 Kernel 29 的瓶颈恰好集中在 TMA-im2col 试图消除的地址生成、warp 发起搬运、L1/TEX 请求和 shared-memory feed 上，因而证据方向高度一致。

4. **建议的产品决策是“加入特性，同时保留回退”，而不是强制所有卷积使用 TMA-im2col。** 对 1×1、depthwise/grouped、很小的张量、低复用或不满足描述符约束的卷积，TMA-im2col 的收益可能较小，仍需要软件路径。

## 3. NCU 核心数据对比

| 指标 | Kernel 14：TMA-im2col | Kernel 29：`LDGSTS` | 观察 |
|---|---:|---:|---|
| SM throughput | **96.77%** | 70.58% | TMA 路径高 26.19 个百分点 |
| Tensor pipe active | **96.43%** | 70.58% | 高 25.85 个百分点，Tensor Core 更接近饱和 |
| NCU 报告 Tensor 运算速率 | **1.683 POp/s** | 0.407 POp/s | 端到端相差 4.13×；不能视作纯 TMA 加速比 |
| L1/TEX throughput | 48.22% | **76.99%** | 非 TMA 路径高 28.78 个百分点 |
| Shared-load pipe | 7.03% | **66.17%** | 非 TMA 路径高 59.14 个百分点 |
| Shared pipe overall | 14.10% | **74.04%** | 非 TMA 路径高 59.93 个百分点 |
| DRAM throughput | 47.31% | 8.39% | Kernel 29 明显不是 HBM 带宽饱和 |
| L2 throughput | 44.09% | 39.74% | 两者均未达到 L2 峰值 |
| L2 load hit rate | 57.22% | **92.92%** | Kernel 29 数据大多已在 L2，仍无法喂满 Tensor Core |
| Issue active | 11.22% | **54.30%** | Kernel 29 需要更多 warp 发射工作 |
| Eligible warps / scheduler | 0.12 | **0.82** | Kernel 14 依靠异步流水而非大量 eligible warps |
| Achieved occupancy | 9.68% | **18.39%** | 更高 occupancy 没有让 Kernel 29 获得更高 Tensor 利用率 |
| Registers / thread | 127 | **160** | 软件搬运/索引路径的寄存器压力更高 26% |
| Shared memory / block | 217.1 KiB | 64.5 KiB | Kernel 14 用更大 tile 和更深的异步流水换吞吐 |
| 动态 TMA load 指令 | **666,432** | 0 | Kernel 14 的 bulk-copy 路径 |
| 动态 `LDGSTS` 指令 | 0 | **12,288,000** | Kernel 29 由 warp 发起大量 global-to-shared 搬运 |
| Shared-load wavefront / 百万 Tensor ops | **8.90** | 457.81 | 观察值相差 51.4×；不同 MMA/粒度下只能作方向性比较 |
| 全部 shared wavefront / 百万 Tensor ops | **17.85** | 512.21 | 观察值相差 28.7×；说明旧路径每单位计算伴随更多 shared 流量 |

注：表中的 `% of peak sustained` 是 NCU 相对于对应硬件管线可持续峰值的利用率，不等价于“指令数量百分比”。wavefront 是 shared memory 执行请求的基本单位；访问被拆成更多 wavefront，通常意味着更多周期和更高管线压力。指标解释参见 [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)。

## 4. Kernel 29 为什么只有约 70.58% Tensor 利用率

### 4.1 不是显存带宽瓶颈

Kernel 29 的 DRAM throughput 只有 8.39%，L2 load hit rate 却达到 92.92%。这意味着数据大多已经进入片上缓存层级，继续增加 HBM 带宽不会直接解决 Tensor Core 的空闲周期。

真正突出的指标是：

- L1/TEX throughput：76.99%
- Shared-load pipe：66.17%
- Shared overall：74.04%
- Tensor pipe：70.58%
- 动态 `LDGSTS`：12,288,000 条

`LDGSTS` 路径仍要由 warp 执行索引/地址相关工作并发起 global-to-shared 搬运。大量请求经过 L1/TEX 与 shared-load 管线，使“把数据送到 Tensor Core”本身成为主要成本之一。

### 4.2 Shared-memory 访问还有拆分与冲突开销

NCU 对 Kernel 29 报告了：

- 30,572,544 个 excessive shared wavefront，约占 195,878,912 个 shared wavefront 的 16%
- 772,888 次 shared-store bank conflict，平均约 2.5-way
- No Eligible 周期约 44.91%
- 每条已发射指令对应的主要 stall：Math Pipe Throttle 1.10、Wait 0.86、Short Scoreboard 0.75、Barrier 0.42 cycles

这说明瓶颈并非单一的 shared bank conflict，而是数据搬运、同步、依赖等待和计算发射共同构成的供数效率问题。TMA-im2col 不能消除所有 shared 冲突，但可以减少 warp 直接参与搬运和动态卷积索引的工作量，为主计算 warp 留出发射、寄存器和流水资源。

### 4.3 更高 occupancy 和 issue active 并没有换来更高 Tensor 利用率

Kernel 29 的 achieved occupancy 为 18.39%，明显高于 Kernel 14 的 9.68%；issue active 也从 11.22% 提高到 54.30%。如果问题主要是“warp 不够多”，Kernel 29 理应更容易隐藏延迟。但实际 Tensor pipe 仍只有 70.58%。

这是一条很重要的架构信号：Kernel 29 在用更多 warp 发射槽执行搬运、索引、同步和依赖管理，而不是把这些资源全部转换成 Tensor 运算。

## 5. Kernel 14 的 TMA-im2col 路径带来了什么

Kernel 14 的 SASS 中包含：

```text
UTMALDG.5D.IM2COL.MULTICAST.2CTA
UTMALDG.5D.MULTICAST.2CTA
HMMA.2CTA
```

NCU 还记录到 666,432 次动态 TMA load、0 次 `LDGSTS`。这说明激活数据的卷积窗口构造和搬运由 TMA-im2col 执行，权重等规则张量使用普通 tiled TMA，并通过 multicast 服务 2CTA/2SM 计算。

CUTLASS 源码也直接说明：fprop 的 tensor A 使用 im2col TMA、tensor B 使用 tiled TMA，见 [SM100 implicit-GEMM collective](../include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp#L210)。对应的 5D、2CTA multicast PTX 发射位于 [SM100 TMA copy implementation](../include/cute/arch/copy_sm100_tma.hpp#L591)。

TMA-im2col 在这里承担的功能包括：

- 按卷积 stride、padding、dilation 和 filter offset 生成输入窗口地址
- 处理边界/OOB 语义，而不需要每个线程重复执行完整边界判断
- 将多维 global-memory tile 异步批量搬到 shared memory
- 用 barrier 表示搬运完成，使 producer/consumer pipeline 与 Tensor 运算重叠
- 通过 multicast 让协作 CTA 共享搬运结果，减少重复流量

官方 CUDA 文档将 TMA 描述为多维 global/shared 异步 bulk copy 机制，可卸载地址计算；PTX ISA 则定义了 `.im2col` tensor copy 形式。参见 [CUDA Programming Guide：Asynchronous Copies](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html) 和 [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)。

### 为什么 SASS 中是 `5D.IM2COL`

这里的 5D 不是说原始 Conv2D 输入突然变成了五维张量，而是 TMA 指令的坐标/描述符表达需要同时编码输出位置与卷积窗口偏移。CUTLASS 将其解释为 `(c, [w,h,d], n, [s,r,t])`，见 [im2col copy traits](../include/cute/atom/copy_traits_sm90_im2col.hpp#L71)。对 Conv2D，深度和 `t` 维退化为 1，但仍可使用统一的 5D 硬件形式。

### 低 issue active 并不代表 Kernel 14 没有工作

Kernel 14 的 issue active 只有 11.22%，eligible warps 只有 0.12，NCU 也能观察到较高 long-scoreboard 等待；但 Tensor pipe 仍达到 96.43%。这是 warp-specialized 异步流水的典型结果：少量 producer 发起粗粒度 TMA，消费者持续执行 2CTA Tensor 指令，不需要所有 warp 每周期都发射数据搬运指令。

Kernel 14 也并非完全没有代价：它每 block 使用约 217.1 KiB shared memory，occupancy 受限于 shared memory；其 shared load/store 仍存在 bank conflict。TMA-im2col 的价值不是让所有等待和冲突归零，而是在这些限制存在时仍能把 Tensor 管线维持在 96% 以上。

## 6. TMA-im2col 的必要性判断

### 6.1 支持加入的强证据

| 证据链 | NCU 事实 | 对硬件决策的含义 |
|---|---|---|
| 非 TMA 路径受片上 feed 限制 | Kernel 29：L1/TEX 76.99%，shared-load 66.17%，DRAM 8.39% | 单纯增加外存带宽无效，需要优化地址生成和 global-to-shared 数据通路 |
| TMA 路径接近计算峰值 | Kernel 14：Tensor 96.43%，SM 96.77% | TMA-im2col 能支撑高吞吐 Tensor pipeline |
| 软件路径发射成本高 | Kernel 29：12.288M `LDGSTS`，issue active 54.30%，160 registers/thread | 硬件卸载有机会减少 warp 指令、索引状态和寄存器压力 |
| TMA 与 2SM 协作互相依赖 | Kernel 14 使用 `.IM2COL.MULTICAST.2CTA` 和 `HMMA.2CTA` | 缺少 TMA-im2col 会削弱 multicast/2CTA 计算结构的整体收益 |
| 单位计算的 shared 活动差距大 | shared-load wavefront/Mop：457.81 vs 8.90 | 非 TMA 方案把明显更多片上执行资源消耗在搬运上 |

从这些数据看，如果芯片的性能目标包含高占比的 3×3 卷积，**不加入 TMA-im2col 的代价不只是少一个便捷指令，而是可能无法稳定喂满已经投入面积的 Tensor Core**。Tensor Core 很昂贵，让它因供数路径停在约 70% 利用率通常不是平衡的微架构方案。

### 6.2 不能从当前数据直接证明的内容

以下结论不能仅凭这两个 profile 下定论：

- **不能说 TMA-im2col 单独带来 4.13×。** 该数字是两条完整 kernel 的 NCU 运算速率之比，同时混合了 shape、tile、SM80/SM100 MMA、2CTA、调度和融合策略差异。
- **不能用两条 kernel 的 duration 直接算加速比。** 工作量相差很大。
- **不能从 NCU 推出面积、频率、功耗和验证成本。** TMA 可能降低搬运指令能耗，但这是合理推断，不是本报告已有的功耗测量。
- **不能证明所有卷积都会获益。** 1×1、depthwise、低复用或小规模问题可能无法摊薄描述符建立和流水启动成本。

因此，对“方向是否值得做”的置信度为**高**，对“平均能加速多少”的置信度目前为**中低**。

## 7. 推荐的硬件能力边界

如果决定加入，建议不要只实现一个受限的地址生成捷径，而应保证它能真正支撑当前 Kernel 14 展示的计算结构：

1. 支持 Conv2D/Conv3D 所需的 3D/4D/5D im2col 坐标表达。
2. 支持 stride、padding、dilation、filter offset 和 OOB zero-fill。
3. 支持异步 global-to-shared bulk copy，并用硬件 barrier 表示完成。
4. 支持 CTA cluster multicast；若目标 Tensor Core 采用 2SM/2CTA MMA，应优先支持 2CTA multicast。
5. 允许与普通 tiled TMA 并行使用，因为卷积激活和权重往往分别走 im2col 与 tiled 模式。
6. 保留普通 `LDGSTS`/软件寻址回退，以覆盖不支持或不划算的 shape。

其中前四项是获得 Kernel 14 这类端到端收益的核心，不宜只根据“功能可运行”裁剪掉 multicast、OOB 或 barrier 语义。

## 8. 立项前必须完成的同构 A/B 实验

建议在同一 GPU、同一频率、同一卷积形状下构造两个 kernel，只改变数据获取方式：

- A：TMA-im2col + async barrier
- B：软件地址生成 + `LDGSTS`

尽量固定 MMA 指令、tile、stage 数、2CTA 策略、融合操作、workspace 和输出数值。测试集至少覆盖：

- filter：1×1、3×3、5×5
- stride：1、2
- padding：0、1 及非对称 padding
- dilation：1 及大于 1
- C/K：对齐与非对齐通道数
- 大/小 batch，大/小 spatial
- 普通、grouped、depthwise convolution

应同时采集：

- kernel duration 和 NCU Tensor ops/s
- Tensor pipe、SM throughput
- TMA 与 `LDGSTS` 动态指令数
- L1/TEX、shared-load/overall throughput
- shared wavefront、excessive wavefront、bank conflict
- issue active、No Eligible、各类 stall
- registers/thread、shared memory/block、occupancy
- L2/DRAM throughput 与 hit rate
- 芯片模型或 RTL 的面积、频率、功耗、能量/operation

建议的初始 go/no-go 门槛：

- 目标卷积集合几何平均 latency 改善 ≥10%～15%，或业务权重最高的 3×3 卷积改善 ≥20%
- 受益 case 的 Tensor-pipe 利用率平均提升 ≥10 个百分点
- L1/TEX 或 shared-load 压力降低 ≥15%
- 非受益 case 通过回退路径控制在 3%～5% 以内的回归
- 面积、频率和验证成本满足项目 PPA 预算

这些数字应根据真实业务权重和芯片成本模型调整，不应取代同构实验本身。

## 9. 最终建议

**建议加入 TMA-im2col。** 当前 NCU 报告已经显示：当没有 TMA-im2col、依赖 warp 发起 `LDGSTS` 时，Kernel 29 在 DRAM 远未饱和的情况下，L1/TEX 和 shared-load 管线已经达到 76.99% 与 66.17%，Tensor Core 只能达到 70.58%；而 Kernel 14 采用 TMA-im2col + multicast + 2CTA 后，在 L1/TEX 48.22%、shared-load 7.03% 的条件下把 Tensor 管线推到 96.43%。

这组数据足以支持“该特性对高吞吐卷积具有架构必要性”的判断，但不足以给出独立的 TMA 加速百分比。合理的工程决策是：

> **把 TMA-im2col 纳入硬件，以保护 Tensor Core 投资和 SM100 级 2CTA 卷积路径的上限；同时保留软件回退，并用同构 A/B + PPA 数据决定最终实现范围。**

## 参考资料

- [NVIDIA Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [CUDA Programming Guide：Asynchronous Copies / TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)
- [PTX ISA：Tensor bulk copy 与 im2col](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [CUTLASS：Implicit GEMM Convolution](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/implicit_gemm_convolution.html)

