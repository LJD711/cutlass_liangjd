# SM100 Conv2d Fprop：从测试第 146 行到 device kernel `operator()`

## 0. 结论先行

本文分析的入口是：

```cpp
// test/unit/conv/device_3x/fprop/
// sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu:146
EXPECT_TRUE(test::conv::device::TestAllConv<Conv>());
```

对应的测试 case 是 `128x64x64_1x1x1`：

```cpp
using MmaTileShape = Shape<_128, _64, Shape<_64>>;
using ClusterShape = Shape<_1, _1, _1>;
using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;
```

第 146 行不会在 host 端直接调用目标 `operator()`。它同时依赖两条链：

```text
编译期类型选择：

CollectiveBuilder<Sm100, ..., KernelScheduleAuto>
  -> SM100 CollectiveMainloop
  -> DispatchPolicy::Schedule
       = KernelScheduleImplicitTmaWarpSpecializedSm100<1, 2>
  -> ConvUniversal<ProblemShape, Mainloop, Epilogue>
  -> 命中 sm100_implicit_gemm_tma_warpspecialized.hpp 中的偏特化

运行时调用：

EXPECT_TRUE(TestAllConv<Conv>())
  -> TestAllConv<Conv>()
  -> ConvTestbed<Conv>::run(...)
  -> conv_op.initialize(args, workspace)
  -> conv_op()
  -> ConvUniversalAdapter<ConvKernel>::operator()()
  -> ConvUniversalAdapter<ConvKernel>::run(params_)
  -> device_kernel<ConvKernel><<<grid, block, smem_size, stream>>>(params_)
  -> device_kernel 内的 op(params, smem)
  -> ConvKernel::operator()(Params const&, char*)
```

最后一跳发生在通用 CUDA 入口 [`device_kernel.h:118-123`](include/cutlass/device_kernel.h#L118-L123)。此时：

- `Operator` 就是测试中定义的 `ConvKernel`；
- `params` 是 adapter 在 host 端准备好、随后按值传入 kernel 的 `ConvKernel::Params`；
- `smem` 是 CUDA 动态共享内存首地址；
- `op(params, smem)` 因而解析到 [`sm100_implicit_gemm_tma_warpspecialized.hpp:389`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L389) 的目标函数。

---

## 1. 第 146 行中的 `Conv` 到底是什么

入口 case 位于 [`sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu:107-146`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L107-L146)。关键配置如下：

| 配置 | 本例取值 |
| --- | --- |
| 架构 | `cutlass::arch::Sm100` |
| 卷积操作 | `cutlass::conv::Operator::kFprop` |
| A / activation | `cutlass::half_t`, `TensorNHWC`, alignment = 8 elements |
| B / filter | `cutlass::half_t`, `TensorNHWC`, alignment = 8 elements |
| C、D、Accumulator | `cutlass::half_t` |
| epilogue compute | `float` |
| MMA tile | `Shape<_128, _64, Shape<_64>>` |
| cluster | `Shape<_1, _1, _1>` |
| mainloop schedule | `KernelScheduleAuto` |
| epilogue schedule | `EpilogueScheduleAuto` |

测试代码依次构造了四个重要类型。

### 1.1 `CollectiveEpilogue`

[`测试文件:116-124`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L116-L124) 调用 SM100 epilogue builder：

```cpp
using CollectiveEpilogue = typename
  cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    MmaTileShape,
    ClusterShape,
    ...,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;
```

`EpilogueScheduleAuto` 偏特化在 [`sm100_builder.inl:1677`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1677)。它通过 [`is_2sm()`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1698) 判断 epilogue 使用 1SM 还是 2SM。

本例 `ClusterShape_M = 1`，不是 2 的倍数，因此 [`is_2sm()`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1703) 返回 false，选择 `TmaWarpSpecialized1Sm`，然后委派给 [`sm100_builder.inl:1617`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1617) 的 TMA epilogue builder。

这一步主要影响最终 kernel 组合进去的 epilogue 类型和 `CollectiveEpilogue::SharedStorage` 大小。

### 1.2 `CollectiveMainloop`

[`测试文件:126-135`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L126-L135) 实例化 conv mainloop builder：

```cpp
using CollectiveMainloop = typename
  cutlass::conv::collective::CollectiveBuilder<
    cutlass::arch::Sm100,
    cutlass::arch::OpClassTensorOp,
    cutlass::conv::Operator::kFprop,
    cutlass::half_t, cutlass::layout::TensorNHWC, 8,
    cutlass::half_t, cutlass::layout::TensorNHWC, 8,
    cutlass::half_t,
    Shape<_128, _64, Shape<_64>>,
    Shape<_1, _1, _1>,
    StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>,
    KernelScheduleAuto
  >::CollectiveOp;
```

`CollectiveBuilder` 的主模板位于 [`collective_builder.hpp:65-84`](include/cutlass/conv/collective/collective_builder.hpp#L65-L84)，文件末尾包含 SM100 builder [`collective_builder.hpp:92-93`](include/cutlass/conv/collective/collective_builder.hpp#L92-L93)。

本例会匹配 [`sm100_umma_builder.inl:61`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl#L61) 的偏特化，原因是：

1. `ArchTag == arch::Sm100`；
2. `OpClass == arch::OpClassTensorOp`；
3. `KernelScheduleType == KernelScheduleAuto`；
4. A、B 都是 16-bit `half_t`，alignment 是 8 个元素，所以每次对齐宽度是 `2 * 8 = 16B`，满足 TMA alignment 要求。

对于 `kFprop`，builder 令 A、B 都为 K-major：

```cpp
UmmaMajorA = cute::UMMA::Major::K;
UmmaMajorB = cute::UMMA::Major::K;
```

见 [`sm100_umma_builder.inl:89-99`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl#L89-L99)。

接着 `KernelScheduleAuto` 进入 [`sm100_make_tiled_mma()`](include/cutlass/conv/collective/builders/sm100_common.inl#L145)：

```cpp
if constexpr (ClusterShape_M % 2 == 0 && TileShape_M % 128 == 0) {
  // 2SM
}
else {
  // 1SM
}
```

本例虽然 `TileShape_M = 128`，但 `ClusterShape_M = 1`，所以选择 1SM 路径 [`sm100_common.inl:170-173`](include/cutlass/conv/collective/builders/sm100_common.inl#L170-L173)。对于 half 输入，底层 MMA atom 进一步选择：

```cpp
cute::SM100_MMA_F16BF16_SS<
  half_t, half_t, half_t,
  128, 64,
  UMMA::Major::K, UMMA::Major::K>
```

对应选择逻辑位于 [`gemm/collective/builders/sm100_common.inl:320-324`](include/cutlass/gemm/collective/builders/sm100_common.inl#L320-L324)。这里的 `SS` 表示 UMMA 的 A、B operand 都由 shared-memory descriptor 提供。

builder 最后生成如下 dispatch policy：

```cpp
using DispatchPolicy =
  MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<
    kFprop,
    PipelineStages,
    2,                    // TensorNHWC -> 2D convolution
    1,                    // SchedulerPipelineStageCount
    2,                    // AccumulatorPipelineStageCount
    Shape<_1,_1,_1>,
    arch::Sm100>;
```

并将它装配进 `CollectiveConv<...>`，见 [`sm100_umma_builder.inl:233-251`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl#L233-L251)。因此可将测试里的别名概括成：

```cpp
using CollectiveMainloop =
  cutlass::conv::collective::CollectiveConv<
    MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<...>,
    Shape<_128, _64, Shape<_64>>,
    half_t,
    half_t,
    TiledMma,
    TileTraitsA,
    TileTraitsB>;
```

该 `CollectiveConv` 自身又命中 [`sm100_implicit_gemm_umma_warpspecialized.hpp:76`](include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp#L76) 的 mainloop 特化。

### 1.3 `ProblemShape`

[`测试文件:137`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L137) 写的是：

```cpp
using ProblemShape = cutlass::conv::ConvProblemShape<
  CollectiveMainloop::DispatchPolicy::ConvOp,
  CollectiveMainloop::DispatchPolicy::NumSpatialDimensions>;
```

上一步已经得到：

```cpp
ConvOp               = cutlass::conv::Operator::kFprop;
NumSpatialDimensions = 2;  // TensorNHWC
```

所以它等价于：

```cpp
using ProblemShape =
  cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>;
```

### 1.4 `ConvKernel` 为什么落入目标头文件

[`测试文件:138-142`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L138-L142)：

```cpp
using ConvKernel = cutlass::conv::kernel::ConvUniversal<
  ProblemShape,
  CollectiveMainloop,
  CollectiveEpilogue>;
```

`ConvUniversal` 的第四、第五模板参数使用默认值，因此完整形式是：

```cpp
ConvUniversal<
  ProblemShape,
  CollectiveMainloop,
  CollectiveEpilogue,
  void,  // TileSchedulerTag_
  void   // Enable
>
```

主模板定义在 [`conv_universal.hpp:42-55`](include/cutlass/conv/kernel/conv_universal.hpp#L42)，它本身只会报“找不到有效 kernel 特化”。该文件随后包含 SM90 和 SM100 kernel 实现，见 [`conv_universal.hpp:61-62`](include/cutlass/conv/kernel/conv_universal.hpp#L61)。

SM100 偏特化的匹配条件是 [`sm100_implicit_gemm_tma_warpspecialized.hpp:63-69`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L63-L69)：

```cpp
cute::is_base_of_v<
  KernelImplicitTmaWarpSpecializedSm100,
  typename CollectiveMainloop::DispatchPolicy::Schedule>
```

而上面 builder 产生的 dispatch policy 在 [`dispatch_policy.hpp:128`](include/cutlass/conv/dispatch_policy.hpp#L128) 中定义：

```cpp
using Schedule =
  KernelScheduleImplicitTmaWarpSpecializedSm100<1, 2>;
```

该 schedule 又在 [`dispatch_policy.hpp:107`](include/cutlass/conv/dispatch_policy.hpp#L107) 明确继承：

```cpp
struct KernelScheduleImplicitTmaWarpSpecializedSm100
  : KernelImplicitTmaWarpSpecializedSm100 { ... };
```

所以 `is_base_of_v` 为 true，`enable_if_t<true>` 得到 `void`，正好匹配 `ConvUniversal` 的第五个模板参数。至此，`ConvKernel` 就是目标头文件中定义了 device `operator()` 的那个类特化。

### 1.5 `Conv` 只是 host adapter

[`测试文件:144`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L144)：

```cpp
using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;
```

`ConvUniversalAdapter` 不是最终 CUDA kernel。它是 host 侧有状态句柄，保存一个：

```cpp
Params params_;
```

并负责 Arguments→Params 转换、workspace 初始化、grid/block/smem 计算以及 kernel launch，定义见 [`conv_universal_adapter.hpp:60-143`](include/cutlass/conv/device/conv_universal_adapter.hpp#L60-L143)。

---

## 2. 第 146 行的运行时调用链

### 2.1 `EXPECT_TRUE` 先执行 `TestAllConv<Conv>()`

`EXPECT_TRUE(expr)` 是 GoogleTest 断言。它先求值 `expr`，所以进入 [`testbed_conv.hpp:702`](test/unit/conv/device_3x/testbed_conv.hpp#L702)：

```cpp
template <typename Conv, bool SupportStrides = ...>
bool TestAllConv(...) {
  ConvTestbed<Conv> testbed;
  auto problem_vector = get_conv_problem_vector<
    Conv::NumSpatialDimensions,
    Conv::DispatchPolicy::ConvOp,
    SupportStrides>();

  for (auto conv_problem : problem_vector) {
    ...
    passed = testbed.run(conv_problem, ...);
  }
  return passed;
}
```

本例模板实参展开为：

```cpp
get_conv_problem_vector<2, cutlass::conv::Operator::kFprop, true>()
```

二维 fprop problem 列表定义在 [`conv_problem_sizes.hpp:191`](test/unit/conv/device_3x/conv_problem_sizes.hpp#L191)。`TestAllConv` 在 [`testbed_conv.hpp:728-752`](test/unit/conv/device_3x/testbed_conv.hpp#L728-L752) 遍历列表，并针对每个 problem 调用一次 `ConvTestbed<Conv>::run()`。

因此需要注意：第 146 行不是“只执行一个卷积问题”。它会对测试列表中的多个 problem 重复走后面的 launch 链。

### 2.2 `ConvTestbed::run()` 构造 host-facing `Arguments`

`ConvTestbed<Conv>::run()` 从 [`testbed_conv.hpp:332`](test/unit/conv/device_3x/testbed_conv.hpp#L332) 开始。与调用链直接相关的工作是：

1. 检查设备可用 SMEM 是否至少为 `Conv::ConvKernel::SharedStorageSize`；
2. 初始化 A、B、C、D 等测试 tensor；
3. 创建 `Conv conv_op`，也就是 `ConvUniversalAdapter<ConvKernel>`；
4. 构造 mainloop arguments、epilogue arguments、hardware info 和 scheduler arguments；
5. 汇总成 `typename Conv::Arguments args`。

`args` 的构造位于 [`testbed_conv.hpp:394-410`](test/unit/conv/device_3x/testbed_conv.hpp#L394-L410)：

```cpp
auto args = typename Conv::Arguments {
  problem_shape,
  mainloop_args,
  epilogue_args,
  hw_info,
  scheduler_args
};
```

由于 adapter 中：

```cpp
using Arguments = typename ConvKernel::Arguments;
```

这里构造的实际类型就是目标 `ConvKernel::Arguments`，其字段定义在 [`sm100_implicit_gemm_tma_warpspecialized.hpp:192-199`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L192-L199)。

### 2.3 `can_implement()` 只检查，不启动 kernel

[`testbed_conv.hpp:447-453`](test/unit/conv/device_3x/testbed_conv.hpp#L447-L453) 调用：

```cpp
status = conv_op.can_implement(args);
```

adapter 将检查委托给 `ConvKernel::can_implement(args)`，而 kernel 再检查 mainloop、epilogue、tile scheduler、cluster shape 和 multicast 对齐等条件。这里没有 CUDA kernel launch。

### 2.4 `initialize()` 将 `Arguments` 降为 `Params`

[`testbed_conv.hpp:455-459`](test/unit/conv/device_3x/testbed_conv.hpp#L455-L459) 先申请 workspace，再调用：

```cpp
status = conv_op.initialize(args, workspace.data().get());
```

adapter 的 [`initialize()`](include/cutlass/conv/device/conv_universal_adapter.hpp#L230) 做两件关键事情：

```cpp
ConvKernel::initialize_workspace(args, workspace, stream, cuda_adapter);
params_ = ConvKernel::to_underlying_arguments(args, workspace);
```

`ConvKernel::to_underlying_arguments()` 位于 [`sm100_implicit_gemm_tma_warpspecialized.hpp:233-263`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L233-L263)，大致完成以下映射：

| `Arguments` 中的 host-facing 数据 | `Params` 中的 device-facing 数据 |
| --- | --- |
| 原始 `ConvProblemShape` | mainloop 转换得到的隐式 GEMM `problem_shape` |
| A/B 指针等 `mainloop` 参数 | TMA descriptor/stride 等 `mainloop` params |
| C/D 与 fusion `epilogue` 参数 | device epilogue params |
| scheduler arguments | 带 workspace 指针的 scheduler params |
| `KernelHardwareInfo` | `hw_info` |

最终存进 adapter 成员 `params_` 的类型就是 [`ConvKernel::Params`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L202)。这正是目标 device `operator()` 的第一个参数类型。

如果共享内存超过 48 KiB，`initialize()` 还会对 `device_kernel<ConvKernel>` 设置 `cudaFuncAttributeMaxDynamicSharedMemorySize`，见 [`conv_universal_adapter.hpp:254-267`](include/cutlass/conv/device/conv_universal_adapter.hpp#L254-L267)。

### 2.5 `conv_op()` 选择无参重载

[`testbed_conv.hpp:466-467`](test/unit/conv/device_3x/testbed_conv.hpp#L466-L467)：

```cpp
status = conv_op();
```

adapter 有两个 `operator()`：

```cpp
operator()(Arguments const& args, ...);  // 第一个参数不可省略
operator()(cudaStream_t stream = nullptr);
```

这里没有传入 `Arguments`，所以选择 [`conv_universal_adapter.hpp:440-442`](include/cutlass/conv/device/conv_universal_adapter.hpp#L440-L442) 的无参重载：

```cpp
Status operator()(cudaStream_t stream = nullptr) {
  return run(params_, stream);
}
```

也就是说，它复用第 2.4 节已经保存好的 `params_`。

### 2.6 adapter 的静态 `run(Params&)` 选择 launch 路径

随后进入 [`conv_universal_adapter.hpp:289`](include/cutlass/conv/device/conv_universal_adapter.hpp#L289)：

```cpp
static Status run(Params& params, cudaStream_t stream = nullptr, ...) {
  dim3 const block = ConvKernel::get_block_shape();
  dim3 const grid  = get_grid_shape(params);
  int smem_size    = ConvKernel::SharedStorageSize;
  ...
}
```

本例各个编译期分支如下：

| 分支条件 | 本例结果 | 后果 |
| --- | --- | --- |
| `ArchTag::kMinComputeCapability >= 90` | true（SM100） | 进入 cluster-aware launch 部分 |
| `ClusterShape` 是静态 shape | true | cluster 在编译期已知 |
| `size(ClusterShape{}) == 1` | true（1×1×1） | `is_static_1x1x1 == true` |
| `CUTLASS_ENABLE_CUDA_HOST_ADAPTER` | 默认 false | 使用原生 CUDA launch 路径 |
| compute capability 为 90 或 100 | true（100） | 进入直接/cluster launch 分支 |

`CUTLASS_ENABLE_CUDA_HOST_ADAPTER` 的默认值可见 [`cuda_host_adapter.hpp:75-76`](include/cutlass/cuda_host_adapter.hpp#L75-L76)。于是最终执行 [`conv_universal_adapter.hpp:343-345`](include/cutlass/conv/device/conv_universal_adapter.hpp#L343-L345)：

```cpp
device_kernel<ConvKernel>
    <<<grid, block, smem_size, stream>>>(params);
```

这里四个 launch 配置参数分别是：

- `grid`：`ConvKernel::get_grid_shape(params)` 通过 tile scheduler 计算；
- `block`：`ConvKernel::get_block_shape()`，即 `(MaxThreadsPerBlock, 1, 1)`；
- `smem_size`：`sizeof(ConvKernel::SharedStorage)`；
- `stream`：本次调用没有显式指定，因此为默认 stream。

若 cluster 不是静态 1×1×1，这里会改走 `ClusterLauncher`；但本例不会经过该路径。

### 2.7 `device_kernel<ConvKernel>` 是真正的 `__global__` 入口

`device_kernel` 定义于 [`device_kernel.h:111-126`](include/cutlass/device_kernel.h#L111-L126)：

```cpp
template <typename Operator>
CUTLASS_GLOBAL
__launch_bounds__(Operator::MaxThreadsPerBlock,
                  Operator::MinBlocksPerMultiprocessor)
void device_kernel(
    CUTLASS_GRID_CONSTANT typename Operator::Params const params) {
  extern __shared__ char smem[];
  Operator op;
  op(params, smem);
  cutlass::arch::synclog_print();
}
```

模板代入后就是：

```cpp
void device_kernel<ConvKernel>(ConvKernel::Params const params) {
  extern __shared__ char smem[];
  ConvKernel op;
  op(params, smem);
}
```

这解释了最后一跳为什么没有额外的显式函数名：它依赖函数对象调用语法 `op(...)`。

### 2.8 `op(params, smem)` 命中目标 `operator()`

此时 `op` 的静态类型就是第 1.4 节已经选定的 SM100 `ConvUniversal` 偏特化，所以：

```cpp
op(params, smem);
```

编译器只能解析到该类中的：

```cpp
CUTLASS_DEVICE
void operator()(Params const& params, char* smem_buf);
```

也就是 [`sm100_implicit_gemm_tma_warpspecialized.hpp:387-389`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L387-L389)。

需要特别区分两个名字相似的调用：

- host 上的 `conv_op()`：对象类型是 `ConvUniversalAdapter<ConvKernel>`，负责 launch；
- device 上的 `op(params, smem)`：对象类型是 `ConvKernel`，执行实际卷积 kernel。

它们是两个不同类的 `operator()`。

---

## 3. 目标函数两个参数从哪里来

### 3.1 `Params const& params`

完整来源是：

```text
ConvTestbed 构造 Conv::Arguments
  -> ConvUniversalAdapter::initialize(args, workspace)
  -> ConvKernel::to_underlying_arguments(args, workspace)
  -> 保存到 ConvUniversalAdapter::params_
  -> adapter 的 run(params_)
  -> params_ 按值传给 device_kernel
  -> device_kernel 再以 const 引用传给 ConvKernel::operator()
```

因此 target `operator()` 接收到的引用指向 CUDA kernel 参数区中的 `Params` 对象，而不是直接引用 host 上 adapter 的 `params_` 内存。

### 3.2 `char* smem_buf`

来源是 `device_kernel` 中：

```cpp
extern __shared__ char smem[];
```

其容量由 launch 的第三个配置参数：

```cpp
smem_size = ConvKernel::SharedStorageSize;
```

决定。动态共享内存按 CTA 分配，同一 CTA 中所有线程看到相同的 `smem` 首地址；不同 CTA 拥有各自独立的 shared memory。

目标函数一进入就把它恢复成强类型对象，见 [`sm100_implicit_gemm_tma_warpspecialized.hpp:413-415`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L413-L415)：

```cpp
SharedStorage& shared_storage =
    *reinterpret_cast<SharedStorage*>(smem_buf);

static_assert(
    SharedStorageSize <= ArchTag::kSharedMemoryCapacityBytes,
    "SMEM usage exceeded capacity.");
```

`SharedStorage` 包含 mainloop pipeline、epilogue pipeline、load-order barrier、CLC pipeline、accumulator pipeline、CLC response、TMEM base pointer，以及 mainloop/epilogue tensor storage，定义见 [`sm100_implicit_gemm_tma_warpspecialized.hpp:161-188`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L161-L188)。

---

## 4. 进入目标 `operator()` 后做什么

这不是调用链的必需部分，但有助于理解为什么 `device_kernel` 的每个 CUDA thread 都调用同一个 `operator()`。

### 4.1 每个线程先确定 warp 角色

[`sm100_implicit_gemm_tma_warpspecialized.hpp:398-401`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L398-L401) 根据 warp index 分配角色：

```cpp
MMA          = 0;
Sched        = 1;
MainloopLoad = 2;
EpilogueLoad = 3;
Epilogue     = 4;  // 以及后续 epilogue warps
```

所以 `operator()` 虽然由 block 内所有线程执行，但随后不同 warp 进入不同分支，分别承担 scheduler、TMA load、UMMA、epilogue load/store 等职责。

### 4.2 构造 mainloop 和 epilogue collective

[`sm100_implicit_gemm_tma_warpspecialized.hpp:417-418`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L417-L418)：

```cpp
CollectiveMainloop collective_mainloop(
    params.mainloop, cluster_shape, cta_rank_in_cluster);

CollectiveEpilogue collective_epilogue(
    params.epilogue, shared_storage.tensors.epilogue);
```

### 4.3 主要计算阶段

关键调用点为：

- mainloop TMA load：[`collective_mainloop.load()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L584)
- UMMA consume/compute：[`collective_mainloop.mma()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L700)
- epilogue：[`collective_epilogue.store()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L834)

在 host 侧，launch 后 adapter 只立即检查 `cudaGetLastError()`。测试框架随后在 [`testbed_conv.hpp:475`](test/unit/conv/device_3x/testbed_conv.hpp#L475) 调用 `cudaDeviceSynchronize()`，此处才等待 GPU 执行完成并检查异步执行错误，然后再和 host reference 结果比较。

---

## 5. 一张表串起所有关键跳转

| 阶段 | 位置 | 发生的事情 |
| --- | --- | --- |
| 测试入口 | [`测试文件:146`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L146) | 求值 `TestAllConv<Conv>()` |
| 遍历问题 | [`testbed_conv.hpp:702-752`](test/unit/conv/device_3x/testbed_conv.hpp#L702-L752) | 创建 testbed，遍历 2D fprop problem 列表 |
| 单个 problem | [`testbed_conv.hpp:332`](test/unit/conv/device_3x/testbed_conv.hpp#L332) | 准备 tensor、Arguments、workspace |
| 参数 lowering | [`conv_universal_adapter.hpp:230-267`](include/cutlass/conv/device/conv_universal_adapter.hpp#L230-L267) | `Arguments -> ConvKernel::Params` 并保存到 `params_` |
| host 调用 | [`testbed_conv.hpp:467`](test/unit/conv/device_3x/testbed_conv.hpp#L467) | 调用 adapter 的无参 `operator()` |
| adapter 重载 | [`conv_universal_adapter.hpp:440-442`](include/cutlass/conv/device/conv_universal_adapter.hpp#L440-L442) | 转发到 `run(params_)` |
| launch 配置 | [`conv_universal_adapter.hpp:289-316`](include/cutlass/conv/device/conv_universal_adapter.hpp#L289-L316) | 计算 grid、block、动态 SMEM |
| CUDA launch | [`conv_universal_adapter.hpp:343-345`](include/cutlass/conv/device/conv_universal_adapter.hpp#L343-L345) | 启动 `device_kernel<ConvKernel>` |
| `__global__` 入口 | [`device_kernel.h:118-123`](include/cutlass/device_kernel.h#L118-L123) | 声明动态 SMEM，构造 `ConvKernel op`，执行 `op(params, smem)` |
| 最终目标 | [`sm100_implicit_gemm_tma_warpspecialized.hpp:389`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L389) | 进入 `ConvKernel::operator()(Params const&, char*)` |
| 等待完成 | [`testbed_conv.hpp:475`](test/unit/conv/device_3x/testbed_conv.hpp#L475) | `cudaDeviceSynchronize()` 并检查执行错误 |

---

## 6. 容易混淆的几点

1. `CollectiveBuilder`、`CollectiveConv`、`ConvUniversal` 的模板匹配都发生在编译期，不是第 146 行运行时逐个调用的函数。
2. `ConvUniversal` 是 CUDA kernel 的函数对象类型；真正带 `__global__` 的入口是 `device_kernel<ConvKernel>`。
3. `ConvUniversalAdapter::operator()` 和 `ConvKernel::operator()` 不是同一个函数：前者在 host 侧 launch，后者在 device 侧执行。
4. `Params` 由 `Arguments` lowering 得到；`smem_buf` 则由 CUDA runtime 按 launch 的动态共享内存大小提供。
5. 一个 `device_kernel` CUDA thread 会调用一次 `ConvKernel::operator()`；warp-specialized 分支让不同 warp 执行不同职责。
6. `TestAllConv` 会遍历多个 problem，因此第 146 行通常产生多次 kernel launch；每次 launch 又包含多个 CTA。
7. 本例是静态 `1x1x1` cluster，所以直接使用 `<<<...>>>`；其他 cluster 配置可能通过 `ClusterLauncher` 启动，但最终仍进入同一个 `device_kernel<ConvKernel>` 包装层。
