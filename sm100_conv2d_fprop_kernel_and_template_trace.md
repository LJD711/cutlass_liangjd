# SM100 Conv2d Fprop：kernel 调用链与模板解析

本文追踪以下 CUTLASS 单元测试的第一个 case：

```cpp
TEST(SM100_device_conv2d_fprop_implicitgemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16,
     64x64x64_1x1x1)
```

源码：[`test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu)。

该 case 的配置为：

- 架构：`cutlass::arch::Sm100`
- 算法：`cutlass::conv::Operator::kFprop`（前向卷积）
- 数据类型：A/B/C/D/Accumulator 为 `cutlass::half_t`，计算标量为 `float`
- 布局：A、B、C/D 均为 `TensorNHWC`
- MMA tile：`Shape<_64, _64, Shape<_64>>`
- cluster：`Shape<_1, _1, _1>`

> 注意：同一个测试 suite 还定义了其他 tile/cluster case；每个 case 都会通过模板实例化产生一个独立的 CUDA kernel。

## 1. 最终启动的 CUDA kernel

这个测试没有调用某个固定名字的手写 `__global__` 函数。最终由 `ConvUniversalAdapter` 实例化并启动的是：

```cpp
cutlass::device_kernel<ConvKernel>
```

完整调用链：

```text
TEST(...)
  -> test::conv::device::TestAllConv<Conv>()
  -> ConvTestbed<Conv>::run()
  -> conv_op()                                  // ConvUniversalAdapter<ConvKernel>
  -> ConvUniversalAdapter::run(params)
  -> device_kernel<ConvKernel><<<grid, block, smem, stream>>>(params)
  -> ConvKernel::operator()(params, smem)
```

具体位置如下：

1. 测试将 `ConvKernel` 包装为 `ConvUniversalAdapter`，然后交给 `TestAllConv`：
   [`sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu:92-100`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu#L92-L100)。
2. 测试框架完成参数构造与 `initialize()` 后，调用 `conv_op()`：
   [`testbed_conv.hpp:459-467`](test/unit/conv/device_3x/testbed_conv.hpp#L459-L467)。
3. `ConvUniversalAdapter::run()` 对静态 `1x1x1` cluster 直接使用 CUDA 三尖括号语法启动 `device_kernel<ConvKernel>`：
   [`conv_universal_adapter.hpp:289-345`](include/cutlass/conv/device/conv_universal_adapter.hpp#L289-L345)。
4. 通用 CUDA entry point 创建算子对象并执行其 `operator()`：
   [`device_kernel.h:111-126`](include/cutlass/device_kernel.h#L111-L126)。
5. `ConvKernel::operator()` 实际落到 SM100 TMA warp-specialized 隐式 GEMM 实现：
   [`sm100_implicit_gemm_tma_warpspecialized.hpp:63-69`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L63-L69)，
   [`sm100_implicit_gemm_tma_warpspecialized.hpp:387-410`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L387-L410)。

这个 device-side `operator()` 内部，mainloop 的 TMA load 及 MMA 分别位于：

- [`collective_mainloop.load()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L584-L590)
- [`collective_mainloop.mma()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L699-L705)
- [`collective_epilogue.store()`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L831-L840)

## 2. 测试源码中五个模板别名的定义与实际特化

| 测试行 | 模板别名/类型 | 主模板定义 | 本例实际匹配的实现 |
| --- | --- | --- | --- |
| 70 | `cutlass::epilogue::collective::CollectiveBuilder` | [`epilogue/collective/collective_builder.hpp:49-73`](include/cutlass/epilogue/collective/collective_builder.hpp#L49-L73) | SM100 TensorOp 的 `EpilogueScheduleAuto` 特化：[`sm100_builder.inl:1677-1738`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1677-L1738)。对于此 `64x64x64, 1x1x1` case，自动选择 `TmaWarpSpecialized1Sm`，并委派给 TMA epilogue builder：[`sm100_builder.inl:1617-1665`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1617-L1665)。最终组装出的类型为 `CollectiveEpilogue<...>`，构造逻辑在 [`Sm100TmaBuilderImpl`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1169-L1179) 中，类型别名生成在 [`sm100_builder.inl:1335-1353`](include/cutlass/epilogue/collective/builders/sm100_builder.inl#L1335-L1353)。 |
| 80 | `cutlass::conv::collective::CollectiveBuilder` | [`conv/collective/collective_builder.hpp:65-84`](include/cutlass/conv/collective/collective_builder.hpp#L65-L84) | SM100 UMMA mainloop builder：[`sm100_umma_builder.inl:61-88`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl#L61-L88)。它匹配 `Sm100 + OpClassTensorOp + kFprop + KernelScheduleAuto`，并产出 `CollectiveConv<...>`：[`sm100_umma_builder.inl:203-225`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl#L203-L225)。具体 mainloop 特化是 [`sm100_implicit_gemm_umma_warpspecialized.hpp:76-105`](include/cutlass/conv/collective/sm100_implicit_gemm_umma_warpspecialized.hpp#L76-L105)。 |
| 91 | `cutlass::conv::ConvProblemShape` | [`convnd_problem_shape.hpp:56-92`](include/cutlass/conv/convnd_problem_shape.hpp#L56-L92) | 此例具体为 `ConvProblemShape<cutlass::conv::Operator::kFprop, 2>`：`ConvOp` 从第 80 行生成的 mainloop `DispatchPolicy` 取得，两个 `TensorNHWC` layout 推导出二维空间卷积。它是 host/device 共用的问题形状描述类型，不是 CUDA kernel。 |
| 92 | `cutlass::conv::kernel::ConvUniversal` | [`conv_universal.hpp:43-56`](include/cutlass/conv/kernel/conv_universal.hpp#L43-L56) | 匹配 SM100 TMA warp-specialized 特化：[`sm100_implicit_gemm_tma_warpspecialized.hpp:58-70`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp#L58-L70)。该特化组合 mainloop 与 epilogue，并定义实际 device-side `operator()`。 |
| 98 | `cutlass::conv::device::ConvUniversalAdapter` | [`conv_universal_adapter.hpp:53-73`](include/cutlass/conv/device/conv_universal_adapter.hpp#L53-L73) | 这是 host 侧的通用适配器，没有 SM100 专有类特化。它持有上述 `ConvUniversal<...>`，处理参数转换、workspace、动态共享内存与 cluster launch，然后启动 `device_kernel<ConvKernel>`。其 `run()` 实现在 [`conv_universal_adapter.hpp:286-400`](include/cutlass/conv/device/conv_universal_adapter.hpp#L286-L400)。 |

## 3. 本例的类型关系

```cpp
// 80 行：SM100 隐式 GEMM mainloop
using CollectiveMainloop =
  cutlass::conv::collective::CollectiveConv<
    MainloopSm100TmaUmmaWarpSpecializedImplicitGemm<...>, ...>;

// 70 行：SM100 TMA epilogue
using CollectiveEpilogue =
  cutlass::epilogue::collective::CollectiveEpilogue<...>;

// 91、92 行：组合卷积问题定义、mainloop、epilogue
using ConvKernel =
  cutlass::conv::kernel::ConvUniversal<
    cutlass::conv::ConvProblemShape<cutlass::conv::Operator::kFprop, 2>,
    CollectiveMainloop,
    CollectiveEpilogue>;

// 98 行：host 侧启动封装
using Conv = cutlass::conv::device::ConvUniversalAdapter<ConvKernel>;
```

## 4. 继续阅读的推荐顺序

1. [`sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu`](test/unit/conv/device_3x/fprop/sm100_conv2d_fprop_implicit_gemm_f16_f16_f16_tensorop_f16.cu)：查看测试具体实例参数。
2. [`conv/collective/builders/sm100_umma_builder.inl`](include/cutlass/conv/collective/builders/sm100_umma_builder.inl)：理解 mainloop 类型如何由 builder 选出。
3. [`epilogue/collective/builders/sm100_builder.inl`](include/cutlass/epilogue/collective/builders/sm100_builder.inl)：理解 auto epilogue schedule 与 `CollectiveEpilogue` 的生成。
4. [`conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp`](include/cutlass/conv/kernel/sm100_implicit_gemm_tma_warpspecialized.hpp)：理解设备端的 scheduler、TMA load、UMMA MMA 与 epilogue 协作流程。
5. [`conv/device/conv_universal_adapter.hpp`](include/cutlass/conv/device/conv_universal_adapter.hpp)：理解 host 端参数准备和 kernel launch。
