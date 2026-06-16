# CUTLASS Conv2d 调用链梳理

本文以 CUTLASS SM80 fprop 单测为例，整理从 GTest 入口到最终 CUDA kernel 执行的完整调用链，并说明 `analytic` 与 `optimized` 两种路径是如何选择的。

示例文件：

```text
/data/cutlass/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu
```

## 总览

```text
GTest TEST
  -> DefaultConv2dFprop<..., IteratorAlgorithm>::Kernel
  -> conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>
  -> TestAllConv2d<Conv2dFprop>()
  -> TestbedConv2d<Conv2dFprop>::run()
  -> conv2d_op.initialize(arguments, workspace)
  -> conv2d_op()
  -> ImplicitGemmConvolution::run()
  -> cutlass::Kernel<UnderlyingKernel><<<...>>>(params)
  -> UnderlyingKernel::operator()(params, shared_storage)
  -> IteratorA / IteratorB
  -> Mma mainloop
  -> Epilogue writeback
```

## 1. Test 入口

文件：

```text
/data/cutlass/test/unit/conv/device/conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu
```

analytic case 显式传入：

```cpp
cutlass::conv::IteratorAlgorithm::kAnalytic
```

位置：

```text
conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu:78
```

optimized case 显式传入：

```cpp
cutlass::conv::IteratorAlgorithm::kOptimized
```

位置：

```text
conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu:119
```

随后包装成 device operator：

```cpp
using Conv2dFprop =
  cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>;
```

然后调用：

```cpp
EXPECT_TRUE(test::conv::device::TestAllConv2d<Conv2dFprop>());
```

## 2. TestAllConv2d

文件：

```text
/data/cutlass/test/unit/conv/device/conv2d_testbed.h
```

入口：

```text
conv2d_testbed.h:579
```

它创建 testbed：

```cpp
TestbedConv2d<ImplicitGemm> testbed;
```

位置：

```text
conv2d_testbed.h:590
```

然后遍历 problem size，并调用：

```cpp
testbed.run(conv_problem, cutlass::conv::SplitKMode::kSerial);
```

位置：

```text
conv2d_testbed.h:682
```

## 3. TestbedConv2d::run

入口：

```text
/data/cutlass/test/unit/conv/device/conv2d_testbed.h:218
```

核心流程：

```cpp
initialize(problem_size);

Conv2d conv2d_op;

typename Conv2d::Arguments conv2d_args(...);

status = conv2d_op.initialize(conv2d_args, workspace.get());

status = conv2d_op();
```

关键位置：

```text
构造 Conv2d conv2d_op:      conv2d_testbed.h:245
构造 Conv2d::Arguments:     conv2d_testbed.h:247
调用 initialize():          conv2d_testbed.h:262
调用 operator():            conv2d_testbed.h:287
```

这里的 `Conv2d` 就是 test 中传进来的模板参数 `Conv2dFprop`：

```cpp
cutlass::conv::device::ImplicitGemmConvolution<Conv2dFpropKernel>
```

## 4. Device Wrapper

文件：

```text
/data/cutlass/include/cutlass/conv/device/implicit_gemm_convolution.h
```

入口：

```text
implicit_gemm_convolution.h:52
```

定义：

```cpp
template<typename ImplicitGemmKernel_>
class ImplicitGemmConvolution
```

模板参数 `ImplicitGemmKernel_` 就是前面由 `DefaultConv2dFprop<...>::Kernel` 生成的 `Conv2dFpropKernel`。

`initialize()` 中会把 test 传入的 `Arguments` 转成真正 kernel 用的 `Params`：

```cpp
params_ = typename UnderlyingKernel::Params(
  args,
  static_cast<int *>(workspace)
);
```

位置：

```text
implicit_gemm_convolution.h:278
```

`run()` 中计算 grid/block，然后 launch：

```cpp
cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);
```

位置：

```text
implicit_gemm_convolution.h:348
```

## 5. CUDA Kernel Wrapper

文件：

```text
/data/cutlass/include/cutlass/device_kernel.h
```

入口：

```text
device_kernel.h:72
```

核心逻辑：

```cpp
template <typename Operator>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params) {
  extern __shared__ int SharedStorageBase[];

  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

  Operator op;

  op(params, *shared_storage);
}
```

也就是说，真正执行的是：

```cpp
UnderlyingKernel::operator()(params, shared_storage)
```

## 6. 真正的 Conv Kernel Body

文件：

```text
/data/cutlass/include/cutlass/conv/kernel/implicit_gemm_convolution.h
```

入口：

```text
implicit_gemm_convolution.h:280
```

核心流程：

```cpp
// 计算 CTA tile
threadblock_tile_idx = threadblock_swizzle.get_tile_offset(...);

// 构造 A/B iterator
typename Mma::IteratorA iterator_A(...);
typename Mma::IteratorB iterator_B(...);

// 构造 threadblock MMA
Mma mma(...);

// 执行 implicit GEMM mainloop
mma(params.gemm_k_iterations,
    accumulators,
    iterator_A,
    iterator_B,
    accumulators,
    params.gemm_k_iterations_per_channel);

// epilogue 写回
epilogue(output_op, iterator_D, accumulators, iterator_C);
```

关键位置：

```text
构造 IteratorA / IteratorB:  implicit_gemm_convolution.h:310
构造 Mma:                   implicit_gemm_convolution.h:342
执行 mainloop:              implicit_gemm_convolution.h:350
epilogue 写回:              implicit_gemm_convolution.h:424
```

## 7. analytic 和 optimized 如何判断

这里不是运行时根据 problem size 自动判断，而是编译期模板参数决定。

`DefaultConv2dFprop` 的模板参数里有：

```cpp
conv::IteratorAlgorithm IteratorAlgorithm = IteratorAlgorithm::kOptimized
```

位置：

```text
/data/cutlass/include/cutlass/conv/kernel/default_conv2d_fprop.h:78
```

因此，如果 test 没有显式传入 `IteratorAlgorithm`，默认就是 `kOptimized`。

## 8. analytic 路径

当模板参数是：

```cpp
IteratorAlgorithm::kAnalytic
```

会匹配 analytic 特化：

```text
/data/cutlass/include/cutlass/conv/kernel/default_conv2d_fprop.h:112
```

它选择：

```cpp
Conv2dFpropActivationTileAccessIteratorAnalytic
Conv2dFpropFilterTileAccessIteratorAnalytic
```

位置：

```text
activation iterator: default_conv2d_fprop.h:145
filter iterator:     default_conv2d_fprop.h:158
```

然后继续构造：

```cpp
using Mma = threadblock::ImplicitGemmMultistage<...>;

using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
  Mma,
  Epilogue,
  ThreadblockSwizzle,
  conv::Operator::kFprop
>;
```

也就是说，analytic 和 optimized 最终使用的是同一个 kernel skeleton：

```cpp
cutlass::conv::kernel::ImplicitGemmConvolution
```

区别主要在 `IteratorA` 和 `IteratorB` 的类型。

## 9. optimized 路径

当模板参数是：

```cpp
IteratorAlgorithm::kOptimized
```

会匹配 optimized 特化：

```text
/data/cutlass/include/cutlass/conv/kernel/default_conv2d_fprop.h:1081
```

它选择：

```cpp
Conv2dFpropActivationTileAccessIteratorOptimized
Conv2dFpropFilterTileAccessIteratorOptimized
```

位置：

```text
activation iterator: default_conv2d_fprop.h:1115
filter iterator:     default_conv2d_fprop.h:1129
```

然后同样构造：

```cpp
using Mma = threadblock::ImplicitGemmMultistage<...>;

using Kernel = cutlass::conv::kernel::ImplicitGemmConvolution<
  Mma,
  Epilogue,
  ThreadblockSwizzle,
  conv::Operator::kFprop
>;
```

因此 optimized 与 analytic 的主要区别不是 kernel body，而是 global memory iterator 的实现。

## 10. can_implement 的作用

`can_implement()` 不是选择 analytic/optimized 的地方。

它只检查“当前已经实例化好的 kernel 能不能跑这个 problem”。

device wrapper 中：

```cpp
Status status = UnderlyingKernel::Mma::IteratorA::can_implement(args.problem_size);

status = UnderlyingKernel::Mma::IteratorB::can_implement(args.problem_size);
```

位置：

```text
/data/cutlass/include/cutlass/conv/device/implicit_gemm_convolution.h:104
```

因此，完整逻辑是：

```text
test 中选择 IteratorAlgorithm
  -> 编译期匹配 DefaultConv2dFprop 特化
  -> 生成不同 IteratorA / IteratorB 类型
  -> 生成对应 Conv2dFpropKernel
  -> runtime 只做 can_implement 合法性检查
  -> launch 这个已经确定好的 kernel
```

## 11. analytic 与 optimized 的本质差异

analytic iterator：

```text
Conv2dFpropActivationTileAccessIteratorAnalytic
Conv2dFpropFilterTileAccessIteratorAnalytic
```

特点：

```text
在 device 端按公式计算 pointer offset、mask、filter r/s/c 前进逻辑。
实现更直接，适配性更强，但主循环中整数计算和条件判断更多。
```

optimized iterator：

```text
Conv2dFpropActivationTileAccessIteratorOptimized
Conv2dFpropFilterTileAccessIteratorOptimized
```

特点：

```text
把部分 kernel-invariant pointer delta 放到 host 端 Params 构造时预计算。
device 端 iterator advance 主要查 inc_next delta table。
使用 fast divmod 映射 GEMM M/N/K 到卷积 tensor 坐标。
减少 mainloop 中的整数计算开销。
```

## 12. 最终结论

```text
analytic / optimized 不是运行时自动判断出来的。

它们由 DefaultConv2dFprop 的 IteratorAlgorithm 模板参数在编译期决定。

二者最终都会生成 cutlass::conv::kernel::ImplicitGemmConvolution kernel。

区别在于 DefaultConv2dFprop 特化时选择了不同的 IteratorA / IteratorB：

  kAnalytic
    -> Conv2dFpropActivationTileAccessIteratorAnalytic
    -> Conv2dFpropFilterTileAccessIteratorAnalytic

  kOptimized
    -> Conv2dFpropActivationTileAccessIteratorOptimized
    -> Conv2dFpropFilterTileAccessIteratorOptimized

runtime 的 can_implement 只是检查当前实例化好的 kernel 是否支持当前 problem。
```
