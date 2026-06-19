# CUTLASS Conv2d Implicit GEMM: Shared Memory 与 Swizzle 调用关系笔记

本文整理三个问题：

1. `implicit_gemm_convolution.h::operator()` 里，kernel launch 之后如何一步步调用到 shared memory layout 的 swizzle。
2. `main_loop` 是什么，`typename Mma::SharedStorage main_loop` 的内存何时分配。
3. 为什么可以确定 `MmaBase::LayoutA()` 最终就是 `SmemLayoutA`。

相关文件：

- [implicit_gemm_convolution.h](/data/cutlass/include/cutlass/conv/kernel/implicit_gemm_convolution.h)
- [implicit_gemm_multistage.h](/data/cutlass/include/cutlass/conv/threadblock/implicit_gemm_multistage.h)
- [mma_base.h](/data/cutlass/include/cutlass/gemm/threadblock/mma_base.h)
- [default_mma_core_sm80.h](/data/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h)
- [regular_tile_access_iterator_tensor_op.h](/data/cutlass/include/cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h)
- [tensor_op_multiplicand_sm75.h](/data/cutlass/include/cutlass/layout/tensor_op_multiplicand_sm75.h)
- [default_mma_tensor_op.h](/data/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h)
- [mma_tensor_op.h](/data/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h)
- [device_kernel.h](/data/cutlass/include/cutlass/device_kernel.h)

## 1. 从 kernel `operator()` 到 swizzle 的调用链

在 [implicit_gemm_convolution.h](/data/cutlass/include/cutlass/conv/kernel/implicit_gemm_convolution.h) 中，真正的 device kernel 入口是 `ImplicitGemmConvolution::operator()`。

核心代码形态如下：

```cpp
Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

mma(
  params.gemm_k_iterations,
  accumulators,
  iterator_A,
  iterator_B,
  accumulators,
  params.gemm_k_iterations_per_channel);
```

这里分两步：

1. 构造 `Mma mma(...)`。
2. 调用 `mma(...)` 执行 mainloop。

swizzle 不是在 `implicit_gemm_convolution.h` 中直接调用的，而是在构造和使用 shared memory iterator 时被间接调用。

完整路径可以按下面理解：

```text
conv/kernel/implicit_gemm_convolution.h
  ImplicitGemmConvolution::operator()

    -> Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx)

      -> conv/threadblock/implicit_gemm_multistage.h
         ImplicitGemmMultistage constructor

        -> smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx)

          -> transform/threadblock/regular_tile_access_iterator_tensor_op.h
             RegularTileAccessIterator<RowMajorTensorOpMultiplicandCrosswise> constructor

            -> UnderlyingIterator(ref, thread_id)

              -> TensorRef::offset(thread_offset)

                -> layout_(coord)

                  -> layout/tensor_op_multiplicand_sm75.h
                     RowMajorTensorOpMultiplicandCrosswise::operator()

                    -> TensorOpMultiplicandCrosswise::operator()

                      -> TensorOpMultiplicand::operator()
                         这里执行 XOR swizzle 地址映射
```

也就是说：

- `implicit_gemm_convolution.h` 只负责搭好 kernel 内部对象并启动 mainloop。
- `ImplicitGemmMultistage` 负责 global memory 到 shared memory 的 mainloop pipeline。
- `RegularTileAccessIterator` 负责按线程映射访问 shared memory tile。
- `SmemLayoutA` 决定 shared memory 中逻辑坐标到线性 offset 的映射。
- `TensorOpMultiplicand::operator()` 是真正执行 shared memory swizzle 计算的位置。

### 1.1 构造 `Mma` 时发生了什么

在 [implicit_gemm_multistage.h](/data/cutlass/include/cutlass/conv/threadblock/implicit_gemm_multistage.h) 中，`Mma` 类型通常是 `ImplicitGemmMultistage`。

它的构造函数中会构造 shared memory iterator：

```cpp
smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx)
smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx)
```

这里的关键是 `shared_storage.operand_A_ref()`。

它返回一个 `TensorRef`：

```cpp
TensorRefA{operand_A.data(), LayoutA()}
```

`TensorRef` 里面包含两部分：

1. shared memory 中 operand A 的起始地址。
2. layout 对象，也就是 `LayoutA()`。

后续 iterator 只要根据坐标求 offset，就会调用：

```cpp
ref.offset(coord)
```

而 `TensorRef::offset(coord)` 本质会调用：

```cpp
layout_(coord)
```

这就进入 layout 的 `operator()`。

### 1.2 为什么会调用到 `TensorOpMultiplicand::operator()`

对于这里分析的 SM80 Tensor Core 路径，`SmemLayoutA` 是：

```cpp
using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
    sizeof_bits<ElementA>::value,
    Shape::kK>;
```

以 `fp16 + Crosswise = 64` 为例：

```cpp
RowMajorTensorOpMultiplicandCrosswise<16, 64>
```

这个 layout 是一个 row-major 矩阵视角的包装。它收到的是 `MatrixCoord(row, column)`，但是底层 `TensorOpMultiplicand` 使用的是 pitch-linear 坐标：

```cpp
PitchLinearCoord(contiguous, strided)
```

因此 `RowMajorTensorOpMultiplicandCrosswise::operator()` 会做一次坐标转换：

```cpp
return layout_(TensorCoord(coord.column(), coord.row()));
```

也就是：

```text
MatrixCoord(row, column)
  -> PitchLinearCoord(contiguous = column, strided = row)
```

然后进入：

```cpp
TensorOpMultiplicandCrosswise::operator()
```

再进入底层：

```cpp
TensorOpMultiplicand::operator()
```

这里就是 shared memory swizzle 的核心计算。

## 2. `main_loop` 是什么，shared memory 是什么时候分配的

在 [implicit_gemm_convolution.h](/data/cutlass/include/cutlass/conv/kernel/implicit_gemm_convolution.h) 中，kernel 的 shared storage 通常长这样：

```cpp
union SharedStorage {
  typename Mma::SharedStorage main_loop;
  typename Epilogue::SharedStorage epilogue;
};
```

这里用 `union` 的原因是：

- mainloop 阶段需要 shared memory 存 A/B tile。
- epilogue 阶段需要 shared memory 做输出重排、归约或写回辅助。
- 两个阶段不是同时使用，所以可以复用同一块 shared memory。

因此：

```cpp
shared_storage.main_loop
```

不是单独分配的一块新内存，而是整个 kernel dynamic shared memory 里的一个视图。

### 2.1 为什么在 `implicit_gemm_multistage.h` 找不到 `SharedStorage`

`ImplicitGemmMultistage` 自己没有直接定义 `SharedStorage`，它继承自：

```cpp
gemm::threadblock::MmaBase<Shape_, Policy_, Stages>
```

所以：

```cpp
typename Mma::SharedStorage
```

实际来自基类 `MmaBase`。

在 [mma_base.h](/data/cutlass/include/cutlass/gemm/threadblock/mma_base.h) 中，`MmaBase::SharedStorage` 大致包含：

```cpp
class SharedStorage {
public:
  using ShapeA = MatrixShape<...>;
  using ShapeB = MatrixShape<...>;

  AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;
  AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;
};
```

也就是说，`Mma::SharedStorage` 里面主要就是：

- `operand_A`：A tile 的 shared memory buffer。
- `operand_B`：B tile 的 shared memory buffer。

对于 multistage mainloop，`ShapeA` 和 `ShapeB` 中还会乘上 `kStages`，因为 shared memory 里要放多个 pipeline stage 的 tile。

### 2.2 `main_loop` 传参时内存已经分配了吗

是的，从 CUDA kernel 的角度看，传给 `ImplicitGemmConvolution::operator()` 时，dynamic shared memory 已经由 CUDA runtime 为当前 CTA 分配好了。

CUTLASS 的 device kernel wrapper 在 [device_kernel.h](/data/cutlass/include/cutlass/device_kernel.h) 中：

```cpp
extern __shared__ int SharedStorageBase[];

typename Operator::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

op(params, *shared_storage);
```

这里的逻辑是：

1. CUDA launch 时第三个参数指定 dynamic shared memory 字节数。
2. device 端通过 `extern __shared__` 拿到这块 CTA-local shared memory。
3. CUTLASS 把这块裸 shared memory reinterpret 成 `Operator::SharedStorage *`。
4. 然后把 `*shared_storage` 传给 kernel operator。

host 端 launch 时会计算 shared memory 大小，形态类似：

```cpp
int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

cutlass::Kernel<UnderlyingKernel>
  <<<grid, block, smem_size, stream>>>(params_);
```

因此，`main_loop` 不是 C++ 普通栈对象分配出来的一块内存，而是 CUDA dynamic shared memory 的 typed view。

更准确地说：

```text
CUDA runtime 分配 CTA shared memory
  -> device_kernel.h 通过 extern __shared__ 拿到裸地址
  -> reinterpret_cast 成 Operator::SharedStorage*
  -> ImplicitGemmConvolution::operator(params, *shared_storage)
  -> shared_storage.main_loop
  -> Mma constructor
  -> operand_A_ref() / operand_B_ref()
  -> shared memory iterator
```

## 3. 为什么 `MmaBase::LayoutA()` 最终就是 `SmemLayoutA`

关键要看模板参数是怎么一路传下去的。

在 [default_mma_core_sm80.h](/data/cutlass/include/cutlass/gemm/threadblock/default_mma_core_sm80.h) 中，shared memory layout A 被定义为：

```cpp
using SmemLayoutA = layout::RowMajorTensorOpMultiplicandCrosswise<
    sizeof_bits<ElementA>::value,
    Shape::kK>;
```

然后它被传给 warp-level MMA：

```cpp
using MmaTensorOp = typename cutlass::gemm::warp::DefaultMmaTensorOp<
    WarpShape,
    InstructionShape,
    ElementA,
    SmemLayoutA,
    ElementB,
    SmemLayoutB,
    ElementC,
    LayoutC,
    Operator,
    WarpCount::kK>::Type;
```

注意这里的关键位置：

```cpp
ElementA, SmemLayoutA,
ElementB, SmemLayoutB,
```

也就是说，`SmemLayoutA` 被作为 warp MMA 的 `LayoutA` 模板实参传下去了。

### 3.1 `DefaultMmaTensorOp` 怎么继续传

在 [default_mma_tensor_op.h](/data/cutlass/include/cutlass/gemm/warp/default_mma_tensor_op.h) 中：

```cpp
using Type = cutlass::gemm::warp::MmaTensorOp<
    WarpShape_,
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    Policy>;
```

这里的 `LayoutA` 就是刚才传入的 `SmemLayoutA`。

所以：

```text
DefaultMmaCore::SmemLayoutA
  -> DefaultMmaTensorOp<..., ElementA, SmemLayoutA, ...>
  -> warp::MmaTensorOp<..., ElementA, LayoutA, ...>
```

### 3.2 `warp::MmaTensorOp` 怎么保存这个类型

在 [mma_tensor_op.h](/data/cutlass/include/cutlass/gemm/warp/mma_tensor_op.h) 中，`MmaTensorOp` 内部会把模板参数保存成类型别名：

```cpp
using LayoutA = LayoutA_;
```

因为 `LayoutA_` 是从上层传入的 `SmemLayoutA`，所以：

```text
warp::MmaTensorOp::LayoutA == SmemLayoutA
```

### 3.3 `MmaBase::LayoutA()` 为什么用的是这个 Layout

在 [mma_base.h](/data/cutlass/include/cutlass/gemm/threadblock/mma_base.h) 中：

```cpp
using Operator = typename Policy::Operator;
```

这里的 `Operator` 就是 warp-level MMA operator，通常也就是上面得到的 `MmaTensorOp`。

然后：

```cpp
static typename Operator::LayoutA LayoutA() {
  return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
}
```

所以这个函数返回的类型是：

```cpp
typename Operator::LayoutA
```

而前面已经确定：

```text
Operator::LayoutA == warp::MmaTensorOp::LayoutA == SmemLayoutA
```

因此：

```text
MmaBase::LayoutA() 返回的 layout 类型
  == Operator::LayoutA
  == MmaTensorOp::LayoutA
  == DefaultMmaCore 里定义的 SmemLayoutA
```

这就是为什么可以确定：

```cpp
TensorRefA{operand_A.data(), LayoutA()}
```

里面的 `LayoutA()` 最终就是 `SmemLayoutA`。

## 4. 把三个问题串起来

可以用下面这条主线理解：

```text
DefaultConv2dFprop
  -> DefaultMmaCore
    -> 定义 SmemLayoutA = RowMajorTensorOpMultiplicandCrosswise<...>

  -> DefaultMmaTensorOp
    -> 把 SmemLayoutA 作为 warp::MmaTensorOp::LayoutA

  -> MmaBase
    -> Operator = Policy::Operator
    -> Operator::LayoutA == SmemLayoutA
    -> LayoutA() 返回 SmemLayoutA::packed(...)

  -> MmaBase::SharedStorage
    -> operand_A / operand_B 是 shared memory buffer
    -> operand_A_ref() 返回 TensorRefA{operand_A.data(), LayoutA()}

  -> ImplicitGemmConvolution::operator()
    -> Mma mma(shared_storage.main_loop, ...)

  -> ImplicitGemmMultistage constructor
    -> smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx)

  -> RegularTileAccessIterator
    -> 根据 thread_idx 得到 thread_offset
    -> 调用 TensorRef::offset(thread_offset)

  -> TensorRef::offset
    -> 调用 layout_(coord)

  -> RowMajorTensorOpMultiplicandCrosswise::operator()
    -> MatrixCoord(row, column)
    -> PitchLinearCoord(contiguous = column, strided = row)

  -> TensorOpMultiplicand::operator()
    -> 执行 XOR swizzle
    -> 返回 shared memory 线性 offset
```

一句话总结：

`implicit_gemm_convolution.h` 中看不到 swizzle，是因为它只创建并调用 `Mma`；真正的 swizzle 隐藏在 `Mma` 构造 shared-memory iterator 时使用的 `TensorRef + SmemLayoutA` 里。`SmemLayoutA` 通过 `DefaultMmaCore -> DefaultMmaTensorOp -> MmaTensorOp -> MmaBase::Operator::LayoutA` 一路传递，最后在 `MmaBase::operand_A_ref()` 中变成 shared memory `TensorRef` 的 layout。
