# CUTLASS DefaultConv2dFprop 默认参数与偏特化匹配分析

## 1. 分析目标

本文从下面这个测试文件中的 kernel 实例化开始分析：

```text
test/unit/conv/device/
conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu
```

重点回答以下问题：

1. `AlignmentA = 8` 是怎么得到的？
2. 调用 `DefaultConv2dFprop` 时没有显式传入 `StrideSupport`、`AlignmentA`、
   `AlignmentB`，它们从哪里来？
3. 为什么这个实例最终选择
   `default_conv2d_fprop.h` 第 1101 行附近的偏特化？
4. C++ 类模板偏特化的匹配原则是什么？
5. `AlignmentA` 和 epilogue 中的向量长度是否是同一个参数？

---

## 2. 从测试文件第 100 行开始

测试文件第 100 行附近的实例化为：

```cpp
using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultConv2dFprop<
        ElementA,
        cutlass::layout::TensorNHWC,

        ElementB,
        cutlass::layout::TensorNHWC,

        ElementC,
        cutlass::layout::TensorNHWC,

        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,

        cutlass::gemm::GemmShape<128, 128, 64>,
        cutlass::gemm::GemmShape<64, 64, 64>,
        cutlass::gemm::GemmShape<16, 8, 16>,

        cutlass::epilogue::thread::LinearCombination<
            ElementC,
            128 / cutlass::sizeof_bits<ElementC>::value,
            ElementAccumulator,
            ElementCompute
        >,

        cutlass::gemm::threadblock::
            GemmIdentityThreadblockSwizzle<>,

        3,
        cutlass::arch::OpMultiplyAdd,
        cutlass::conv::IteratorAlgorithm::kOptimized
    >::Kernel;
```

相关元素类型为：

```cpp
using ElementA           = cutlass::half_t;
using ElementB           = cutlass::half_t;
using ElementC           = cutlass::half_t;
using ElementAccumulator = cutlass::half_t;
using ElementCompute     = cutlass::half_t;
```

调用处最后一个显式模板实参是：

```cpp
cutlass::conv::IteratorAlgorithm::kOptimized
```

也就是说，调用处显式提供了前 17 个模板实参。

---

## 3. DefaultConv2dFprop 主模板声明

`DefaultConv2dFprop` 的主模板声明位于：

```text
include/cutlass/conv/kernel/default_conv2d_fprop.h
```

定义形式为：

```cpp
template <
    typename ElementA,             // 1
    typename LayoutA,              // 2
    typename ElementB,             // 3
    typename LayoutB,              // 4
    typename ElementC,             // 5
    typename LayoutC,              // 6
    typename ElementAccumulator,   // 7
    typename OperatorClass,        // 8
    typename ArchTag,              // 9
    typename ThreadblockShape,     // 10
    typename WarpShape,            // 11
    typename InstructionShape,     // 12
    typename EpilogueOutputOp,     // 13
    typename ThreadblockSwizzle,   // 14
    int Stages,                    // 15
    typename MathOperatorTag,      // 16

    conv::IteratorAlgorithm IteratorAlgorithm =
        IteratorAlgorithm::kOptimized,             // 17

    conv::StrideSupport StrideSupport =
        StrideSupport::kUnity,                     // 18

    int AlignmentA =
        128 / cutlass::sizeof_bits<ElementA>::value, // 19

    int AlignmentB =
        128 / cutlass::sizeof_bits<ElementB>::value  // 20
>
struct DefaultConv2dFprop;
```

模板一共有 20 个参数。

最后四个参数带有默认值：

```text
17. IteratorAlgorithm
18. StrideSupport
19. AlignmentA
20. AlignmentB
```

调用者可以从尾部开始省略这些参数。

---

## 4. 调用处到底传了多少个参数

测试文件第 100 行的调用显式传入：

| 编号 | 主模板参数 | 测试文件传入的实参 |
|---:|---|---|
| 1 | `ElementA` | `cutlass::half_t` |
| 2 | `LayoutA` | `layout::TensorNHWC` |
| 3 | `ElementB` | `cutlass::half_t` |
| 4 | `LayoutB` | `layout::TensorNHWC` |
| 5 | `ElementC` | `cutlass::half_t` |
| 6 | `LayoutC` | `layout::TensorNHWC` |
| 7 | `ElementAccumulator` | `cutlass::half_t` |
| 8 | `OperatorClass` | `arch::OpClassTensorOp` |
| 9 | `ArchTag` | `arch::Sm80` |
| 10 | `ThreadblockShape` | `GemmShape<128,128,64>` |
| 11 | `WarpShape` | `GemmShape<64,64,64>` |
| 12 | `InstructionShape` | `GemmShape<16,8,16>` |
| 13 | `EpilogueOutputOp` | `LinearCombination<...>` |
| 14 | `ThreadblockSwizzle` | `GemmIdentityThreadblockSwizzle<>` |
| 15 | `Stages` | `3` |
| 16 | `MathOperatorTag` | `arch::OpMultiplyAdd` |
| 17 | `IteratorAlgorithm` | `IteratorAlgorithm::kOptimized` |

调用处没有显式提供：

```text
18. StrideSupport
19. AlignmentA
20. AlignmentB
```

因此编译器从主模板声明中补上默认值。

---

## 5. 省略的三个参数如何补齐

### 5.1 StrideSupport

主模板声明：

```cpp
conv::StrideSupport StrideSupport =
    StrideSupport::kUnity
```

因此：

```text
StrideSupport = conv::StrideSupport::kUnity
```

`kUnity` 表示该 kernel 的 iterator specialization 面向 unit-stride
卷积场景。

这里需要注意：

```text
StrideSupport 是 kernel 实现能力/特化选择参数，
不是当前某个 Conv2dProblemSize 的 stride_h/stride_w 数值本身。
```

---

### 5.2 AlignmentA

主模板声明：

```cpp
int AlignmentA =
    128 / cutlass::sizeof_bits<ElementA>::value
```

当前：

```cpp
ElementA = cutlass::half_t
```

`half_t` 的定义中使用一个 `uint16_t` 保存数据：

```cpp
struct alignas(2) half_t {
  uint16_t storage;
};
```

所以：

```text
sizeof(half_t) = 2 bytes
sizeof_bits<half_t>::value = sizeof(half_t) × 8
                           = 2 × 8
                           = 16 bits
```

代入默认表达式：

```text
AlignmentA = 128 / 16
           = 8
```

因此：

```text
AlignmentA = 8 个 half 元素
```

其总访问宽度为：

```text
8 × 16 bits = 128 bits = 16 bytes
```

---

### 5.3 AlignmentB

主模板声明：

```cpp
int AlignmentB =
    128 / cutlass::sizeof_bits<ElementB>::value
```

当前：

```cpp
ElementB = cutlass::half_t
```

所以：

```text
AlignmentB = 128 / 16
           = 8
```

---

## 6. 补齐默认参数后的完整类型

虽然调用处只写到 `IteratorAlgorithm::kOptimized`，但是在模板默认参数补齐后，
编译器看到的完整模板实参等价于：

```cpp
DefaultConv2dFprop<
    cutlass::half_t,                         // ElementA
    cutlass::layout::TensorNHWC,             // LayoutA

    cutlass::half_t,                         // ElementB
    cutlass::layout::TensorNHWC,             // LayoutB

    cutlass::half_t,                         // ElementC
    cutlass::layout::TensorNHWC,             // LayoutC

    cutlass::half_t,                         // ElementAccumulator

    cutlass::arch::OpClassTensorOp,          // OperatorClass
    cutlass::arch::Sm80,                     // ArchTag

    cutlass::gemm::GemmShape<128,128,64>,    // ThreadblockShape
    cutlass::gemm::GemmShape<64,64,64>,      // WarpShape
    cutlass::gemm::GemmShape<16,8,16>,       // InstructionShape

    EpilogueOutputOp,                        // EpilogueOutputOp
    GemmIdentityThreadblockSwizzle<>,        // ThreadblockSwizzle

    3,                                       // Stages
    cutlass::arch::OpMultiplyAdd,            // MathOperatorTag

    IteratorAlgorithm::kOptimized,           // IteratorAlgorithm
    StrideSupport::kUnity,                   // 默认补齐
    8,                                       // AlignmentA，默认计算
    8                                        // AlignmentB，默认计算
>
```

这是后续进行偏特化匹配时真正使用的完整参数列表。

---

## 7. 重要原则：默认参数不是由偏特化提供的

默认模板实参写在主模板声明上：

```cpp
template <..., int AlignmentA = ..., int AlignmentB = ...>
struct DefaultConv2dFprop;
```

偏特化中的：

```cpp
int AlignmentA,
int AlignmentB
```

不是在重新声明默认值。

例如第 1101 行偏特化头部：

```cpp
template <
    ...,
    conv::StrideSupport StrideSupport,
    int AlignmentA,
    int AlignmentB
>
struct DefaultConv2dFprop<...>;
```

这里的 `AlignmentA`、`AlignmentB` 是从已经补齐的完整主模板实参中推导出来的。

流程是：

```text
调用者提供部分模板实参
        ↓
主模板声明补齐省略的尾部默认实参
        ↓
形成完整的 20 个模板实参
        ↓
用完整实参匹配所有可用偏特化
        ↓
选择最特化的合法匹配
```

不是：

```text
先选偏特化，再让偏特化补默认参数
```

---

## 8. 为什么匹配第 1101 行偏特化

第 1101 行附近的偏特化模式可以简化为：

```cpp
template <
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    typename ArchTag,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename EpilogueOutputOp,
    typename ThreadblockSwizzle,
    int Stages,
    typename MathOperatorTag,
    conv::StrideSupport StrideSupport,
    int AlignmentA,
    int AlignmentB
>
struct DefaultConv2dFprop<
    ElementA,
    LayoutA,
    ElementB,
    LayoutB,
    ElementC,
    LayoutC,
    ElementAccumulator,

    arch::OpClassTensorOp,             // 固定条件 1

    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    MathOperatorTag,

    IteratorAlgorithm::kOptimized,     // 固定条件 2

    StrideSupport,
    AlignmentA,
    AlignmentB
> {
  // ...
};
```

这个偏特化固定了两个关键模板实参：

```text
OperatorClass     = arch::OpClassTensorOp
IteratorAlgorithm = IteratorAlgorithm::kOptimized
```

测试文件补齐后的完整实参正好满足：

```text
OperatorClass     = arch::OpClassTensorOp
IteratorAlgorithm = IteratorAlgorithm::kOptimized
```

其余参数都可以被偏特化自己的模板参数推导出来：

```text
ArchTag       → arch::Sm80
Stages        → 3
StrideSupport → kUnity
AlignmentA    → 8
AlignmentB    → 8
LayoutA       → TensorNHWC
LayoutB       → TensorNHWC
```

因此第 1101 行偏特化可以匹配。

---

## 9. 为什么不是第 112 行 Analytic 偏特化

第 112 行偏特化固定：

```cpp
IteratorAlgorithm::kAnalytic
```

而测试实例传入：

```cpp
IteratorAlgorithm::kOptimized
```

比较如下：

```text
偏特化要求：IteratorAlgorithm = kAnalytic
实际实参：  IteratorAlgorithm = kOptimized
```

两者不同，因此第 112 行偏特化匹配失败。

测试文件前面的 analytic 测试：

```cpp
DefaultConv2dFprop<
    ...,
    IteratorAlgorithm::kAnalytic
>
```

才会选择 analytic 对应的偏特化。

---

## 10. 为什么不是 FixedChannels 偏特化

FixedChannels 偏特化固定：

```cpp
IteratorAlgorithm::kFixedChannels
```

实际实参是：

```cpp
IteratorAlgorithm::kOptimized
```

因此不匹配。

---

## 11. 为什么不是 FewChannels 偏特化

FewChannels 偏特化固定：

```cpp
IteratorAlgorithm::kFewChannels
```

实际实参是：

```cpp
IteratorAlgorithm::kOptimized
```

因此不匹配。

---

## 12. 为什么不是 SIMT 偏特化

SIMT 偏特化固定：

```cpp
OperatorClass = arch::OpClassSimt
```

实际实参是：

```cpp
OperatorClass = arch::OpClassTensorOp
```

因此不匹配。

当前测试名称中的：

```text
tensor_op
```

也对应其显式传入的：

```cpp
cutlass::arch::OpClassTensorOp
```

---

## 13. 为什么不是 interleaved-layout 偏特化

interleaved 偏特化要求 A/B layout 具有下面的具体模式：

```cpp
LayoutA = layout::TensorNCxHWx<InterleavedK>
LayoutB = layout::TensorCxRSKx<InterleavedK>
```

当前测试传入：

```cpp
LayoutA = layout::TensorNHWC
LayoutB = layout::TensorNHWC
```

`TensorNHWC` 不能匹配：

```cpp
TensorNCxHWx<InterleavedK>
```

也不能匹配：

```cpp
TensorCxRSKx<InterleavedK>
```

因此 interleaved-layout 偏特化匹配失败。

---

## 14. 为什么不是两阶段 Stages=2 偏特化

另一个 optimized TensorOp 偏特化把 stage 数固定为：

```cpp
Stages = 2
```

它的模式类似：

```cpp
DefaultConv2dFprop<
    ...,
    arch::OpClassTensorOp,
    ...,
    2,
    MathOperatorTag,
    IteratorAlgorithm::kOptimized,
    ...
>
```

当前测试传入：

```text
Stages = 3
```

所以：

```text
偏特化要求：Stages = 2
实际实参：  Stages = 3
```

该偏特化不能匹配。

第 1101 行偏特化则使用：

```cpp
int Stages
```

它可以推导为：

```text
Stages = 3
```

所以第 1101 行版本匹配成功。

---

## 15. 匹配过程汇总

当前完整实参中的关键条件是：

```text
OperatorClass     = OpClassTensorOp
IteratorAlgorithm = kOptimized
Stages            = 3
LayoutA           = TensorNHWC
LayoutB           = TensorNHWC
StrideSupport     = kUnity
AlignmentA        = 8
AlignmentB        = 8
```

候选偏特化比较：

| 偏特化类型 | 匹配结果 | 原因 |
|---|---|---|
| TensorOp + Analytic | 不匹配 | iterator 要求 `kAnalytic` |
| TensorOp + FixedChannels | 不匹配 | iterator 要求 `kFixedChannels` |
| TensorOp + FewChannels | 不匹配 | iterator 要求 `kFewChannels` |
| TensorOp + Optimized + interleaved | 不匹配 | layout 不是 `TensorNCxHWx/TensorCxRSKx` |
| TensorOp + Optimized + Stages=2 | 不匹配 | 当前 `Stages=3` |
| SIMT + Optimized | 不匹配 | OperatorClass 不是 `OpClassSimt` |
| TensorOp + Optimized + generic Stages/layout | 匹配 | 所有固定条件都满足 |

因此最终选择：

```text
default_conv2d_fprop.h 第 1101 行附近的偏特化
```

---

## 16. C++ 类模板偏特化的匹配原则

可以用下面四步理解。

### 第一步：处理主模板默认实参

调用：

```cpp
DefaultConv2dFprop<A, B, ..., kOptimized>
```

省略了尾部参数，因此先使用主模板声明中的默认值补齐。

补齐后形成完整模板参数列表。

---

### 第二步：检查每个偏特化是否可以匹配

编译器把完整实参代入每个偏特化模式。

偏特化中的普通模板参数可以推导，例如：

```cpp
typename LayoutA
int Stages
int AlignmentA
```

偏特化中写死的类型或数值必须完全相等，例如：

```cpp
arch::OpClassTensorOp
IteratorAlgorithm::kOptimized
Stages = 2
layout::TensorNCxHWx<InterleavedK>
```

只要某个固定条件不满足，该偏特化就不是候选。

---

### 第三步：如果有多个候选，选择更特化的版本

如果多个偏特化都能匹配，编译器执行 partial ordering，选择约束更具体的版本。

例如理论上下面两个模式都可能接受某些实例：

```cpp
// 泛化版本：Stages 可以是任何整数
template <int Stages>
struct X<OpClassTensorOp, Stages, kOptimized>;

// 更具体版本：Stages 必须为 2
struct X<OpClassTensorOp, 2, kOptimized>;
```

当实际 `Stages=2` 时，固定 `Stages=2` 的版本通常更特化。

当实际 `Stages=3` 时，固定 `Stages=2` 的版本根本不能匹配，只剩泛化版本。

在当前实例中：

```text
Stages = 3
```

所以两阶段版本不会进入最终候选集合。

---

### 第四步：无法唯一选择时编译报错

可能出现三种最终情况：

1. 恰好一个最合适的偏特化：使用它。
2. 没有偏特化匹配：尝试使用主模板定义。
3. 多个匹配且无法判断谁更特化：产生 ambiguous partial specialization 编译错误。

当前 `DefaultConv2dFprop` 的主模板只有声明：

```cpp
struct DefaultConv2dFprop;
```

没有通用实现体。

因此，如果没有任何偏特化匹配，而代码又尝试访问：

```cpp
DefaultConv2dFprop<...>::Kernel
```

通常会因为类型不完整或不存在 `Kernel` 而编译失败。

---

## 17. AlignmentA=8 后如何传入 activation iterator

选中第 1101 行偏特化后，其中定义：

```cpp
using AccessTypeA =
    cutlass::AlignedArray<ElementA, AlignmentA>;
```

代入当前值：

```cpp
using AccessTypeA =
    cutlass::AlignedArray<cutlass::half_t, 8>;
```

然后传给 activation iterator：

```cpp
using IteratorA =
    Conv2dFpropActivationTileAccessIteratorOptimized<
        MatrixShape<
            ThreadblockShape::kM,
            ThreadblockShape::kK
        >,
        ElementA,
        LayoutA,
        ThreadMapA,
        AccessTypeA
    >;
```

因此在：

```text
conv2d_fprop_activation_tile_access_iterator_optimized.h
```

中：

```cpp
using AccessType = AccessType_;
```

最终等价于：

```cpp
using AccessType =
    cutlass::AlignedArray<cutlass::half_t, 8>;
```

所以：

```text
AccessType::kElements = 8
```

---

## 18. ThreadMap::kElementsPerAccess 为什么也是 8

第 1101 行偏特化使用：

```cpp
using MmaCore =
    cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        ElementA,
        layout::RowMajor,
        ElementB,
        layout::ColumnMajor,
        ElementAccumulator,
        layout::RowMajor,
        arch::OpClassTensorOp,
        Stages,
        MathOperatorTag
    >;
```

当前 SM80 half TensorOp 路径中：

```cpp
using IteratorThreadMapA =
    PitchLinearWarpRakedThreadMap<
        ...,
        kAccessSizeInBits /
            sizeof_bits<ElementA>::value
    >;
```

其中：

```text
kAccessSizeInBits = 128
sizeof_bits<ElementA>::value = 16
```

所以：

```text
ThreadMapA::kElementsPerAccess = 128 / 16 = 8
```

最终 activation iterator 中：

```cpp
kAccessesPerVector =
    ThreadMap::kElementsPerAccess /
    AccessType::kElements
```

代入：

```text
ThreadMap::kElementsPerAccess = 8
AccessType::kElements         = 8
```

得到：

```text
kAccessesPerVector = 8 / 8 = 1
```

含义是：

```text
ThreadMap 给当前线程分配的一个 8-half 逻辑向量，
恰好可以通过一次 AlignedArray<half,8> 内存访问完成。
```

---

## 19. 不要把 epilogue 向量长度误认为 AlignmentA

测试实例中还有：

```cpp
cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementCompute
>
```

其中第二个参数也等于：

```text
128 / 16 = 8
```

但是这个 `8` 和 `AlignmentA=8` 不是同一个模板参数。

它们只是因为 A、C 都是 `half_t`，并且两处都选择了 128-bit 向量宽度，
所以数值碰巧相同。

二者语义如下：

| 参数 | 所属阶段 | 含义 |
|---|---|---|
| `AlignmentA` | mainloop global load | activation/A 一次向量访问的元素数 |
| `AlignmentB` | mainloop global load | filter/B 一次向量访问的元素数 |
| `LinearCombination` 第二参数 | epilogue | 输出 C/D 每次 epilogue 向量操作的元素数 |

不能通过：

```cpp
LinearCombination<ElementC, 8, ...>
```

直接推导：

```text
AlignmentA = 8
```

`AlignmentA=8` 的可靠来源是主模板默认实参：

```cpp
int AlignmentA =
    128 / sizeof_bits<ElementA>::value
```

---

## 20. 同一测试文件中的 Alignment=2 对照实例

同一个测试文件后面有显式 alignment 为 2 的实例：

```cpp
DefaultConv2dFprop<
    ...,
    IteratorAlgorithm::kOptimized,
    StrideSupport::kStrided,
    2,
    2
>
```

这里显式提供：

```text
StrideSupport = kStrided
AlignmentA    = 2
AlignmentB    = 2
```

所以不再使用第 18 至 20 个参数的默认值。

完整关键参数变为：

```text
OperatorClass     = OpClassTensorOp
IteratorAlgorithm = kOptimized
Stages            = 3
LayoutA/B         = TensorNHWC
StrideSupport     = kStrided
AlignmentA/B      = 2
```

它仍然匹配第 1101 行偏特化，因为该偏特化中的：

```cpp
StrideSupport
AlignmentA
AlignmentB
```

都是可推导参数，并没有固定为 `kUnity` 或 `8`。

选中偏特化后：

```cpp
using AccessTypeA =
    AlignedArray<half_t, 2>;
```

于是：

```text
AccessType::kElements = 2
ThreadMap::kElementsPerAccess = 8
kAccessesPerVector = 8 / 2 = 4
```

也就是 ThreadMap 分配的 8 个 half 被拆成 4 次访问，每次读取 2 个 half。

---

## 21. 同一测试文件中的 Alignment=4 对照实例

文件中还有：

```cpp
DefaultConv2dFprop<
    ...,
    IteratorAlgorithm::kOptimized,
    StrideSupport::kStrided,
    4,
    4
>
```

这时：

```text
AlignmentA = 4
AccessTypeA = AlignedArray<half_t,4>
AccessType::kElements = 4
```

因此：

```text
kAccessesPerVector = 8 / 4 = 2
```

ThreadMap 的一个 8-half 逻辑向量被拆成两次实际 load，每次读取 4 个 half。

对照关系：

| `AlignmentA` | `AccessTypeA` | `AccessType::kElements` | `kAccessesPerVector` |
|---:|---|---:|---:|
| 8，默认 | `AlignedArray<half,8>` | 8 | 1 |
| 4，显式 | `AlignedArray<half,4>` | 4 | 2 |
| 2，显式 | `AlignedArray<half,2>` | 2 | 4 |

---

## 22. 为什么模板参数必须从尾部省略

C++ 模板默认实参和普通函数默认实参类似。

如果后面的参数依赖默认值，就必须从尾部连续省略。

当前参数顺序是：

```text
IteratorAlgorithm
StrideSupport
AlignmentA
AlignmentB
```

因此下面是合法的：

```cpp
DefaultConv2dFprop<
    ...,
    MathOperatorTag
>
```

四个尾部参数全部采用默认值。

下面也是合法的：

```cpp
DefaultConv2dFprop<
    ...,
    MathOperatorTag,
    IteratorAlgorithm::kOptimized
>
```

后三个参数采用默认值。

如果要显式指定 `AlignmentA`，就不能跳过位于它前面的 `StrideSupport`：

```cpp
DefaultConv2dFprop<
    ...,
    MathOperatorTag,
    IteratorAlgorithm::kOptimized,
    StrideSupport::kStrided,
    4,
    4
>
```

这解释了为什么 alignment=2/4 的测试必须同时写出：

```cpp
StrideSupport::kStrided
```

即使关注点主要是 alignment，也必须按照模板参数顺序填写前置参数。

---

## 23. 如何阅读这种 CUTLASS 模板调用

以后遇到类似实例化，可以按下面步骤追踪。

### 步骤一：找到主模板声明

不要先从某个偏特化猜参数。

先找到：

```cpp
template <...>
struct DefaultConv2dFprop;
```

记录完整模板参数顺序和默认值。

### 步骤二：给调用实参编号

把调用处的实参逐个对应到：

```text
1, 2, 3, ..., N
```

避免因为 CUTLASS 模板参数很多而发生错位。

### 步骤三：补齐默认实参

根据主模板声明补齐调用处省略的尾部参数。

### 步骤四：写出关键完整参数

尤其关注偏特化常用的固定条件：

```text
OperatorClass
LayoutA/LayoutB
Stages
IteratorAlgorithm
ArchTag
```

### 步骤五：逐个排除偏特化

检查偏特化模式中固定的类型或常量是否与实际实参完全相同。

### 步骤六：进入最终偏特化继续展开 using

例如：

```text
AlignmentA
  ↓
AccessTypeA = AlignedArray<ElementA, AlignmentA>
  ↓
IteratorA 的 AccessType_
  ↓
AccessType::kElements
  ↓
kAccessesPerVector
```

---

## 24. 本例最终推导链

```text
测试文件第 100 行
DefaultConv2dFprop<..., kOptimized>
        ↓
显式提供前 17 个模板实参
        ↓
主模板补齐：
  StrideSupport = kUnity
  AlignmentA = 128 / sizeof_bits<half_t> = 8
  AlignmentB = 128 / sizeof_bits<half_t> = 8
        ↓
完整关键参数：
  OpClassTensorOp
  TensorNHWC / TensorNHWC
  Stages = 3
  IteratorAlgorithm = kOptimized
  StrideSupport = kUnity
  AlignmentA/B = 8
        ↓
排除：
  Analytic
  FixedChannels
  FewChannels
  interleaved layout
  SIMT
  Stages=2
        ↓
选择 default_conv2d_fprop.h 第 1101 行偏特化
        ↓
AccessTypeA = AlignedArray<half_t,8>
        ↓
AccessType::kElements = 8
        ↓
ThreadMap::kElementsPerAccess = 8
        ↓
kAccessesPerVector = 8 / 8 = 1
```

---

## 25. 最终结论

### AlignmentA=8 的来源

不是从测试调用处显式传入，也不是从 epilogue 参数猜出来的。

它来自主模板声明中的默认实参：

```cpp
int AlignmentA =
    128 / sizeof_bits<ElementA>::value
```

当前：

```text
ElementA = half_t
sizeof_bits<half_t> = 16
AlignmentA = 128 / 16 = 8
```

### 未传 StrideSupport/AlignmentA/AlignmentB 为什么仍有值

因为它们是主模板的尾部默认模板实参：

```text
StrideSupport = kUnity
AlignmentA = 8
AlignmentB = 8
```

### 为什么选择第 1101 行偏特化

因为补齐后的完整模板实参满足：

```text
OperatorClass     = OpClassTensorOp
IteratorAlgorithm = kOptimized
Stages            = 3
LayoutA/B         = TensorNHWC
```

它匹配第 1101 行的 TensorOp + Optimized + 通用 Stages/layout 偏特化，
而其他偏特化至少有一个固定条件不满足。

### 模板匹配的核心顺序

```text
主模板默认参数补齐
        ↓
检查所有偏特化是否可匹配
        ↓
选择最特化的唯一候选
        ↓
实例化该偏特化的 Kernel/Iterator/Mma 类型
```
