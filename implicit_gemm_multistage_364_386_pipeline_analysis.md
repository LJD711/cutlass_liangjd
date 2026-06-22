# CUTLASS `ImplicitGemmMultistage` 第 364-386 行流水线分析

## 1. 分析范围

本文分析：

```text
include/cutlass/conv/threadblock/implicit_gemm_multistage.h
```

中的以下代码：

```cpp
// Pair of fragments used to overlap shared memory loads and math
// instructions
WarpLoadedFragmentA warp_loaded_frag_A[2];
WarpLoadedFragmentB warp_loaded_frag_B[2];
WarpTransformedFragmentA warp_transformed_frag_A[2];
WarpTransformedFragmentB warp_transformed_frag_B[2];

Operator warp_mma;

this->warp_tile_iterator_A_.set_kgroup_index(0);
this->warp_tile_iterator_B_.set_kgroup_index(0);

this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
this->warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);

++this->warp_tile_iterator_A_;
++this->warp_tile_iterator_B_;

// Start issuing the first group of the next stage outside of the mainloop
copy_tiles_and_advance(iterator_A, iterator_B);

int smem_write_stage_idx = Base::kStages - 1;
int smem_read_stage_idx = 0;

warp_mma.transform(
    warp_transformed_frag_A[0],
    warp_transformed_frag_B[0],
    warp_loaded_frag_A[0],
    warp_loaded_frag_B[0]);
```

重点说明：

1. `stage`、software copy group、warp MMA K-group 的区别；
2. 四种 fragment 的来源和作用；
3. 为什么 fragment 数组长度为 2；
4. `set_kgroup_index(0)`、`load()` 和 `operator++()` 分别做什么；
5. 为什么第 381 行只发射下一个 stage 的第一组 copy；
6. 为什么第一组必须放在 mainloop 外；
7. `warp_mma.transform()` 在当前 FP16 路径中的实际行为；
8. 第 386 行执行完以后整个流水线处于什么状态。

---

## 2. 当前分析使用的实例参数

本文沿用下面的 SM80 F16 NHWC align4 测试配置：

```text
test/unit/conv/device/
conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu
```

关键参数：

```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;

using ThreadblockShape =
    cutlass::gemm::GemmShape<128, 128, 64>;

using WarpShape =
    cutlass::gemm::GemmShape<64, 64, 64>;

using InstructionShape =
    cutlass::gemm::GemmShape<16, 8, 16>;

static int const Stages = 3;
static int const AlignmentA = 4;
static int const AlignmentB = 4;
```

因此：

```text
CTA tile             = 128 x 128 x 64
warp tile            = 64 x 64 x 64
Tensor Core MMA      = 16 x 8 x 16
shared-memory stages = 3
ElementA/B           = half，16 bit
global AccessType    = 4 half，64 bit，8 bytes
```

---

## 3. 先区分四个层级

理解第 364-386 行前，必须区分：

```text
CTA K-tile
shared-memory stage
software copy group
warp MMA K-group
```

它们不是同一个概念。

### 3.1 CTA K-tile

当前：

```text
ThreadblockShape::kK = 64
```

一个 CTA mainloop iteration 处理 GEMM K 方向的一个 `K=64` tile。

对 A 和 B 来说，一个 CTA K-tile 包含：

```text
A tile = 128 x 64
B tile = 64 x 128
```

---

### 3.2 Shared-memory stage

一个 stage 是 shared-memory 环形缓冲区中的一个槽位。

当前 `Stages=3`：

```text
shared stage0
shared stage1
shared stage2
```

每个 stage 可以保存一个完整 CTA K-tile：

```text
stageX:
  A = 128 x 64
  B = 64 x 128
```

三个 stage 形成环形缓冲：

```text
stage0 -> stage1 -> stage2 -> stage0
```

---

### 3.3 Software copy group

一个完整 stage 的 global-to-shared copy 不会在 mainloop 中一次性全部发射。

CUTLASS 将它拆成多个软件 copy group：

```text
group0
group1
group2
group3
```

当前 A operand：

```text
ThreadMap::Iterations::kCount = 8
kWarpGemmIterations           = 4
```

所以：

```cpp
kAccessesPerGroupA = ceil(8 / 4) = 2;
```

一个 software copy group 包含：

```text
2 个 ThreadMap 逻辑 access
```

由于 align4：

```text
ThreadMap::kElementsPerAccess = 8 half
AccessType::kElements         = 4 half
kAccessesPerVector            = 8 / 4 = 2
```

一个 ThreadMap 逻辑 access 需要拆成两条 `cp.async`。

因此每线程：

```text
一个 copy group
    = 2 个 ThreadMap access
    = 2 x 2
    = 4 条 cp.async

一个完整 stage
    = 4 个 copy group
    = 16 条 cp.async
```

具体划分：

| Copy group | `group_start_A` | ThreadMap access | Iterator index |
|---:|---:|---|---|
| 0 | 0 | 0、1 | 0-3 |
| 1 | 2 | 2、3 | 4-7 |
| 2 | 4 | 4、5 | 8-11 |
| 3 | 6 | 6、7 | 12-15 |

software copy group 描述的数据路径是：

```text
global memory
    ↓ cp.async
shared memory
```

---

### 3.4 Warp MMA K-group

一个 shared-memory stage 中保存的是 `K=64`。

Tensor Core 指令一次处理：

```text
InstructionShape::kK = 16
```

因此一个 stage 被 warp 计算拆成：

```text
kWarpGemmIterations
    = WarpShape::kK / InstructionShape::kK
    = 64 / 16
    = 4
```

四个 warp MMA K-group：

```text
K-group0 = K[0..15]
K-group1 = K[16..31]
K-group2 = K[32..47]
K-group3 = K[48..63]
```

warp MMA K-group 描述的数据路径是：

```text
shared memory
    ↓ ldmatrix/ldsm
warp registers
    ↓ mma.sync
accumulator registers
```

---

### 3.5 Copy group 与 K-group 的关系

二者都被分成四组，但它们不是同一组数据。

```text
software copy group:
  搬运未来的 CTA K-tile

warp MMA K-group:
  计算当前的 CTA K-tile
```

它们被安排成四组，是为了形成交错：

```text
计算当前 stage K-group0
同时发射未来 stage copy group1

计算当前 stage K-group1
同时发射未来 stage copy group2

计算当前 stage K-group2
同时发射未来 stage copy group3

计算当前 stage K-group3
同时发射再下一个 stage copy group0
```

---

## 4. 执行到第 364 行前的状态

在第 364 行之前，prologue 已执行：

```cpp
for (int stage = 0;
     stage < Base::kStages - 1;
     ++stage) {
  // 完整复制 A stage
  // 完整复制 B stage
  // iterator_A/B.advance()
  // shared write iterator 前进
  // cp_async_fence()
}
```

当前 `Stages=3`，所以 prologue 完整预取两个 stage：

```text
global tile0 -> shared stage0
global tile1 -> shared stage1
```

prologue 结束后：

```text
global iterator_A/B -> tile2
shared write iterator -> stage2
```

随后：

```cpp
cp_async_wait<Base::kStages - 2>();
__syncthreads();
```

当前：

```text
Base::kStages - 2 = 1
```

`wait<1>` 保证最老的已提交 cp.async group 已经完成。

所以 shared stage0 已经可以被 warp 安全读取。

第 364 行开始时可画成：

```text
Global:
  iterator_A/B -> tile2

Shared:
  stage0 = tile0，已完成，可读
  stage1 = tile1，已提交
  stage2 = 空闲，准备写 tile2

Warp registers:
  尚未装载 stage0 的 K-group0
```

---

## 5. Fragment 类型的来源

`ImplicitGemmMultistage` 中定义：

```cpp
using Operator = typename Policy::Operator;

using WarpLoadedFragmentA =
    typename Operator::FragmentA;

using WarpLoadedFragmentB =
    typename Operator::FragmentB;

using WarpTransformedFragmentA =
    typename Operator::TransformedFragmentA;

using WarpTransformedFragmentB =
    typename Operator::TransformedFragmentB;
```

当前 `Operator` 最终来自：

```text
DefaultMmaCore
  -> DefaultMmaTensorOp
  -> warp::MmaTensorOp
```

相关源码：

```text
include/cutlass/gemm/threadblock/default_mma_core_sm80.h
include/cutlass/gemm/warp/default_mma_tensor_op.h
include/cutlass/gemm/warp/mma_tensor_op.h
```

---

## 6. `WarpLoadedFragmentA/B` 是什么

`FragmentA` 和 `FragmentB` 来源于 warp shared-memory iterator：

```cpp
using FragmentA = typename IteratorA::Fragment;
using FragmentB = typename IteratorB::Fragment;
```

这里的 iterator 不是 global iterator。

它们是：

```text
warp_tile_iterator_A_
warp_tile_iterator_B_
```

用于：

```text
shared memory -> warp registers
```

### 6.1 当前 fragment 元素数量

底层 iterator 定义：

```cpp
using Fragment =
    Array<
        Element,
        Shape::kStrided *
        InstructionShape::kContiguous /
        kThreads
    >;
```

对当前 warp：

```text
A K-group tile = 64 x 16
B K-group tile = 16 x 64
warp threads   = 32
```

因此每个线程：

```text
A fragment = 64 x 16 / 32 = 32 half
B fragment = 16 x 64 / 32 = 32 half
```

可理解为：

```cpp
WarpLoadedFragmentA ~= Array<half_t, 32>;
WarpLoadedFragmentB ~= Array<half_t, 32>;
```

这里是每个线程的 fragment，不是整个 warp 的总 fragment。

整个 warp 的 32 个线程合起来保存：

```text
A = 32 x 32 half = 1024 half = 64 x 16
B = 32 x 32 half = 1024 half = 16 x 64
```

正好对应一个 `K=16` 的 warp MMA K-group。

### 6.2 `smem_iterator_A_` 并不保存 shared-memory 数据

这里需要区分三个不同的对象：

| 对象 | 实际作用 | 是否保存 A 数据 |
|---|---|---|
| `shared_storage.operand_A` | shared memory 中真正存放 A tile 的 buffer | 是 |
| `smem_iterator_A_` | 计算 global-to-shared 写入地址 | 否 |
| `warp_tile_iterator_A_` | 计算 shared-to-register 读取地址 | 否 |

真正的数据实体定义在：

```text
include/cutlass/gemm/threadblock/mma_base.h
```

```cpp
AlignedBuffer<
    typename Operator::ElementA,
    ShapeA::kCount
> operand_A;
```

`smem_iterator_A_` 和 `warp_tile_iterator_A_` 都只是带有内部指针、偏移和
迭代状态的地址计算器。它们本身不拥有，也不复制
`shared_storage.operand_A` 中的数据。

因此下面这种理解不准确：

```text
smem_iterator_A_ 中保存数据
    ↓ 把 smem_iterator_A_ 传给 warp iterator
warp_loaded_frag_A
```

准确的数据流是：

```text
global memory
    ↓ cp.async，写地址由 smem_iterator_A_ 计算
shared_storage.operand_A
    ↓ ldmatrix，读地址由 warp_tile_iterator_A_ 计算
warp_loaded_frag_A
```

### 6.3 两个 iterator 为什么会访问同一个 shared-memory buffer

`SharedStorage::operand_A_ref()` 返回：

```cpp
TensorRefA operand_A_ref() {
  return TensorRefA{
      operand_A.data(),
      LayoutA()
  };
}
```

这个 `TensorRef` 包含：

```text
operand_A.data()  : shared-memory buffer 的基地址
LayoutA()         : 从逻辑坐标映射到物理地址的 layout
```

构造 `ImplicitGemmMultistage` 时，C++ 会先构造基类 `Base`：

```cpp
Base(shared_storage, thread_idx, warp_idx, lane_idx)
```

`MmaBase` 在其构造函数中建立 warp 读取 iterator：

```cpp
warp_tile_iterator_A_(
    shared_storage.operand_A_ref(),
    lane_idx)
```

随后再构造当前类自己的 shared-memory 写 iterator：

```cpp
smem_iterator_A_(
    shared_storage.operand_A_ref(),
    thread_idx)
```

两次调用 `operand_A_ref()` 会构造两个独立的 `TensorRef` 值，但它们的
`data()` 都指向同一个：

```text
shared_storage.operand_A.data()
```

可画成：

```text
                         +-----------------------------+
                         | shared_storage.operand_A    |
                         | 同一个 shared-memory buffer |
                         +-----------------------------+
                              ↑                 ↑
                              | write           | read
                              |                 |
                    smem_iterator_A_    warp_tile_iterator_A_
                    CTA thread view       warp lane view
```

所以：

```cpp
this->warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
```

不需要接收 `smem_iterator_A_`。`warp_tile_iterator_A_` 在构造时已经保存了
指向同一 shared buffer 的读取基址和 layout 信息。

### 6.4 两个 iterator 保存的是不同地址状态

虽然二者具有相同的底层 buffer 基地址，但它们不能互相替代。

`smem_iterator_A_` 使用：

```text
thread_idx = CTA 内线程编号
```

它根据 `IteratorThreadMapA` 给 CTA 中每个线程分配 global-to-shared copy
位置，回答的问题是：

```text
当前 CTA 线程的下一条 cp.async 应该写到 shared memory 的哪里？
```

`warp_tile_iterator_A_` 使用：

```text
lane_idx = warp 内 lane 编号，范围 0..31
```

并在 `ImplicitGemmMultistage` 构造函数中加入当前 warp 的 tile 偏移：

```cpp
this->warp_tile_iterator_A_.add_tile_offset(
    {warp_idx_m,
     Base::kWarpGemmIterations * warp_idx_k});
```

它回答的问题是：

```text
当前 warp 的当前 lane 应为 ldmatrix 提供哪个 shared-memory 地址？
```

因此它们共享：

```text
buffer identity
```

但不共享：

```text
thread/lane 映射
iteration index
stage 位置
K-group 位置
byte offset
```

### 6.5 当前实例中 `operand_A` 的实际大小

当前：

```text
ThreadblockShape = <128,128,64>
Stages           = 3
ElementA         = half = 2 bytes
```

忽略当前为 0 的 padding 后：

```cpp
ShapeA =
    MatrixShape<
        Shape::kM,
        Shape::kK * kStages
    >
  = MatrixShape<128, 64 * 3>
  = MatrixShape<128, 192>;
```

因此：

```text
一个 A stage:
  128 x 64 half
  = 8192 half
  = 16384 bytes
  = 16 KiB

三个 A stages:
  128 x 192 half
  = 24576 half
  = 49152 bytes
  = 48 KiB
```

这 48 KiB 才是 shared memory 中真正保存 A 数据的区域。

### 6.6 写入和读取之间依靠同步建立可见性

“两个 iterator 指向同一 buffer”只解决地址关系，还不能自动保证读到的是
已经写完的数据。

prologue 中写入 A 的核心链路是：

```cpp
auto *dst_ptr =
    reinterpret_cast<IteratorA::AccessType *>(
        smem_iterator_A_.get());

cutlass::arch::cp_async_zfill<kSrcBytes>(
    dst_ptr + v,
    iterator_A.get(),
    iterator_A.valid());
```

其中：

```text
iterator_A.get()       : global-memory 源地址
smem_iterator_A_.get() : operand_A 中的 shared-memory 目标地址
```

随后通过：

```cpp
cutlass::arch::cp_async_fence();
cutlass::arch::cp_async_wait<Base::kStages - 2>();
__syncthreads();
```

建立顺序：

```text
发射 cp.async
    ↓
提交 async-copy group
    ↓
等待最老 stage 完成
    ↓
CTA 线程同步
    ↓
warp_tile_iterator_A_ 才能读取 stage0
```

`warp_tile_iterator_A_.load()` 没有接收 `smem_iterator_A_`，但它必须依赖上述
同步协议；否则地址即使相同，也可能读取到尚未完成写入的数据。

---

## 7. `WarpTransformedFragmentA/B` 是什么

`TransformedFragmentA/B` 定义为：

```cpp
using TransformedFragmentA =
    Array<
        typename ArchMmaOperator::ElementA,
        FragmentA::kElements
    >;

using TransformedFragmentB =
    Array<
        typename ArchMmaOperator::ElementB,
        FragmentB::kElements
    >;
```

它们表示：

```text
已经转换成底层 mma.sync 指令所需元素类型和打包格式的 fragment
```

数据流：

```text
shared memory
    ↓ warp iterator load()
WarpLoadedFragment
    ↓ warp_mma.transform()
WarpTransformedFragment
    ↓ warp_mma()
mma.sync
```

当前输入和 Tensor Core 指令都使用：

```text
half_t
```

所以 transformed fragment 元素数量仍然是：

```text
A = 32 half / thread
B = 32 half / thread
```

---

## 8. 第 364-367 行：创建寄存器双缓冲

源码：

```cpp
WarpLoadedFragmentA warp_loaded_frag_A[2];
WarpLoadedFragmentB warp_loaded_frag_B[2];

WarpTransformedFragmentA warp_transformed_frag_A[2];
WarpTransformedFragmentB warp_transformed_frag_B[2];
```

数组长度为 2，表示寄存器双缓冲。

两个槽位：

```text
slot0
slot1
```

在稳定 mainloop 中：

```text
一个槽位保存当前要计算的 K-group
另一个槽位同时加载下一个 K-group
```

例如：

```text
warp_mma_k = 0:
  transformed[0] -> 计算 K-group0
  loaded[1]      <- 加载 K-group1

warp_mma_k = 1:
  transformed[1] -> 计算 K-group1
  loaded[0]      <- 加载 K-group2

warp_mma_k = 2:
  transformed[0] -> 计算 K-group2
  loaded[1]      <- 加载 K-group3

warp_mma_k = 3:
  transformed[1] -> 计算 K-group3
  loaded[0]      <- 加载下一 stage K-group0
```

槽位索引为：

```cpp
当前计算：
warp_mma_k % 2

下一次加载：
(warp_mma_k + 1) % 2
```

---

## 9. 为什么同时需要 loaded 和 transformed 两套数组

可以只设计一套 fragment，但 CUTLASS 分成：

```text
loaded fragment
transformed fragment
```

原因是通用 `MmaTensorOp` 需要支持：

```text
源数据类型 != Tensor Core 指令输入类型
```

例如：

```text
float -> TF32
float -> half
float -> bfloat16
mixed input conversion
complex transform
```

所以逻辑上分为：

```text
load:
  shared memory -> source-type fragment

transform:
  source-type fragment -> instruction-type fragment

mma:
  instruction-type fragment -> accumulator
```

在当前 FP16 到 FP16 路径中，转换很轻，但模板接口仍保持统一。

---

## 10. 第 369 行：构造 warp MMA 操作器

源码：

```cpp
Operator warp_mma;
```

当前 `Operator` 是 warp-level：

```text
MmaTensorOp<WarpShape<64,64,64>, ...>
```

它内部使用的硬件指令为：

```text
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
```

底层定义：

```text
include/cutlass/arch/mma_sm80.h
```

对应：

```cpp
Mma<
    GemmShape<16, 8, 16>,
    32,
    half_t, RowMajor,
    half_t, ColumnMajor,
    half_t, RowMajor,
    OpMultiplyAdd
>
```

一条硬件指令计算：

```text
M = 16
N = 8
K = 16
```

一个 warp K-group 需要计算：

```text
64 x 64 x 16
```

因此一次 `warp_mma()` 内部执行：

```text
M 方向：64 / 16 = 4
N 方向：64 / 8  = 8

总 mma.sync 数 = 4 x 8 = 32
```

这 32 条 `mma.sync` 协同完成当前 warp 的一个 K-group。

---

## 11. 第 371-372 行：设置 K-group0

源码：

```cpp
this->warp_tile_iterator_A_.set_kgroup_index(0);
this->warp_tile_iterator_B_.set_kgroup_index(0);
```

作用是：

```text
通知 A/B warp shared-memory iterator：
当前逻辑状态是本 stage 的 K-group0
```

底层实现：

```cpp
void set_kgroup_index(int k_group) {
  k_group_idx_ =
      k_group %
      (Policy::kGroupsPerTile / kPartitionsK);
}
```

### 11.1 它不会做什么

`set_kgroup_index(0)`：

```text
不会加载 shared memory
不会执行 ldmatrix
不会执行 mma.sync
不会把 iterator 前进到下一个 K-group
```

### 11.2 它真正设置什么

crosswise shared-memory layout 不是简单线性布局。

warp iterator 在 `operator++()` 时需要通过：

```cpp
byte_offset_ ^= ...
```

选择下一个 swizzled 地址。

`k_group_idx_` 用来记录：

```text
当前处于本 stage 的第几个 K-group
```

显式设置为 0 可以：

1. 让 iterator 的内部状态与当前逻辑 K-group 一致；
2. 支持后续 XOR 地址更新；
3. 让编译器在完全展开的循环中进行常量折叠。

---

## 12. 第 374-375 行：加载 stage0 的 K-group0

源码：

```cpp
this->warp_tile_iterator_A_.load(
    warp_loaded_frag_A[0]);

this->warp_tile_iterator_B_.load(
    warp_loaded_frag_B[0]);
```

执行的数据流：

```text
shared stage0 A K-group0
    ↓
warp_loaded_frag_A[0]

shared stage0 B K-group0
    ↓
warp_loaded_frag_B[0]
```

### 12.1 为什么 `load()` 不需要传入 `smem_iterator_A_`

这一行：

```cpp
this->warp_tile_iterator_A_.load(
    warp_loaded_frag_A[0]);
```

可以只传目标 fragment，是因为 `warp_tile_iterator_A_` 已经是一个有状态对象。

构造完成后，它内部已经保存：

```text
pointer_       : 指向 shared_storage.operand_A 的读取基址
stride_        : shared layout 的物理 stride
sections_      : shared buffer 中 crosswise sections 的数量
byte_offset_   : 当前 lane 和当前 K-group 的字节偏移
k_group_idx_   : 当前逻辑 K-group
```

所以 `load()` 的完整输入实际来自两个位置：

```text
显式参数:
  warp_loaded_frag_A[0]      目标寄存器 fragment

iterator 内部状态:
  pointer_ + byte_offset_    shared-memory 源地址
```

`smem_iterator_A_` 的职责已经在更早的 `cp.async` 阶段结束。读取时只需要保证：

1. 两个 iterator 来自同一个 `shared_storage.operand_A_ref()`；
2. shared stage 已通过 `cp_async_wait` 和 `__syncthreads()` 变为可读；
3. `warp_tile_iterator_A_` 当前地址状态指向正确的 stage 和 K-group。

### 12.2 从构造到 `load()` 的对象关系

完整调用关系如下：

```text
ImplicitGemmMultistage(shared_storage, ...)
    |
    +-- Base(shared_storage, ...)
    |     |
    |     +-- warp_tile_iterator_A_(
    |             shared_storage.operand_A_ref(),
    |             lane_idx)
    |
    +-- smem_iterator_A_(
              shared_storage.operand_A_ref(),
              thread_idx)
```

写阶段：

```text
smem_iterator_A_.get()
    ↓ 得到 operand_A 内的目标地址
cp_async_zfill(dst, global_src, valid)
    ↓
shared_storage.operand_A
```

读阶段：

```text
warp_tile_iterator_A_.load(fragment)
    ↓ 使用自己保存的 operand_A 基址
ldmatrix
    ↓
warp_loaded_frag_A[0]
```

这不是：

```text
smem_iterator_A_ -> warp_tile_iterator_A_
```

而是：

```text
smem_iterator_A_  -> 同一个 operand_A buffer <- warp_tile_iterator_A_
```

### 12.3 `load()` 跳转到哪里

矩阵 layout wrapper 最终调用底层 pitch-linear crosswise iterator：

```text
include/cutlass/gemm/warp/
mma_tensor_op_tile_iterator.h
```

wrapper 的：

```cpp
void load(Fragment &frag) const {
  iterator_.load(frag);
}
```

继续进入：

```cpp
void load(Fragment &frag) const {
  load_with_byte_offset(frag, 0);
}
```

因此 A 的调用链是：

```text
warp_tile_iterator_A_.load(warp_loaded_frag_A[0])
    ↓
MmaTensorOpMultiplicandTileIterator::load(Fragment &)
    ↓
load_with_byte_offset(frag, 0)
    ↓
计算 source_ptr 和 source_byte_ptr
    ↓
cutlass::arch::ldsm<RowMajor, 4>()
    ↓
ldmatrix.sync.aligned.x4.m8n8.shared.b16
    ↓
warp_loaded_frag_A[0] 的 lane-local registers
```

### 12.4 当前实例中 `load_with_byte_offset()` 如何形成地址

当前 warp iterator 的关键常量为：

```text
Element                         = half
Layout::kElementsPerAccess      = 8 half
sizeof(AccessType)              = 8 x 2 = 16 bytes
Policy::LdsmShape               = <2,2>
Policy::LdsmShape::kCount       = 4
Policy::LdsmIterations          = <1,4>
Fragment                        = Array<half,32>
sizeof(Fragment)                = 64 bytes / lane
```

构造时：

```cpp
pointer_ =
    reinterpret_cast<AccessType const *>(
        ref.data());
```

其中：

```text
ref.data() = shared_storage.operand_A.data()
```

当前：

```text
ref.stride(0) = 192 half
kCrosswise     = 64
```

所以：

```text
sections_ = 192 / 64 = 3

stride_
  = 192 elements / 8 elements-per-AccessType
  = 24 AccessType
  = 24 x 16 bytes
  = 384 bytes
```

进入 `load_with_byte_offset(frag, 0)` 后：

```cpp
Array<unsigned, 4> *fetch_ptr =
    reinterpret_cast<Array<unsigned, 4> *>(&frag);
```

即把每线程的 64-byte fragment 看成四块：

```text
fetch_ptr[0] = 16 bytes
fetch_ptr[1] = 16 bytes
fetch_ptr[2] = 16 bytes
fetch_ptr[3] = 16 bytes
```

循环范围：

```text
c = 0
s = 0,1,2,3
```

每次计算：

```cpp
source_ptr =
    pointer_ +
    Policy::LdsmShape::kContiguous * c +
    Policy::kLdsmOpInner /
        Layout::kFactor *
        Policy::LdsmShape::kStrided *
        s * stride_;
```

代入当前值：

```text
c                         = 0
Policy::kLdsmOpInner      = 8
Layout::kFactor           = 1
LdsmShape::kStrided       = 2
stride_                   = 24 AccessType
sizeof(AccessType)        = 16 bytes
```

得到：

```text
source_ptr
  = pointer_ + 8 / 1 * 2 * s * 24
  = pointer_ + 384*s AccessType

相邻 s 的字节跨度
  = 384 * 16
  = 6144 bytes
```

然后叠加当前 lane 和当前 K-group 的偏移：

```cpp
source_byte_ptr =
    reinterpret_cast<char const *>(source_ptr) +
    byte_offset +
    byte_offset_;
```

这里：

```text
byte_offset  = load() 调用者给出的额外偏移，当前为 0
byte_offset_ = iterator 构造和 operator++() 维护的 lane/K-group 偏移
```

因此不能只看 `pointer_`。当前 lane 最终提供给 `ldmatrix` 的真实地址是：

```text
operand_A base
+ warp tile offset
+ s iteration offset
+ lane mapping offset
+ current K-group XOR offset
```

### 12.5 一个 lane 0、warp-M0、K-group0 的地址例子

以 A iterator 的：

```text
lane_idx   = 0
warp_idx_m = 0
K-group    = 0
```

为例，构造函数对 lane 0 算得：

```text
access_contiguous = 0
access_strided    = 0
byte_offset_      = 0
```

且 warp-M0 没有额外的 strided tile 偏移。

于是四次 `s` 迭代提供的起始地址相对于 `operand_A.data()` 为：

| `s` | `source_ptr` 偏移 | 加上 lane/K-group 偏移 |
|---:|---:|---:|
| 0 | 0 bytes | 0 bytes |
| 1 | 6144 bytes | 6144 bytes |
| 2 | 12288 bytes | 12288 bytes |
| 3 | 18432 bytes | 18432 bytes |

这些不是 lane 0 独自读取的四个连续矩阵块。每一次都是整个 warp 的 32 个 lane
共同发射一条 `ldmatrix.x4`；每个 lane 提供自己的地址，并接收属于自己的
寄存器分片。

当 iterator 从 K-group0 前进到 K-group1 时，当前实例通过 XOR 修改：

```text
byte_offset_ ^= 32 bytes
```

所以同一个 lane 在下一 K-group 中会根据 crosswise swizzle 访问另一组物理地址，
而不是简单地执行：

```text
address += 32
```

### 12.6 `ldsm<RowMajor,4>` 最终执行什么

真正执行 shared-memory load 的位置是：

```cpp
cutlass::arch::ldsm<
    layout::RowMajor,
    Policy::LdsmShape::kCount
>(
    fetch_ptr[access_idx],
    source_byte_ptr
);
```

它最终进入：

```text
include/cutlass/arch/memory_sm75.h
```

并执行：

```cpp
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
    "{%0, %1, %2, %3}, [%4];"
    ...
);
```

一次 `ldmatrix.x4` 为每个 lane 产生：

```text
4 x uint32
= 16 bytes
= 8 half
```

当前外层 `s` 循环执行四次，所以每 lane 最终得到：

```text
4 x 16 bytes
= 64 bytes
= 32 half
```

正好完整填充：

```cpp
warp_loaded_frag_A[0]
```

### 12.7 为什么不用普通标量 load

shared memory 中的 A/B 已按 Tensor Core crosswise layout 排列。

warp 的 32 个 lane 通过 `ldmatrix` 协作：

```text
从 swizzled shared memory 读取矩阵片段
        ↓
自动分发到每个 lane 的寄存器
        ↓
形成 mma.sync 要求的 lane-register 布局
```

因此：

```text
warp_loaded_frag_A[0]
warp_loaded_frag_B[0]
```

不是普通连续数组意义下的矩阵切片，而是当前 lane 按 Tensor Core 规则持有的
寄存器分片。

完整过程可归纳为：

```text
global A tile
    |
    | iterator_A.get() 给出 global source
    | smem_iterator_A_.get() 给出 shared destination
    v
cp.async
    |
    v
shared_storage.operand_A
    |
    | cp_async_wait + __syncthreads
    |
    | warp_tile_iterator_A_ 已经持有同一 buffer 的读指针
    | lane mapping + warp offset + K-group XOR
    v
ldmatrix.sync.aligned.x4.m8n8.shared.b16
    |
    v
warp_loaded_frag_A[0]
    |
    v
warp_mma.transform()
    |
    v
warp_transformed_frag_A[0]
    |
    v
mma.sync
```

---

## 13. 第 377-378 行：warp iterator 前进到 K-group1

源码：

```cpp
++this->warp_tile_iterator_A_;
++this->warp_tile_iterator_B_;
```

执行前：

```text
iterator -> stage0 K-group0
```

执行后：

```text
iterator -> stage0 K-group1
```

已经加载到：

```text
warp_loaded_frag_[0]
```

中的 K-group0 数据不会受影响。

### 13.1 operator++() 的底层行为

底层代码会根据 K-group 数量修改：

```text
byte_offset_
k_group_idx_
```

关键逻辑：

```cpp
byte_offset_ ^= ...;
++k_group_idx_;
```

当一个 stage 的所有 K-group 走完时，再移动到下一个 stage。

### 13.2 为什么使用 XOR

shared memory 采用：

```text
TensorOpMultiplicandCrosswise
```

布局。

逻辑 K-group：

```text
0 -> 1 -> 2 -> 3
```

在物理 shared-memory 地址中不一定是简单的固定字节递增。

iterator 通过 XOR 更新地址，以匹配 shared-memory swizzle 并降低
`ldmatrix` bank conflict。

---

## 14. 第 380-381 行：预发下一个 stage 的第一组 copy

源码：

```cpp
// Start issuing the first group of the next stage outside of the mainloop
copy_tiles_and_advance(iterator_A, iterator_B);
```

默认参数：

```cpp
group_start_A = 0;
group_start_B = 0;
```

所以等价于：

```cpp
copy_tiles_and_advance(
    iterator_A,
    iterator_B,
    0,
    0);
```

---

## 15. 第 381 行中的 next stage 是哪个 stage

prologue 已完成：

```text
global tile0 -> shared stage0
global tile1 -> shared stage1
```

并已推进：

```text
global iterator -> tile2
shared write iterator -> stage2
```

因此第 381 行实际发射：

```text
global tile2 的 copy group0
        ↓ cp.async
shared stage2
```

它不是重新复制 stage0 或 stage1。

---

## 16. 第 381 行为什么只复制第一组

`copy_tiles_and_advance()` 内部只循环：

```cpp
for (int j = 0;
     j < Detail::kAccessesPerGroupA;
     ++j) {
  ...
}
```

当前：

```text
kAccessesPerGroupA = 2
```

所以本次只处理：

```text
ThreadMap logical access 0、1
```

align4 下，每个逻辑 access 又拆成两个实际访问：

```text
logical access 0:
  v=0
  v=1

logical access 1:
  v=0
  v=1
```

因此本次每线程发射：

```text
4 条 cp.async
```

一个完整 stage 每线程需要：

```text
16 条 cp.async
```

所以第 381 行只完成 stage2 的四分之一。

---

## 17. 为什么不在第 381 行发射完整 stage2

如果在 mainloop 前一次性发射完整 stage2：

```text
发射 stage2 全部 cp.async
        ↓
开始 stage0 的全部 MMA
```

访存与计算的交错粒度较粗。

CUTLASS 将 stage2 的 copy 分散到 stage0 的四次 warp MMA 附近：

```text
mainloop 前:
  发射 tile2 copy group0

stage0 warp_mma_k=0:
  发射 tile2 copy group1
  计算 stage0 K-group0

stage0 warp_mma_k=1:
  发射 tile2 copy group2
  计算 stage0 K-group1

stage0 warp_mma_k=2:
  发射 tile2 copy group3
  计算 stage0 K-group2

stage0 warp_mma_k=3:
  发射 tile3 copy group0
  计算 stage0 K-group3
```

这样：

```text
当前 stage 的 shared load 和 Tensor Core 计算
```

可以隐藏：

```text
未来 stage 的 global-memory latency
```

---

## 18. 为什么 group0 必须在 mainloop 外发射

mainloop 中 copy group 的起点计算为：

```cpp
if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
  group_start_iteration_A = 0;
}
else {
  group_start_iteration_A =
      (warp_mma_k + 1) *
      Detail::kAccessesPerGroupA;
}
```

当前：

```text
kAccessesPerGroupA = 2
kWarpGemmIterations = 4
```

所以：

| `warp_mma_k` | `group_start_A` | 发射内容 |
|---:|---:|---|
| 0 | 2 | 当前 future tile group1 |
| 1 | 4 | 当前 future tile group2 |
| 2 | 6 | 当前 future tile group3 |
| 3 | 0 | 下一个 future tile group0 |

稳定运行时，一个 tile 的完整 copy 分布为：

```text
前一轮 warp_mma_k=3:
  group0

当前轮 warp_mma_k=0:
  group1

当前轮 warp_mma_k=1:
  group2

当前轮 warp_mma_k=2:
  group3
```

第一次进入 mainloop 时不存在“前一轮 `warp_mma_k=3`”。

如果没有第 381 行：

```text
第一次 mainloop 只会发射 tile2 group1、group2、group3
tile2 group0 将缺失
```

所以第 381 行用于人工补上第一次流水线所缺少的 group0。

这属于：

```text
pipeline bootstrap
```

或：

```text
pipeline priming
```

---

## 19. `copy_tiles_and_advance()` 的名字容易误解

第 381 行调用：

```cpp
copy_tiles_and_advance(iterator_A, iterator_B);
```

但该函数内部不会调用：

```cpp
iterator_A.advance();
iterator_B.advance();
```

它内部主要通过：

```cpp
++iterator_A;
++iterator_B;
```

推进当前 tile 内的子访问。

这里有两种 advance 层级。

### 19.1 `operator++()`

表示：

```text
当前 global tile 内
移动到下一个 AccessType 子访问
```

### 19.2 `advance()`

表示：

```text
从当前完整 GEMM K-tile
移动到下一个 GEMM K-tile
```

真正的：

```cpp
iterator_A.advance();
iterator_B.advance();
```

要等 tile2 的 `group3` 发射完成后执行。

因此函数名中的 `advance` 更准确地理解为：

```text
发射当前 copy group，并推进组内访问 iterator
```

而不是：

```text
发射后立即切换到下一个完整 stage
```

---

## 20. 第 383-384 行：初始化 shared 环形缓冲索引

源码：

```cpp
int smem_write_stage_idx =
    Base::kStages - 1;

int smem_read_stage_idx = 0;
```

当前 `Stages=3`：

```text
smem_write_stage_idx = 2
smem_read_stage_idx  = 0
```

含义：

```text
shared stage0:
  warp 当前读取和计算

shared stage1:
  prologue 已经预取

shared stage2:
  global iterator 正在写入 tile2
```

这两个整数用于后续环形回绕判断。

它们本身：

```text
不保存 A/B 数据
不执行 shared-memory load
不直接改变 iterator 地址
```

真正的 iterator 地址移动仍由：

```cpp
smem_iterator_A_.add_tile_offset(...)
warp_tile_iterator_A_.add_tile_offset(...)
```

等操作完成。

---

## 21. 第 386 行：转换第一个 loaded fragment

源码：

```cpp
warp_mma.transform(
    warp_transformed_frag_A[0],
    warp_transformed_frag_B[0],
    warp_loaded_frag_A[0],
    warp_loaded_frag_B[0]);
```

通用语义：

```text
loaded source fragment
        ↓ 类型转换、打包、必要的重排
MMA-ready transformed fragment
```

即：

```text
warp_loaded_frag_A[0]
    -> warp_transformed_frag_A[0]

warp_loaded_frag_B[0]
    -> warp_transformed_frag_B[0]
```

---

## 22. transform() 跳转到哪里

实现位于：

```text
include/cutlass/gemm/warp/mma_tensor_op.h
```

函数：

```cpp
void transform(
    TransformedFragmentA &dst_A,
    TransformedFragmentB &dst_B,
    FragmentA const &A,
    FragmentB const &B) const;
```

内部使用：

```cpp
detail::ConvertAndPack<
    ArchMmaElement,
    SourceElement,
    ElementCount,
    RoundStyle
>
```

进行转换。

---

## 23. 当前 FP16 路径中 transform() 做了什么

当前：

```text
Source ElementA/B      = half_t
ArchMmaOperator ElementA/B = half_t
```

因此匹配同类型特化：

```cpp
template <
    typename T,
    int N,
    FloatRoundStyle Round
>
struct ConvertAndPack<T, T, N, Round> {

  Array<T, N> operator()(
      Array<T, N> const &source) {
    return source;
  }
};
```

所以本实例中：

```text
FP16 -> FP16
```

不发生数值类型转换。

逻辑上主要是：

```text
loaded fragment
    ↓
复制/寄存器重命名
    ↓
transformed fragment
```

编译器可能把它优化为寄存器重命名或很少量的寄存器移动。

保留 `transform()` 是为了统一其他路径，例如：

```text
float -> TF32
float -> half
float -> bfloat16
mixed input
complex transform
```

---

## 24. 为什么第一个 transform 放在 mainloop 外

mainloop 内有：

```cpp
if (warp_mma_k > 0) {
  warp_mma.transform(...);
}
```

当第一次进入 mainloop：

```text
warp_mma_k = 0
```

这个条件为 false。

原因是：

```text
K-group0 已经在 mainloop 外完成 load 和 transform
```

如果第 386 行不提前 transform：

```text
第一次 warp_mma_k=0
没有 MMA-ready fragment 可以直接计算
```

所以第 386 行和第 381 行具有相似的流水线启动作用：

```text
第 381 行:
  为 future tile 预发 copy group0

第 386 行:
  为 current tile 预转换 K-group0
```

它们都在补齐稳定 mainloop 的“前一轮状态”。

---

## 25. 第 364-386 行的两条并行流水线

这段代码同时启动两条流水线。

### 25.1 Global-to-shared 流水线

```text
global tile2 group0
    ↓ cp.async
shared stage2
```

对应：

```cpp
copy_tiles_and_advance(iterator_A, iterator_B);
```

### 25.2 Shared-to-register-to-MMA 流水线

```text
shared stage0 K-group0
    ↓ ldmatrix
loaded fragment[0]
    ↓ transform
transformed fragment[0]
    ↓ mainloop warp_mma
accumulator
```

对应：

```cpp
warp_tile_iterator.load(...)
warp_mma.transform(...)
```

这两条流水线作用于不同时间层级的数据：

```text
当前计算:
  stage0 / tile0

未来预取:
  stage2 / tile2
```

---

## 26. 第 386 行执行后的完整状态

第 386 行执行后：

```text
Global iterator:
  仍然对应 tile2
  组内 iterator 已经过 group0 的访问

Shared memory:
  stage0 = tile0，当前被读取
  stage1 = tile1，已预取
  stage2 = 正在接收 tile2 group0

Warp shared iterator:
  已从 stage0 K-group0 前进到 K-group1

Register loaded buffer:
  loaded[0] = stage0 K-group0
  loaded[1] = 尚未加载

Register transformed buffer:
  transformed[0] = stage0 K-group0，MMA-ready
  transformed[1] = 尚未转换

Accumulator:
  尚未执行本轮 K-group0 的 warp_mma
```

图示：

```text
Global tile2 group0
        │
        │ cp.async 已发射
        ▼
Shared stage2


Shared stage0 K-group0
        │
        │ ldmatrix 已完成
        ▼
warp_loaded_frag_[0]
        │
        │ transform 已完成
        ▼
warp_transformed_frag_[0]
        │
        │ 等待 mainloop warp_mma_k=0
        ▼
mma.sync
```

---

## 27. 进入 mainloop 后第一次迭代

第一次：

```text
warp_mma_k = 0
```

主要发生：

```text
1. 从 shared stage0 加载 K-group1 到 loaded[1]
2. warp shared iterator 前进到 K-group2
3. 不转换 slot0，因为 slot0 已在 mainloop 外转换
4. 发射 global tile2 copy group1
5. 使用 transformed[0] 计算 stage0 K-group0
```

状态交错：

```text
计算:
  stage0 K-group0

shared -> register:
  stage0 K-group1

global -> shared:
  tile2 copy group1
```

---

## 28. 第二次 warp MMA iteration

当：

```text
warp_mma_k = 1
```

主要发生：

```text
1. 从 shared stage0 加载 K-group2 到 loaded[0]
2. 将 loaded[1] 的 K-group1 转换到 transformed[1]
3. 发射 global tile2 copy group2
4. 使用 transformed[1] 计算 stage0 K-group1
```

slot0 和 slot1 开始交替使用。

---

## 29. 当前 stage 的完整时间线

```text
mainloop 外:
  load stage0 K-group0 -> loaded[0]
  transform loaded[0]  -> transformed[0]
  copy tile2 group0    -> shared stage2

warp_mma_k=0:
  load stage0 K-group1 -> loaded[1]
  copy tile2 group1
  MMA stage0 K-group0 using transformed[0]

warp_mma_k=1:
  load stage0 K-group2 -> loaded[0]
  transform loaded[1]  -> transformed[1]
  copy tile2 group2
  MMA stage0 K-group1 using transformed[1]

warp_mma_k=2:
  load stage0 K-group3 -> loaded[1]
  transform loaded[0]  -> transformed[0]
  copy tile2 group3
  cp_async_fence for complete tile2
  wait for next readable stage
  advance global iterator to tile3
  MMA stage0 K-group2 using transformed[0]

warp_mma_k=3:
  load stage1 K-group0 -> loaded[0]
  transform loaded[1]  -> transformed[1]
  copy tile3 group0
  transform next stage K-group0
  MMA stage0 K-group3 using transformed[1]
```

下一轮开始时：

```text
current compute stage = stage1
future copy tile       = tile3
future copy group0     = 已发射
next K-group0 fragment = 已加载并转换
```

流水线进入稳定状态。

---

## 30. 为什么要提前加载下一 K-group

如果采用顺序执行：

```text
load K-group0
MMA K-group0
load K-group1
MMA K-group1
```

每次 MMA 前都要等待 shared-memory load。

双缓冲后：

```text
MMA K-group0
同时准备 K-group1

MMA K-group1
同时准备 K-group2
```

虽然单个 warp 的指令仍按程序顺序发射，但编译器和 GPU pipeline 可以让：

```text
ldmatrix
fragment transform
mma.sync
```

在不同依赖链上交错，从而隐藏 shared-memory load latency。

---

## 31. 为什么要提前发射未来 stage 的 copy group

同理，如果采用：

```text
完整计算 tile0
然后开始加载 tile1
```

计算结束后会等待 global memory。

multistage pipeline 改为：

```text
计算 tile0 时加载 tile2
计算 tile1 时加载 tile3
计算 tile2 时加载 tile4
```

当前：

```text
stage0 正在计算
stage1 已经预取
stage2 正在填充
```

这样可以用 Tensor Core 计算时间隐藏 global-memory latency。

---

## 32. 两级双缓冲

当前代码实际上包含两级 buffering。

### 32.1 Shared-memory 多 stage 缓冲

```text
stage0
stage1
stage2
```

用于重叠：

```text
global memory -> shared memory
```

与：

```text
shared memory -> Tensor Core computation
```

### 32.2 Warp register 双缓冲

```text
fragment slot0
fragment slot1
```

用于重叠：

```text
shared memory -> register
```

与：

```text
register -> mma.sync
```

整体结构：

```text
Global memory
      │
      │ cp.async
      ▼
Shared stage0 / stage1 / stage2
      │
      │ ldmatrix
      ▼
Register slot0 / slot1
      │
      │ transform + mma.sync
      ▼
Accumulator registers
```

---

## 33. 三种 group 再总结

### Software copy group

```text
global -> shared
```

当前一个 stage 拆成四个 copy group。

第 381 行说的 `first group` 就是这个 group0。

### Warp MMA K-group

```text
shared -> register -> MMA
```

当前 `K=64` 拆成四个 `K=16` group。

### cp.async commit group

```cpp
cp_async_fence();
```

把此前发出的 `cp.async` 提交为硬件跟踪的异步 copy group。

当前通常在完整 future tile 的四个 software copy group 都发出后提交。

---

## 34. 源码调用链

### Fragment 类型

```text
implicit_gemm_multistage.h
  -> Policy::Operator
  -> warp::MmaTensorOp
  -> FragmentA/B
  -> TransformedFragmentA/B
```

主要文件：

```text
include/cutlass/conv/threadblock/implicit_gemm_multistage.h
include/cutlass/gemm/warp/mma_tensor_op.h
```

### warp iterator

```text
MmaBase::warp_tile_iterator_A/B_
  -> Operator::IteratorA/B
  -> MmaTensorOpMultiplicandTileIterator
  -> TensorOpMultiplicandCrosswise iterator
```

主要文件：

```text
include/cutlass/gemm/threadblock/mma_base.h
include/cutlass/gemm/warp/mma_tensor_op_tile_iterator.h
```

### Tensor Core 指令

```text
warp::MmaTensorOp::operator()
  -> ArchMmaOperator
  -> arch::Mma<GemmShape<16,8,16>, ...>
  -> mma.sync.aligned.m16n8k16
```

主要文件：

```text
include/cutlass/gemm/warp/default_mma_tensor_op.h
include/cutlass/gemm/warp/mma_tensor_op.h
include/cutlass/arch/mma_sm80.h
```

---

## 35. 最终总结

第 364-386 行不是正式 mainloop 计算，而是在建立稳定流水线所需的初始状态。

它完成：

```text
1. 创建 loaded/transformed 两套寄存器双缓冲
2. 构造 warp-level Tensor Core MMA 操作器
3. 将 shared stage0 iterator 定位到 K-group0
4. 通过 ldmatrix 将 stage0 K-group0 加载到 loaded[0]
5. 将 shared iterator 提前推进到 K-group1
6. 发射 global tile2 到 shared stage2 的 copy group0
7. 初始化 shared-memory 环形读写索引
8. 将 loaded[0] 转换成 MMA-ready transformed[0]
```

第 381 行的核心意义：

```text
为第一次 mainloop 人工预发 future tile 的 group0。
稳定流水线中，这个 group0 本应由前一轮 warp_mma_k=3 发射。
```

第 386 行的核心意义：

```text
为第一次 mainloop 人工预转换 current tile 的 K-group0。
稳定流水线中，下一轮 K-group0 会由前一轮末尾提前准备。
```

因此二者都是 pipeline bootstrap：

```text
第 381 行启动 global-to-shared 流水线
第 386 行启动 shared-to-register-to-MMA 流水线
```

执行完后：

```text
stage0 K-group0 已经在寄存器中并可直接执行 MMA
stage0 K-group1 已成为下一次 shared load 的目标
stage1 已经预取
stage2 正在接收 tile2 copy group0
```

随后 mainloop 可以稳定地执行：

```text
加载下一 K-group
转换当前 fragment
计算当前 K-group
发射未来 stage 的下一 copy group
```

从而同时隐藏 shared-memory latency 和 global-memory latency。
