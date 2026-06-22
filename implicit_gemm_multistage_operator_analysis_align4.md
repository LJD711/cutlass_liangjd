# CUTLASS `ImplicitGemmMultistage::operator()` 分析（SM80 F16 NHWC align4）

本文从以下单元测试出发，分析 `implicit_gemm_multistage.h` 中的异步拷贝分组和 `operator()` 流水线：

```text
SM80_Device_Conv2d_Fprop_Optimized_ImplicitGemm_
f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_align4
```

对应文件：

```text
test/unit/conv/device/
conv2d_fprop_implicit_gemm_f16nhwc_f16nhwc_f16nhwc_tensor_op_f16_sm80.cu
```

重点解释：

- 为什么复制一个 A tile 时，每个线程会执行 16 个子访问；
- `Detail::kAccessesPerGroupA` 的含义；
- `copy_tiles_and_advance()` 如何复制一个 group；
- `ImplicitGemmMultistage::operator()` 如何实现三阶段流水线；
- `set_iteration_index()`、`operator++()` 和 `advance()` 分别负责哪一层移动。

---

## 1. 当前测试的关键模板参数

测试中的主要配置为：

```cpp
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 64>;
using WarpShape        = cutlass::gemm::GemmShape<64, 64, 64>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

static int const Stages     = 3;
static int const AlignmentA = 4;
static int const AlignmentB = 4;
```

因此：

```text
CTA GEMM tile      = 128 × 128 × 64
Warp GEMM tile     = 64 × 64 × 64
Tensor Core 指令 K = 16
ElementA            = half，16 bit
Global access       = 4 个 half，64 bit，8 bytes
```

---

## 2. 三个不同的访问计数层级

理解这段代码时，需要明确区分三种计数：

1. ThreadMap 逻辑 access；
2. `AccessType` 子访问；
3. warp MMA copy group。

它们不是同一个概念。

### 2.1 ThreadMap 逻辑 access

SM80 `DefaultMmaCore` 将线程访问粒度固定为：

```cpp
static int const kAccessSizeInBits = 128;
```

由于 `ElementA=half`：

```text
ThreadMap::kElementsPerAccess
    = 128 bit / 16 bit
    = 8 half
```

所以 ThreadMap 的一个逻辑 access 表示连续的 8 个 half，即 16 bytes。

对于当前 `128 × 64` 的 A tile，ThreadMap 得到：

```cpp
ThreadMap::Iterations::kContiguous = 1;
ThreadMap::Iterations::kStrided    = 8;
ThreadMap::Iterations::kCount      = 1 * 8 = 8;
```

也就是说，每个线程在一个 A tile 内负责 8 个逻辑位置。

### 2.2 `AccessType` 子访问

测试显式传入：

```cpp
AlignmentA = 4;
```

所以：

```cpp
using AccessTypeA = cutlass::AlignedArray<half_t, 4>;
```

因此：

```cpp
AccessType::kElements = 4;
```

一个实际 global-memory access 包含：

```text
4 half × 16 bit = 64 bit = 8 bytes
```

iterator 中定义：

```cpp
static int const kAccessesPerVector =
    ThreadMap::kElementsPerAccess / AccessType::kElements;
```

代入：

```text
kAccessesPerVector = 8 / 4 = 2
```

因此 ThreadMap 的一个 8-half 逻辑向量被拆成：

```text
v=0：前 4 个 half，8 bytes
v=1：后 4 个 half，8 bytes
```

### 2.3 Warp MMA iteration

`MmaBase` 中定义：

```cpp
static int const kWarpGemmIterations =
    WarpGemm::kK / Operator::Policy::MmaShape::kK;
```

当前：

```text
WarpGemm::kK = 64
MmaShape::kK = 16
```

所以：

```text
kWarpGemmIterations = 64 / 16 = 4
```

当前 shared-memory stage 中的 K=64 被拆成四次 warp MMA：

```text
warp_mma_k=0：K[0..15]
warp_mma_k=1：K[16..31]
warp_mma_k=2：K[32..47]
warp_mma_k=3：K[48..63]
```

---

## 3. 为什么每个线程复制一个 A tile 时有 16 个子访问

Prologue 的 A copy 结构为：

```cpp
for (int j = 0;
     j < Detail::AsyncCopyIterationsPerStageA;
     ++j) {

  for (int v = 0;
       v < IteratorA::kAccessesPerVector;
       ++v) {

    cp_async(...);
  }
}
```

其中：

```text
AsyncCopyIterationsPerStageA
    = ThreadMap::Iterations::kCount
    = 8

kAccessesPerVector
    = 2
```

因此：

```text
每线程每个 A stage 的 cp.async 数
    = 8 × 2
    = 16
```

访问顺序如下：

| `j` | `v` | iterator index | 含义 |
|---:|---:|---:|---|
| 0 | 0 | 0 | strided 0，前 4 half |
| 0 | 1 | 1 | strided 0，后 4 half |
| 1 | 0 | 2 | strided 1，前 4 half |
| 1 | 1 | 3 | strided 1，后 4 half |
| ... | ... | ... | ... |
| 7 | 0 | 14 | strided 7，前 4 half |
| 7 | 1 | 15 | strided 7，后 4 half |

每条 `cp.async` 的大小为：

```cpp
kSrcBytes =
    sizeof_bits<Element>::value *
    ThreadMap::kElementsPerAccess /
    kAccessesPerVector /
    8;
```

代入：

```text
kSrcBytes
    = 16 × 8 / 2 / 8
    = 8 bytes
```

所以每个线程搬运：

```text
16 × 8 bytes = 128 bytes
```

CTA 有 128 个线程，因此总搬运量为：

```text
128 threads × 128 bytes = 16384 bytes
```

完整 A tile 的大小也是：

```text
128 × 64 × sizeof(half)
    = 128 × 64 × 2
    = 16384 bytes
```

两者完全对应。

> 注意：`AsyncCopyIterationsPerStageA` 在 align4 情况下并不等于真实
> `cp.async` 指令数量。它表示 ThreadMap 的逻辑 access 数。真实指令数量还要乘
> `kAccessesPerVector`。

---

## 4. `Detail::kAccessesPerGroupA` 的含义

定义为：

```cpp
static int const kAccessesPerGroupA =
    (AsyncCopyIterationsPerStageA +
     Base::kWarpGemmIterations - 1) /
    Base::kWarpGemmIterations;
```

代入当前参数：

```text
kAccessesPerGroupA
    = ceil(8 / 4)
    = (8 + 4 - 1) / 4
    = 2
```

它表示：

> 将一个 stage 的 8 个 ThreadMap 逻辑 access 分散到 4 个 warp MMA
> iteration 中，每个 group 负责 2 个 ThreadMap 逻辑 access。

它不是每组真实的 `cp.async` 指令数量。

每个 group 的真实 `cp.async` 数量为：

```text
kAccessesPerGroupA × kAccessesPerVector
    = 2 × 2
    = 4 条 cp.async / 线程
```

整个 stage：

```text
4 groups × 4 cp.async
    = 16 cp.async / 线程
```

具体分组如下：

| group | `group_start_A` | ThreadMap 逻辑 access | iterator index | 每线程 `cp.async` 数 |
|---:|---:|---|---|---:|
| 0 | 0 | 0、1 | 0～3 | 4 |
| 1 | 2 | 2、3 | 4～7 | 4 |
| 2 | 4 | 4、5 | 8～11 | 4 |
| 3 | 6 | 6、7 | 12～15 | 4 |

因此源码中将其描述为“每组 cp.async 指令数量”并不十分准确。更准确的描述是：

```text
每个 copy group 包含的 ThreadMap 逻辑 access 数量
```

### 4.1 这里的 group 不是 convolution group

`ImplicitGemmMultistage` 附近同时出现了多种名为 `group` 的概念，阅读时必须
区分。第 381 行注释：

```cpp
// Start issuing the first group of the next stage outside of the mainloop
copy_tiles_and_advance(iterator_A, iterator_B);
```

这里的 `first group` 指的是：

```text
一个完整 global-to-shared stage 中的第一个软件 copy group
```

它不是 group convolution 中的 `problem_size.groups`，也不是一个完整的
shared-memory stage。

#### 4.1.1 Software copy group

本文主要讨论的 group 是软件层面的 copy group：

```text
一个完整 stage 的一部分 global-to-shared copy iteration
```

当前 A operand：

```text
一个完整 stage
    = 8 个 ThreadMap 逻辑 access
    = 16 条 cp.async / 线程

一个 software copy group
    = 2 个 ThreadMap 逻辑 access
    = 4 条 cp.async / 线程

一个完整 stage
    = group0 + group1 + group2 + group3
```

把 stage 拆成四个 copy group，是为了把下一个 stage 的 global-memory load
均匀穿插在当前 stage 的四次 warp MMA 计算之间。

#### 4.1.2 Warp MMA K-group

下面的调用操作的是另一种 group：

```cpp
warp_tile_iterator_A_.set_kgroup_index(...);
warp_tile_iterator_B_.set_kgroup_index(...);
```

它表示当前 shared-memory stage 内，warp 正在读取哪个 K 子区间。

当前：

```text
WarpShape::kK       = 64
MMA instruction kK = 16
```

所以一个 shared-memory stage 有四个 warp MMA K-group：

```text
K-group0 = K[0..15]
K-group1 = K[16..31]
K-group2 = K[32..47]
K-group3 = K[48..63]
```

它描述的是：

```text
shared memory -> warp register -> Tensor Core MMA
```

而 software copy group 描述的是：

```text
global memory -> shared memory
```

二者都被划分成四组，是为了让 copy 和 compute 一一交错，但它们不是同一种
数据对象。

#### 4.1.3 cp.async commit group

还有第三种 group，由：

```cpp
cutlass::arch::cp_async_fence();
```

提交。

在当前实现中，当某个 future tile 的 `group0..group3` 全部发射后，
`cp_async_fence()` 将此前尚未提交的 `cp.async` 指令组成一个硬件可跟踪的
异步 copy group。后面的：

```cpp
cp_async_wait<N>();
```

等待的是这种已提交的 cp.async group。

因此三种 group 可以总结为：

| 名称 | 数据路径 | 当前粒度 |
|---|---|---|
| software copy group | global memory -> shared memory | 完整 stage 的四分之一 |
| warp MMA K-group | shared memory -> register/MMA | K=64 中的一个 K=16 子区间 |
| cp.async commit group | cp.async 硬件队列 | 通常提交一个完整 future stage 的 copy |

第 381 行的 `first group` 明确指第一种：

```text
software copy group0
```

---

## 5. `set_iteration_index()` 的作用

activation iterator 中：

```cpp
void set_iteration_index(Index index) {
  iteration_vector_ = index % kAccessesPerVector;
  int residual_access = index / kAccessesPerVector;

  iteration_contiguous_ =
      residual_access % ThreadMap::Iterations::kContiguous;

  iteration_strided_ =
      residual_access / ThreadMap::Iterations::kContiguous;
}
```

当前：

```text
kAccessesPerVector                 = 2
ThreadMap::Iterations::kContiguous = 1
```

所以可以简化为：

```cpp
iteration_vector_  = index % 2;
iteration_strided_ = index / 2;
iteration_contiguous_ = 0;
```

例如：

```cpp
set_iteration_index(8);
```

得到：

```text
iteration_vector_  = 0
iteration_strided_ = 4
```

表示从第 4 个 ThreadMap 逻辑 access 的第 0 个 align4 子访问开始。

`get()` 使用这些状态计算地址：

```cpp
AccessType const *get() const {
  return reinterpret_cast<AccessType const *>(
      pointer_[iteration_strided_]
  ) + iteration_vector_;
}
```

地址关系为：

```text
pointer_[s] + 0：该行前 4 个 half
pointer_[s] + 1：该行后 4 个 half
```

---

## 6. Global iterator 的 `operator++()`

`operator++()` 按以下顺序移动：

```text
iteration_vector_
    ↓ 回绕
iteration_contiguous_
    ↓ 回绕
iteration_strided_
```

当前配置下，访问顺序为：

```text
(strided=0, vector=0)
(strided=0, vector=1)
(strided=1, vector=0)
(strided=1, vector=1)
...
(strided=7, vector=1)
```

因此：

```text
set_iteration_index()
```

负责随机定位当前 tile 内某个 group 的起点，而：

```text
++iterator_A
```

负责在当前 tile 内顺序遍历子访问。

---

## 7. `advance()` 的作用

`advance()` 不负责 tile 内的子访问，而负责移动到下一个 implicit GEMM K tile。

对于 fprop activation，遍历顺序为：

```text
filter_s_ 最快
filter_r_ 次之
channel tile 最慢
```

逻辑过程为：

```cpp
++filter_s_;

if (filter_s_ == S) {
  filter_s_ = 0;
  ++filter_r_;

  if (filter_r_ == R) {
    filter_r_ = 0;
    filter_c_ += filter_c_delta;
  }
}
```

对于 `R=S=3`、`ThreadblockShape::kK=64`：

```text
(c=0,  r=0,s=0)
(c=0,  r=0,s=1)
(c=0,  r=0,s=2)
(c=0,  r=1,s=0)
...
(c=0,  r=2,s=2)
(c=64, r=0,s=0)
...
```

所以这里所说的一个“global tile”或“CTA K tile”对应一个：

```text
(filter_r, filter_s, channel tile)
```

---

## 8. `copy_tiles_and_advance()` 逐段分析

函数签名：

```cpp
void copy_tiles_and_advance(
    IteratorA &iterator_A,
    IteratorB &iterator_B,
    int group_start_A = 0,
    int group_start_B = 0) {
```

这个函数复制当前 global A/B tile 的一个 group。

它不会调用 global iterator 的 `advance()`。函数名中的 “advance” 主要指：

- global access iterator 通过 `operator++()` 在 tile 内前进；
- shared-memory iterator 通过 `operator++()` 移到下一个逻辑 access。

### 8.1 定位 global A iterator

```cpp
iterator_A.set_iteration_index(
    group_start_A * IteratorA::kAccessesPerVector);
```

`group_start_A` 的单位是 ThreadMap 逻辑 access，而 global iterator index
的单位是 `AccessType` 子访问，所以必须乘：

```text
kAccessesPerVector = 2
```

例如：

```text
group_start_A = 4
index         = 4 × 2 = 8
```

最终定位到：

```text
iteration_strided_ = 4
iteration_vector_  = 0
```

### 8.2 定位 shared-memory iterator

```cpp
smem_iterator_A_.set_iteration_index(group_start_A);
```

shared-memory iterator 以完整 ThreadMap 逻辑 access 为单位，因此不乘
`kAccessesPerVector`。

### 8.3 遍历当前 group

```cpp
for (int j = 0;
     j < Detail::kAccessesPerGroupA;
     ++j) {
```

当前：

```text
kAccessesPerGroupA = 2
```

所以每组遍历两个 ThreadMap 逻辑 access。

```cpp
if (group_start_A + j <
    Detail::AsyncCopyIterationsPerStageA) {
```

该判断用于处理不能整除时的最后一个不完整 group。

例如总逻辑 access 数是 7，每组大小为 2，最后一组中只有一个 access 有效。

当前总数为 8，可以被 4 组整除，所以每组的两个 access 都有效。

### 8.4 获取 shared-memory 目标地址

```cpp
typename IteratorA::AccessType *dst_ptr =
    reinterpret_cast<typename IteratorA::AccessType *>(
        smem_iterator_A_.get());
```

当前：

```cpp
AccessType = AlignedArray<half, 4>
```

shared iterator 指向一个 8-half 逻辑向量的起点。转换为 `AccessType*` 后：

```text
dst_ptr + 0：前 4 half
dst_ptr + 1：后 4 half
```

### 8.5 计算每条 `cp.async` 大小

```cpp
int const kSrcBytes =
    sizeof_bits<typename IteratorA::Element>::value *
    IteratorA::ThreadMap::kElementsPerAccess /
    IteratorA::kAccessesPerVector /
    8;
```

当前：

```text
kSrcBytes = 16 × 8 / 2 / 8 = 8 bytes
```

### 8.6 拆分一个逻辑向量

```cpp
for (int v = 0;
     v < IteratorA::kAccessesPerVector;
     ++v) {
```

当前执行：

```text
v=0、v=1
```

即把一个 8-half 逻辑向量拆成两个 4-half 子访问。

### 8.7 发出异步拷贝

```cpp
cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
    dst_ptr + v,
    iterator_A.get(),
    iterator_A.valid());
```

参数含义：

- `dst_ptr + v`：shared-memory 目标地址；
- `iterator_A.get()`：global-memory activation 地址；
- `iterator_A.valid()`：当前 `(n,h,w,c,r,s)` 是否有效；
- 无效时不访问 global memory，并向 shared memory 写零。

### 8.8 Global iterator 在 tile 内前进

```cpp
++iterator_A;
```

例如：

```text
(strided=0, vector=0)
→ (strided=0, vector=1)
→ (strided=1, vector=0)
```

### 8.9 Shared iterator 前进

```cpp
++smem_iterator_A_;
```

两个 `v` 都复制完成后，shared iterator 才移动到下一个完整的 8-half
逻辑向量。

B operand 的复制过程与 A 相同。

---

## 9. `operator()` 的参数

```cpp
void operator()(
    int gemm_k_iterations,
    FragmentC &accum,
    IteratorA iterator_A,
    IteratorB iterator_B,
    FragmentC const &src_accum,
    int gemm_k_iterations_per_channel = 0,
    int64_t imag_stride_A = 0,
    int64_t imag_stride_B = 0)
```

各参数含义：

- `gemm_k_iterations`：需要计算多少个 CTA K tile；
- `accum`：输出累加器 fragment；
- `iterator_A/B`：每线程持有的 global-memory iterator；
- `src_accum`：累加器初始值；
- 后三个参数在当前实现中未直接使用。

对于 optimized fprop：

```cpp
gemm_k_iterations =
    R * S * ceil(
        ceil(C / split_k_slices) /
        ThreadblockShape::kK
    );
```

当前 `R=S=3`、`Ktile=64`、`split_k=1`：

```text
C=12  -> 3 × 3 × 1 = 9
C=28  -> 3 × 3 × 1 = 9
C=100 -> 3 × 3 × 2 = 18
```

---

## 10. Prologue：预加载 `Stages-1` 个完整 stage

```cpp
for (int stage = 0;
     stage < Base::kStages - 1;
     ++stage, --gemm_k_iterations) {
```

当前：

```text
Stages = 3
```

所以 prologue 预加载两个 stage：

```text
stage0
stage1
```

每预加载一个 stage，剩余 `gemm_k_iterations` 减一。

例如最初：

```text
gemm_k_iterations = 9
```

prologue 后：

```text
gemm_k_iterations = 7
```

### 10.1 重置当前 A tile 内部访问坐标

```cpp
iterator_A.set_iteration_index(0);
smem_iterator_A_.set_iteration_index(0);
```

这里仅将 global/shared iterator 重置到当前 tile 的第 0 个 access。

它不会把 global iterator 回退到前一个 `(r,s,c_tile)`。

### 10.2 完整复制 A tile

```cpp
for (int j = 0;
     j < AsyncCopyIterationsPerStageA;
     ++j) {
```

遍历 8 个 ThreadMap 逻辑 access。

```cpp
for (int v = 0;
     v < kAccessesPerVector;
     ++v) {
```

每个逻辑 access 拆成两个 align4 子访问。

所以：

```text
8 × 2 = 16 条 cp.async / 线程
```

### 10.3 完整复制 B tile

B operand 也执行相同结构：

```text
8 个逻辑 access × 2 个子访问
    = 16 条 cp.async / 线程
```

### 10.4 推进 global iterator

```cpp
iterator_A.advance();
iterator_B.advance();
```

当前 `(r,s,c_tile)` 的完整 A/B tile 已经发出复制，global iterator 移到下一个
implicit GEMM K tile。

例如：

```text
(r=0,s=0,c=0)
    ↓ advance()
(r=0,s=1,c=0)
```

### 10.5 推进 shared-memory 写 stage

```cpp
smem_iterator_A_.add_tile_offset({0, 1});
smem_iterator_B_.add_tile_offset({1, 0});
```

shared-memory 写位置从当前 stage 移动到下一个 stage。

A/B 坐标方向不同，是因为它们的 shared-memory 矩阵布局方向不同。

### 10.6 提交 cp.async group

```cpp
cutlass::arch::cp_async_fence();
```

将此前为当前 A/B tile 发出的 `cp.async` 指令提交为一个异步 copy group。

因此每次 prologue 循环完成：

```text
复制完整 A tile
复制完整 B tile
global iterator advance
shared write iterator 切换 stage
提交 cp.async group
```

Stages=3 时：

```text
stage0：加载 global K-tile0
stage1：加载 global K-tile1
```

---

## 11. 等待第一个 shared stage 就绪

```cpp
accum = src_accum;
```

初始化寄存器累加器。

```cpp
cp_async_wait<Base::kStages - 2>();
```

当前：

```text
cp_async_wait<1>()
```

prologue 已经提交两个 group：

```text
group0：stage0
group1：stage1
```

`wait<1>` 保证最老的 stage0 已经完成；较新的 stage1 可以仍在传输。

```cpp
__syncthreads();
```

保证 CTA 所有线程都完成 shared-memory 可见性同步，随后 warp 才能安全读取 stage0。

---

## 12. Shared memory 到 register 的双缓冲

```cpp
WarpLoadedFragmentA warp_loaded_frag_A[2];
WarpLoadedFragmentB warp_loaded_frag_B[2];
```

保存从 shared memory 读出的原始 fragment。

```cpp
WarpTransformedFragmentA warp_transformed_frag_A[2];
WarpTransformedFragmentB warp_transformed_frag_B[2];
```

保存经过 `warp_mma.transform()` 转换、可以送入 MMA 的 fragment。

数组长度为 2，用奇偶索引实现寄存器双缓冲：

```text
当前 fragment 用于计算
另一个 fragment 同时加载下一组数据
```

```cpp
Operator warp_mma;
```

构造 warp-level Tensor Core MMA 操作。

```cpp
warp_tile_iterator_A_.set_kgroup_index(0);
warp_tile_iterator_B_.set_kgroup_index(0);
```

定位到当前 shared stage 的第 0 个 K=16 group。

```cpp
warp_tile_iterator_A_.load(warp_loaded_frag_A[0]);
warp_tile_iterator_B_.load(warp_loaded_frag_B[0]);
```

从 shared memory 读取 K-group0 到寄存器。

```cpp
++warp_tile_iterator_A_;
++warp_tile_iterator_B_;
```

shared read iterator 指向下一 K group。

---

## 13. Mainloop 前预发未来 tile 的 group0

```cpp
copy_tiles_and_advance(iterator_A, iterator_B);
```

默认：

```text
group_start_A = 0
group_start_B = 0
```

prologue 已经加载：

```text
tile0 -> shared stage0
tile1 -> shared stage1
```

并将 global iterator 推进到 tile2。

因此这里开始复制：

```text
tile2 的 group0
```

即：

```text
逻辑 access 0、1
实际 4 条 cp.async / 线程
```

此时 tile2 的其余 group 尚未发出。

### 13.1 为什么只发 group0，而不是发完整 stage2

如果在这里一次性发射 stage2 的全部 16 条 `cp.async / 线程`，指令顺序会更接近：

```text
集中发射大量 global-to-shared copy
        ↓
集中执行当前 stage 的 warp MMA
```

这样访存和计算之间的交错较粗。

CUTLASS 将 stage2 拆成四个 software copy group，并把它们分散到当前 stage
的四次 `warp_mma_k` 附近：

```text
mainloop 前：      发射 stage2 group0
warp_mma_k = 0：   发射 stage2 group1，同时计算当前 stage
warp_mma_k = 1：   发射 stage2 group2，同时计算当前 stage
warp_mma_k = 2：   发射 stage2 group3，同时计算当前 stage
```

这样下一个 stage 的 global-memory latency 可以被当前 stage 的 shared-memory
load、fragment transform 和 Tensor Core MMA 覆盖。

### 13.2 为什么 group0 必须放在 mainloop 外

mainloop 内 `warp_mma_k=0` 对应的 copy 位置已经被安排给：

```text
当前 future tile 的 group1
```

具体公式为：

```cpp
group_start_iteration_A =
    (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
```

当：

```text
warp_mma_k = 0
kAccessesPerGroupA = 2
```

得到：

```text
group_start_iteration_A = 2
```

也就是直接从 group1 开始。

正常稳定运行时，一个 future tile 的 group0 是在“前一轮”的
`warp_mma_k=3` 发出的：

```text
前一轮 warp_mma_k=3：future tile group0
当前轮 warp_mma_k=0：future tile group1
当前轮 warp_mma_k=1：future tile group2
当前轮 warp_mma_k=2：future tile group3
```

但是第一次进入 mainloop 时不存在前一轮 `warp_mma_k=3`。如果不在第 381 行
预先发射 tile2 的 group0，那么第一次 mainloop 只会发射 group1、group2、
group3，tile2 的 copy 将不完整。

所以第 381 行属于软件流水线的启动步骤，也可以称为：

```text
pipeline bootstrap / pipeline priming
```

它为第一次 mainloop 人工补上本应由“前一轮末尾”发射的 group0。

### 13.3 第 381 行发生时三个 stage 的状态

当前 `Stages=3`，prologue 已完成：

```text
global tile0 -> shared stage0，已经提交
global tile1 -> shared stage1，已经提交
```

并且：

```text
global iterator -> tile2
shared write iterator -> stage2
shared read iterator -> stage0
```

因此第 381 行：

```cpp
copy_tiles_and_advance(iterator_A, iterator_B);
```

实际表示：

```text
从 global tile2 读取 group0
        ↓ cp.async
写入 shared stage2 对应位置
```

此时流水线状态可以画成：

```text
shared stage0：即将被 warp 计算
shared stage1：已经预取，等待后续计算
shared stage2：正在写入 tile2 的 group0
```

这里调用的函数名虽然是 `copy_tiles_and_advance()`，但它不会在函数内部调用
global iterator 的 tile 级 `advance()`。函数中的 “advance” 主要表现为
`operator++()` 推进当前 group 内的子访问。真正从 tile2 移到 tile3 的：

```cpp
iterator_A.advance();
iterator_B.advance();
```

要等到 tile2 的 group3 发射完成后，在 `warp_mma_k=2` 对应的代码中执行。

```cpp
int smem_write_stage_idx = Base::kStages - 1;
```

当前：

```text
smem_write_stage_idx = 2
```

因为 stage0、stage1 已被 prologue 使用，当前正在写 stage2。

```cpp
int smem_read_stage_idx = 0;
```

warp 当前从 stage0 读取。

```cpp
warp_mma.transform(...frag[0]...);
```

提前转换 stage0 的 K-group0。这样进入 mainloop 的 `warp_mma_k=0` 时可以直接计算。

---

## 14. Mainloop 外层循环

```cpp
for (; gemm_k_iterations > (-Base::kStages + 1);) {
```

当前：

```text
gemm_k_iterations > -2
```

循环允许计数变为负数，是为了排空流水线。

当不再有新的有效 global tile 时，shared memory 中仍可能存在已经预取、但尚未完成
MMA 的 stage。负数区间对应 pipeline drain。

---

## 15. 每个 CTA K tile 的四次 warp MMA

```cpp
for (int warp_mma_k = 0;
     warp_mma_k < Base::kWarpGemmIterations;
     ++warp_mma_k) {
```

当前：

```text
Base::kWarpGemmIterations = 4
```

分别处理：

```text
warp_mma_k=0：K[0..15]
warp_mma_k=1：K[16..31]
warp_mma_k=2：K[32..47]
warp_mma_k=3：K[48..63]
```

---

## 16. 预加载下一个 register fragment

```cpp
warp_tile_iterator_A_.set_kgroup_index(
    (warp_mma_k + 1) % Base::kWarpGemmIterations);
```

当前正在计算第 `warp_mma_k` 组，先定位下一 K group。

模 4 的原因是：

```text
当前 stage 的最后一个 K group之后
需要回到下一个 shared stage 的 K-group0
```

```cpp
warp_tile_iterator_A_.load(
    warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
```

把下一 K group 从 shared memory 加载到另一个双缓冲 fragment。

```cpp
++warp_tile_iterator_A_;
++warp_tile_iterator_B_;
```

shared iterator 移到下一个 K group。

```cpp
if (warp_mma_k > 0)
  warp_mma.transform(...);
```

转换当前即将参与计算的 fragment。

`warp_mma_k=0` 时跳过，是因为 K-group0 已在进入 mainloop 前转换完成。

---

## 17. 为当前 warp MMA iteration 计算 copy group

```cpp
if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
  group_start_iteration_A = 0;
  group_start_iteration_B = 0;
}
else {
  group_start_iteration_A =
      (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
}
```

当前：

```text
kAccessesPerGroupA = 2
```

所以：

| `warp_mma_k` | `group_start_A` | 复制内容 |
|---:|---:|---|
| 0 | 2 | 当前 global tile 的 group1 |
| 1 | 4 | 当前 global tile 的 group2 |
| 2 | 6 | 当前 global tile 的 group3 |
| 3 | 0 | 下一个 global tile 的 group0 |

group0 已经在前一轮末尾或者进入 mainloop 前预发。

因此一个 global tile 的 copy 分布为：

```text
前一轮 warp_mma_k=3：group0
当前轮 warp_mma_k=0：group1
当前轮 warp_mma_k=1：group2
当前轮 warp_mma_k=2：group3
```

```cpp
copy_tiles_and_advance(
    iterator_A,
    iterator_B,
    group_start_iteration_A,
    group_start_iteration_B);
```

发出当前 group 的 global-to-shared asynchronous copy。

---

## 18. Tensor Core MMA

当前 F16 test 中：

```text
Detail::kStagedAccumulation == false
```

因此执行：

```cpp
warp_mma(
    accum,
    warp_transformed_frag_A[warp_mma_k % 2],
    warp_transformed_frag_B[warp_mma_k % 2],
    accum);
```

语义是：

```text
accum = A_fragment × B_fragment + accum
```

每次处理 K=16，四次完成当前 CTA K=64 tile。

`kStagedAccumulation` 分支主要用于 tf32x3 等特殊数值模式，当前 test 不进入。

---

## 19. 最后一个 fragment 的特殊 transform

```cpp
if (warp_mma_k + 1 == Base::kWarpGemmIterations)
  warp_mma.transform(...);
```

只在：

```text
warp_mma_k = 3
```

执行。

此时刚加载的是下一个 shared stage 的 K-group0。

下一轮 `warp_mma_k=0` 会跳过普通 transform，因此必须在本轮末尾提前转换。

---

## 20. 为什么在 `warp_mma_k=2` 调用 `advance()`

条件：

```cpp
if (warp_mma_k + 2 == Base::kWarpGemmIterations)
```

当前等价于：

```text
warp_mma_k == 2
```

此时刚刚复制：

```text
group_start = 6
```

即当前 global tile 的最后一个 group3。

当前 tile 的全部 copy 已经发出：

```text
group0、group1、group2、group3
```

因此可以：

```cpp
cutlass::arch::cp_async_fence();
```

将当前完整 tile 的 copy 提交为一个异步 group。

```cpp
arch::cp_async_wait<Base::kStages - 2>();
__syncthreads();
```

等待即将被 warp 读取的 shared stage 就绪。

```cpp
iterator_A.advance();
iterator_B.advance();
```

global iterator 从当前 tile 移动到下一个 tile。

这样在下一次：

```text
warp_mma_k = 3
```

执行 `group_start=0` 时，复制的已经是新 tile 的 group0。

完整顺序为：

```text
warp_mma_k=2:
    复制旧 tile group3
    fence
    advance 到新 tile

warp_mma_k=3:
    复制新 tile group0
```

这就是 `advance()` 放在倒数第二个 warp MMA iteration 中的原因。

---

## 21. Shared-memory 写环形缓冲

```cpp
smem_iterator_A_.add_tile_offset({0, 1});
smem_iterator_B_.add_tile_offset({1, 0});
```

写指针移动到下一个 stage。

当前有三个 stage：

```text
stage0 → stage1 → stage2 → stage0
```

```cpp
if (smem_write_stage_idx == Base::kStages - 1) {
  smem_iterator_A_.add_tile_offset({0, -Base::kStages});
  smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
  smem_write_stage_idx = 0;
}
else {
  ++smem_write_stage_idx;
}
```

如果当前写 stage2，正常前进一步会超出 shared buffer，因此减 3 回绕到 stage0。

---

## 22. Shared-memory 读环形缓冲

```cpp
if (smem_read_stage_idx == Base::kStages - 1) {
  warp_tile_iterator_A_.add_tile_offset(
      {0,
       -Base::kStages *
        Policy::kPartitionsK *
        Base::kWarpGemmIterations});
}
```

warp iterator 在每轮中通过四次 `operator++()` 跨过当前 stage 的四个 K group。

读完 stage2 后，需要减去整个环形缓冲距离回到 stage0。

当前：

```text
Stages              = 3
kPartitionsK        = 1
kWarpGemmIterations = 4
```

所以回退：

```text
3 × 1 × 4 = 12 个 warp K group
```

---

## 23. 更新剩余 K tile 数

```cpp
--gemm_k_iterations;
```

一个外层 mainloop iteration 计算一个 CTA K=64 tile，因此剩余 tile 数减一。

虽然这行代码位于 `warp_mma_k=2` 分支中，但逻辑上代表当前整个 K=64 tile
即将完成。

---

## 24. 三阶段流水线时间线

Prologue：

```text
global tile0 -> shared stage0
global tile1 -> shared stage1
```

进入 mainloop 前：

```text
global tile2 group0 -> shared stage2
```

第一轮 mainloop：

```text
warp_mma_k=0:
    计算 shared stage0 K-group0
    复制 global tile2 group1

warp_mma_k=1:
    计算 shared stage0 K-group1
    复制 global tile2 group2

warp_mma_k=2:
    计算 shared stage0 K-group2
    复制 global tile2 group3
    fence：tile2 完整提交
    global iterator advance 到 tile3
    shared write iterator 回绕/前进

warp_mma_k=3:
    计算 shared stage0 K-group3
    复制 global tile3 group0
```

第二轮 mainloop：

```text
warp_mma_k=0:
    计算 shared stage1 K-group0
    复制 global tile3 group1

warp_mma_k=1:
    计算 shared stage1 K-group1
    复制 global tile3 group2

warp_mma_k=2:
    计算 shared stage1 K-group2
    复制 global tile3 group3
    fence
    global iterator advance 到 tile4

warp_mma_k=3:
    计算 shared stage1 K-group3
    复制 global tile4 group0
```

核心思想是：

```text
计算较老的 shared-memory stage
              +
同时预取未来的 global-memory tile
```

---

## 25. 三种 iterator 移动的层级

| 操作 | 作用范围 | 当前例子中的作用 |
|---|---|---|
| `set_iteration_index()` | 当前 global tile 内 | 定位当前 copy group 的子访问起点 |
| `++iterator_A` | 当前 global tile 内 | 从一个 4-half 子访问移到下一个 |
| `iterator_A.advance()` | global tile 之间 | 从当前 `(r,s,c_tile)` 移到下一个 |
| `add_pointer_offset()` | 所有底层基础指针 | 当前 `operator()` 中没有使用 |

层级关系可以记成：

```text
advance()
└── 切换 CTA K tile
    └── set_iteration_index()
        └── 定位当前 copy group
            └── operator++()
                └── 遍历 4-half 子访问
```

最终，“16 个子访问”的准确含义是：

```text
每线程
× 每个 operand A stage
× 8 个 ThreadMap 逻辑位置
× 每个位置拆成 2 个 align4 访问
= 16 条 8-byte cp.async
```

而：

```text
kAccessesPerGroupA = 2
```

表示每个 copy group 包含两个 ThreadMap 逻辑 access，对应：

```text
2 个逻辑 access
× 每个逻辑 access 2 个 align4 子访问
= 4 条 cp.async / 线程 / group
```
