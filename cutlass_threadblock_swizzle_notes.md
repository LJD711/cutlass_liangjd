# CUTLASS ThreadblockSwizzle 原理分析

## 1. ThreadblockSwizzle 解决什么问题

CUDA 启动 kernel 后，每个 CTA 可以获得物理坐标：

```cpp
blockIdx.x
blockIdx.y
blockIdx.z
```

但是 CUTLASS 的 GEMM kernel 真正需要知道的是：

```text
当前 CTA 应该计算哪一个逻辑 GEMM tile？
```

CUTLASS 使用 `GemmCoord` 表示逻辑 tile 坐标：

```cpp
threadblock_tile_idx.m()  // GEMM M 方向的 tile 编号
threadblock_tile_idx.n()  // GEMM N 方向的 tile 编号
threadblock_tile_idx.k()  // split-K slice 编号
```

`ThreadblockSwizzle` 负责在两套坐标之间建立映射：

```text
CUDA 物理 blockIdx
        ↓
逻辑 GEMM threadblock tile 坐标
```

它只改变 CTA 访问逻辑 tile 的顺序，不改变卷积或 GEMM 的数学结果。

---

## 2. 完整调用流程

相关代码主要位于：

```text
include/cutlass/conv/device/implicit_gemm_convolution.h
include/cutlass/conv/kernel/implicit_gemm_convolution.h
include/cutlass/gemm/threadblock/threadblock_swizzle.h
```

整体调用链如下：

```text
用户构造卷积 Arguments
        ↓
device::ImplicitGemmConvolution::initialize()
        ↓
构造 UnderlyingKernel::Params
        ↓
计算 implicit_gemm_problem_size
        ↓
计算逻辑网格 grid_tiled_shape
        ↓
计算 swizzle_log_tile
        ↓
device::ImplicitGemmConvolution::run()
        ↓
get_grid_shape(grid_tiled_shape)
        ↓
得到 CUDA 物理启动网格 grid
        ↓
Kernel<<<grid, block, smem>>>(params_)
        ↓
每个 CTA 获得自己的 blockIdx
        ↓
get_tile_offset(swizzle_log_tile)
        ↓
得到逻辑坐标 threadblock_tile_idx
        ↓
根据逻辑坐标构造 IteratorA、IteratorB
        ↓
CTA 计算对应的 GEMM/卷积输出 tile
```

理解这段代码时，必须区分下面三个概念：

| 名称 | 含义 | 所在位置 |
|---|---|---|
| `grid_tiled_shape` | GEMM 被切分后需要多少个逻辑 CTA tile | host/device 参数 |
| `grid` / `gridDim` | CUDA kernel 实际启动的物理网格 | host 启动代码 |
| `threadblock_tile_idx` | 当前 CTA 最终负责的逻辑 GEMM tile | device kernel |

当没有实际 swizzle 时，三者看起来很接近；启用 swizzle 后，物理 `blockIdx` 和逻辑 `threadblock_tile_idx` 不再相同。

---

## 3. Conv2d Fprop 如何映射成 GEMM

对于 Conv2d Fprop，CUTLASS 使用 implicit GEMM：

```text
GEMM M = N × P × Q
GEMM N = K
GEMM K = R × S × C
```

其中：

| 参数 | 含义 |
|---|---|
| `N` | batch size |
| `P, Q` | 输出特征图的高度和宽度 |
| `K` | 输出通道数 |
| `R, S` | 卷积核的高度和宽度 |
| `C` | 输入通道数 |

逻辑矩阵为：

```text
A: NPQ × RSC    activation 的隐式 im2col 视图
B: RSC × K      filter
D: NPQ × K      output
```

这里并不会真的生成一个完整的 im2col 矩阵。Activation iterator 根据逻辑 GEMM 坐标，动态计算对应的 NHWC tensor 地址。

---

## 4. 第一步：initialize() 构造 Params

在：

```text
include/cutlass/conv/device/implicit_gemm_convolution.h
```

`initialize()` 中执行：

```cpp
params_ = typename UnderlyingKernel::Params(
    args,
    static_cast<int *>(workspace)
);
```

这里会进入：

```text
include/cutlass/conv/kernel/implicit_gemm_convolution.h
```

中的 `ImplicitGemmConvolution::Params` 构造函数。

`Params` 构造函数首先根据卷积问题得到 implicit GEMM 大小：

```cpp
implicit_gemm_problem_size(
    cutlass::conv::implicit_gemm_problem_size(
        kConvolutionalOperator,
        args.problem_size
    )
)
```

接着计算逻辑 CTA 网格：

```cpp
grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
    implicit_gemm_problem_size,
    {
        ThreadblockShape::kM,
        ThreadblockShape::kN,
        ThreadblockShape::kK
    },
    args.problem_size.split_k_slices
);
```

---

## 5. grid_tiled_shape 是逻辑 CTA 网格

`GemmIdentityThreadblockSwizzle::get_tiled_shape()` 的主要计算是：

```cpp
return GemmCoord(
    ceil_div(problem_size.m(), tile_size.m()),
    ceil_div(problem_size.n(), tile_size.n()),
    split_k_slices
);
```

等价于：

```text
grid_tiled_shape.m =
    ceil(GEMM_M / ThreadblockShape::kM)

grid_tiled_shape.n =
    ceil(GEMM_N / ThreadblockShape::kN)

grid_tiled_shape.k =
    split_k_slices
```

这里得到的是逻辑网格，还不是传给 CUDA kernel 的 `dim3 grid`。

### 数值例子

假设卷积参数为：

```text
N = 1
P = 56
Q = 56
C = 64
K = 128
R = 3
S = 3
split_k_slices = 1
```

implicit GEMM 大小为：

```text
GEMM M = N × P × Q
       = 1 × 56 × 56
       = 3136

GEMM N = K
       = 128

GEMM K = R × S × C
       = 3 × 3 × 64
       = 576
```

假设：

```cpp
using ThreadblockShape =
    cutlass::gemm::GemmShape<128, 128, 64>;
```

那么：

```text
grid_tiled_shape.m = ceil(3136 / 128) = 25
grid_tiled_shape.n = ceil(128  / 128) = 1
grid_tiled_shape.k = 1
```

最终：

```text
grid_tiled_shape = (25, 1, 1)
```

含义是：

```text
M 方向需要 25 个 CTA tile
N 方向需要  1 个 CTA tile
K 方向需要  1 个 split-K slice
```

每个 CTA 负责一个最大为 `128 × 128` 的输出 tile，并沿 GEMM K 方向循环处理数据。

---

## 6. swizzle_log_tile 是什么

计算完 `grid_tiled_shape` 后，`Params` 构造函数继续执行：

```cpp
swizzle_log_tile =
    threadblock_swizzle.get_log_tile(grid_tiled_shape);
```

`swizzle_log_tile` 保存的是 swizzle 分组宽度的以 2 为底的对数：

```text
swizzle_width = 1 << swizzle_log_tile
```

对应关系为：

| `swizzle_log_tile` | `swizzle_width` |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 4 |
| 3 | 8 |

例如：

```text
swizzle_log_tile = 2
```

表示：

```text
swizzle_width = 1 << 2 = 4
```

### 为什么保存 log2，而不直接保存宽度

因为宽度是 2 的幂，device 端可以使用移位和位运算完成除法与取模：

```cpp
x / width
```

可以写成：

```cpp
x >> swizzle_log_tile
```

而：

```cpp
x % width
```

可以写成：

```cpp
x & (width - 1)
```

例如宽度为 4：

```cpp
x / 4 == x >> 2;
x % 4 == x & 3;
```

---

## 7. GemmIdentityThreadblockSwizzle<N> 中的 N

定义为：

```cpp
template <int N = 1>
struct GemmIdentityThreadblockSwizzle;
```

这里的模板参数 `N` 不是 GEMM problem 的 N 维大小。

它表示：

```text
允许使用的最大 swizzle 分组宽度
```

例如：

```cpp
GemmIdentityThreadblockSwizzle<4>
```

表示最多可以使用宽度为 4 的 swizzle。

`get_log_tile()` 会根据模板参数和逻辑 N tile 数量选择实际宽度：

```cpp
if (N >= 8 && tiled_shape.n() >= 6)
  return 3;
else if (N >= 4 && tiled_shape.n() >= 3)
  return 2;
else if (N >= 2 && tiled_shape.n() >= 2)
  return 1;
else
  return 0;
```

例如：

```text
ThreadblockSwizzle = GemmIdentityThreadblockSwizzle<4>
tiled_shape.n() = 7
```

满足：

```text
N >= 4
tiled_shape.n() >= 3
```

所以：

```text
swizzle_log_tile = 2
swizzle_width = 4
```

如果使用默认类型：

```cpp
GemmIdentityThreadblockSwizzle<>
```

则模板参数采用默认值：

```text
N = 1
```

因此：

```text
swizzle_log_tile = 0
swizzle_width = 1
```

此时没有发生实际的 CTA tile 重排。

---

## 8. 第二步：run() 根据逻辑网格计算物理 CUDA grid

在：

```text
include/cutlass/conv/device/implicit_gemm_convolution.h
```

`run()` 中执行：

```cpp
ThreadblockSwizzle threadblock_swizzle;

dim3 grid =
    threadblock_swizzle.get_grid_shape(
        params_.grid_tiled_shape
    );
```

这就是原代码第 324 行附近发生的事情。

`GemmIdentityThreadblockSwizzle::get_grid_shape()` 的实现为：

```cpp
static dim3 get_grid_shape(GemmCoord tiled_shape) {
  int tile = 1 << get_log_tile(tiled_shape);

  return dim3(
      tiled_shape.m() * tile,
      (tiled_shape.n() + tile - 1) / tile,
      tiled_shape.k()
  );
}
```

令：

```text
Mt = grid_tiled_shape.m()
Nt = grid_tiled_shape.n()
Kt = grid_tiled_shape.k()
T  = swizzle_width
```

物理 CUDA grid 为：

```text
grid.x = Mt × T
grid.y = ceil(Nt / T)
grid.z = Kt
```

这一步可以理解为：

```text
逻辑 GEMM tile 网格
        ↓ get_grid_shape()
CUDA 物理启动网格
```

---

## 9. 默认无 Swizzle 时的完整流程

继续使用前面的例子：

```text
grid_tiled_shape = (25, 1, 1)
```

假设使用：

```cpp
GemmIdentityThreadblockSwizzle<>
```

由于默认模板参数是 `1`：

```text
swizzle_log_tile = 0
T = 1 << 0 = 1
```

那么 `get_grid_shape()` 得到：

```text
grid.x = 25 × 1       = 25
grid.y = ceil(1 / 1)  = 1
grid.z = 1
```

即：

```cpp
dim3 grid(25, 1, 1);
```

随后 kernel 启动：

```cpp
cutlass::Kernel<UnderlyingKernel>
    <<<grid, block, smem_size, stream>>>(params_);
```

GPU 创建的 CTA 包括：

```text
blockIdx = (0, 0, 0)
blockIdx = (1, 0, 0)
blockIdx = (2, 0, 0)
...
blockIdx = (24, 0, 0)
```

---

## 10. 第三步：kernel 中调用 get_tile_offset()

每个 CTA 进入：

```text
include/cutlass/conv/kernel/implicit_gemm_convolution.h
```

的 `operator()` 后执行：

```cpp
ThreadblockSwizzle threadblock_swizzle;

cutlass::gemm::GemmCoord threadblock_tile_idx =
    threadblock_swizzle.get_tile_offset(
        params.swizzle_log_tile
    );
```

`GemmIdentityThreadblockSwizzle::get_tile_offset()` 位于：

```text
include/cutlass/gemm/threadblock/threadblock_swizzle.h
```

实现为：

```cpp
static GemmCoord get_tile_offset(int log_tile) {
  int block_idx_x = RematerializeBlockIdxX();
  int block_idx_y = RematerializeBlockIdxY();
  int block_idx_z = RematerializeBlockIdxZ();

  return GemmCoord{
      block_idx_x >> log_tile,
      (block_idx_y << log_tile) +
          (block_idx_x & ((1 << log_tile) - 1)),
      block_idx_z
  };
}
```

其中：

```cpp
RematerializeBlockIdxX() == blockIdx.x
RematerializeBlockIdxY() == blockIdx.y
RematerializeBlockIdxZ() == blockIdx.z
```

令：

```text
T = 1 << log_tile
```

上面的位运算等价于：

```cpp
tile_m = blockIdx.x / T;
tile_n = blockIdx.y * T + blockIdx.x % T;
tile_k = blockIdx.z;
```

这一步可以理解为：

```text
CUDA 物理 blockIdx
        ↓ get_tile_offset()
逻辑 GEMM threadblock_tile_idx
```

---

## 11. 默认无 Swizzle 时 get_tile_offset() 的结果

当：

```text
swizzle_log_tile = 0
T = 1
```

公式变成：

```cpp
tile_m = blockIdx.x / 1;
tile_n = blockIdx.y * 1 + blockIdx.x % 1;
tile_k = blockIdx.z;
```

因为任何整数对 1 取模都是 0：

```cpp
tile_m = blockIdx.x;
tile_n = blockIdx.y;
tile_k = blockIdx.z;
```

例如：

```text
blockIdx = (7, 0, 0)
```

得到：

```text
threadblock_tile_idx = (7, 0, 0)
```

如果 ThreadblockShape 为：

```text
128 × 128 × 64
```

那么该 CTA 负责的 GEMM 输出区域为：

```text
M 范围：
  [tile_m × 128, (tile_m + 1) × 128)
  = [7 × 128, 8 × 128)
  = [896, 1024)

N 范围：
  [tile_n × 128, (tile_n + 1) × 128)
  = [0, 128)
```

对于 Conv2d Fprop：

```text
GEMM M 对应 NPQ
GEMM N 对应输出通道 K
```

因此该 CTA 负责：

```text
NPQ 逻辑位置 896 到 1023
输出通道 0 到 127
```

---

## 12. 宽度为 4 的 Swizzle 完整例子

为了观察真正的坐标重排，假设：

```text
grid_tiled_shape = (3, 7, 1)
ThreadblockSwizzle = GemmIdentityThreadblockSwizzle<4>
```

由于逻辑 N 方向有 7 个 tile：

```text
swizzle_log_tile = 2
swizzle_width = 1 << 2 = 4
```

### 12.1 get_grid_shape()

物理 CUDA grid 为：

```text
grid.x = 3 × 4       = 12
grid.y = ceil(7 / 4) = 2
grid.z = 1
```

所以：

```cpp
dim3 grid(12, 2, 1);
```

物理网格一共有：

```text
12 × 2 = 24 个 CTA
```

逻辑有效 tile 数量为：

```text
3 × 7 = 21 个 tile
```

多出来的 3 个 CTA 会在 kernel 的边界检查中提前退出。

### 12.2 get_tile_offset()

映射公式为：

```cpp
tile_m = blockIdx.x / 4;
tile_n = blockIdx.y * 4 + blockIdx.x % 4;
tile_k = blockIdx.z;
```

部分映射结果：

| `blockIdx.x` | `blockIdx.y` | `tile_m` | `tile_n` |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 0 | 2 |
| 3 | 0 | 0 | 3 |
| 4 | 0 | 1 | 0 |
| 5 | 0 | 1 | 1 |
| 6 | 0 | 1 | 2 |
| 7 | 0 | 1 | 3 |
| 8 | 0 | 2 | 0 |
| 9 | 0 | 2 | 1 |
| 0 | 1 | 0 | 4 |
| 1 | 1 | 0 | 5 |
| 2 | 1 | 0 | 6 |
| 3 | 1 | 0 | 7，无效 |

例如：

```text
blockIdx = (5, 1, 0)
```

计算过程为：

```text
tile_m = 5 / 4
       = 1

tile_n = 1 × 4 + 5 % 4
       = 4 + 1
       = 5

tile_k = 0
```

最终：

```text
threadblock_tile_idx = (1, 5, 0)
```

如果 ThreadblockShape 为 `128 × 128 × 64`，该 CTA 负责：

```text
GEMM M 范围：
  [1 × 128, 2 × 128)
  = [128, 256)

GEMM N 范围：
  [5 × 128, 6 × 128)
  = [640, 768)
```

---

## 13. 为什么会产生无效 CTA

前面的例子中：

```text
Nt = 7
swizzle_width = 4
```

物理 `grid.y` 必须向上取整：

```text
grid.y = ceil(7 / 4) = 2
```

两组每组可以表示 4 个 N tile：

```text
第 0 组：tile_n = 0, 1, 2, 3
第 1 组：tile_n = 4, 5, 6, 7
```

但逻辑上只有：

```text
tile_n = 0 到 6
```

因此 `tile_n = 7` 是填充出来的无效 tile。

kernel 中必须进行边界检查：

```cpp
if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
    params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
  return;
}
```

这也是 `get_log_tile()` 不盲目使用最大宽度的原因之一。过大的宽度可能制造过多无效 CTA。

---

## 14. threadblock_tile_idx 后续如何使用

`get_tile_offset()` 得到逻辑 tile 坐标后，kernel 使用它构造 A、B iterator。

### IteratorA

对于 fprop，A 是 activation 的逻辑 `NPQ × RSC` 矩阵：

```cpp
typename Mma::IteratorA iterator_A(
    params.iterator_A,
    params.problem_size,
    params.ptr_A,
    thread_idx,
    MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
    )
);
```

其中：

```text
threadblock_tile_idx.m() × Mma::Shape::kM
```

确定当前 CTA 从哪个 `NPQ` 位置开始。

### IteratorB

对于 fprop，B 是 filter 的逻辑 `RSC × K` 矩阵：

```cpp
typename Mma::IteratorB iterator_B(
    params.iterator_B,
    params.problem_size,
    params.ptr_B,
    thread_idx,
    MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
    )
);
```

其中：

```text
threadblock_tile_idx.n() × Mma::Shape::kN
```

确定当前 CTA 从哪个输出通道 K 开始。

而：

```text
threadblock_tile_idx.k() × Mma::Shape::kK
```

确定 split-K slice 在 GEMM K/RSC 方向的起始位置。

---

## 15. 为什么要进行 Swizzle

### 15.1 Swizzle 的核心目标

Swizzle 的主要目标是：

```text
调整物理上相邻 CTA 对逻辑 GEMM tile 的访问顺序，
让这些 CTA 更可能复用相同的输入数据，
从而提高 L2 cache 命中率并减少显存流量。
```

它是一种 CTA 调度顺序和数据局部性优化。

---

## 16. 无 Swizzle 时的数据复用关系

无 swizzle 时：

```cpp
tile_m = blockIdx.x;
tile_n = blockIdx.y;
```

当 `blockIdx.x` 连续变化时，逻辑 tile 顺序为：

```text
(M0, N0)
(M1, N0)
(M2, N0)
(M3, N0)
```

对应矩阵乘法：

```text
A0 × B0
A1 × B0
A2 × B0
A3 × B0
```

这些 CTA：

```text
M tile 不同
N tile 相同
```

所以它们主要共享相同的 B tile。

对于 Conv2d Fprop：

```text
A = activation
B = filter
```

因此这种顺序更倾向于复用相同的 filter 区域。

---

## 17. 有 Swizzle 时的数据复用关系

假设 swizzle 宽度为 4，物理上连续的部分 CTA 映射为：

```text
(M0, N0)
(M0, N1)
(M0, N2)
(M0, N3)
```

对应矩阵乘法：

```text
A0 × B0
A0 × B1
A0 × B2
A0 × B3
```

这些 CTA：

```text
M tile 相同
N tile 不同
```

因此它们主要共享相同的 A tile。

对于 Conv2d Fprop：

```text
A = activation 的 NPQ × RSC 逻辑矩阵
```

同一个 M tile 代表相同的一组输出空间位置 `NPQ`。

不同 N tile 代表不同的一组输出通道 K。

所以 swizzle 后，相邻 CTA 可能执行：

```text
相同的 activation 区域
×
不同的 filter/output-channel 区域
```

这使得 activation 数据更可能保留在 L2 cache 中，并被后续 CTA 重用。

---

## 18. Swizzle 为什么可能提升 L2 Cache 命中率

每个 CTA 会把自己的 A、B tile 从 global memory 搬运到 shared memory。

不同 CTA 不能共享 shared memory，但它们共享 GPU 的 L2 cache。

如果多个时间上较接近的 CTA 读取相同的 global-memory 数据：

```text
第一个 CTA：
  HBM/global memory → L2 → SM

后续 CTA：
  L2 → SM
```

后续 CTA 可能不需要再次从更慢的 HBM 中读取同一份数据。

需要注意：

```text
CUDA 不保证 CTA 严格按照 blockIdx 顺序执行。
```

因此 swizzle 不是对实际执行顺序的严格保证，而是一种改善调度局部性的映射策略。

它不参与正确性，只影响性能。

---

## 19. 为什么不能一直使用最大的 Swizzle 宽度

更大的 swizzle 宽度不一定始终更快，原因包括：

1. N 方向 tile 太少时，大宽度会制造较多无效 CTA。
2. 强化 A 数据复用的同时，可能降低 B 数据的局部性。
3. 不同 GEMM 长宽比例适合不同的 CTA 遍历方向。
4. 实际收益还受到 GPU 架构、L2 容量、tile 大小和 CTA 调度影响。

因此 CUTLASS 根据 `tiled_shape.n()` 选择实际宽度：

```text
N tile 很少：
  使用 width=1，不进行重排

N tile 足够多：
  使用 width=2、4 或 8
```

源码中的阈值还考虑了不要产生过多 no-op CTA：

```cpp
// Thresholds picked so that it doesn't cause too many no-op CTAs
```

---

## 20. Swizzle 不会改变什么

Swizzle 不会改变：

```text
GEMM 数学结果
卷积数学结果
ThreadblockShape
WarpShape
InstructionShape
Tensor Core 指令
每个 CTA 内部的 MMA mainloop
最终输出 tensor 布局
```

它只改变：

```text
哪个物理 blockIdx 负责哪个逻辑 GEMM tile
```

所以可以将它理解为：

```text
逻辑工作完全相同，只是重新安排 CTA 领取工作的编号方式。
```

---

## 21. get_grid_shape() 与 get_tile_offset() 的关系

这两个函数必须使用完全一致的 swizzle 规则。

### get_grid_shape()

运行在 host 侧，负责：

```text
逻辑 GEMM tile 网格
        ↓
CUDA 物理启动网格
```

### get_tile_offset()

运行在 device 侧，负责：

```text
CUDA 物理 blockIdx
        ↓
逻辑 GEMM tile 坐标
```

可以将它们理解为配套的编码和解码过程：

```text
grid_tiled_shape
        ↓ get_grid_shape()
physical CUDA grid
        ↓ kernel launch
blockIdx
        ↓ get_tile_offset()
threadblock_tile_idx
```

如果两者的规则不一致，就可能出现：

```text
某些逻辑 tile 被重复计算
某些逻辑 tile 没有 CTA 计算
CTA 访问错误的矩阵区域
输出结果错误
```

---

## 22. 最终总结

`grid_tiled_shape` 表示逻辑工作量：

```text
GEMM 一共需要多少个 CTA tile
```

`get_grid_shape()` 根据 swizzle 规则将逻辑工作量编码成 CUDA 物理网格：

```text
grid_tiled_shape → gridDim
```

CUDA 启动 kernel 后，每个 CTA 得到物理坐标：

```text
blockIdx
```

`get_tile_offset()` 再将物理坐标解码成逻辑 GEMM tile：

```text
blockIdx → threadblock_tile_idx
```

kernel 最终使用 `threadblock_tile_idx` 确定：

```text
IteratorA 从哪个 NPQ/RSC 位置开始
IteratorB 从哪个 RSC/K 位置开始
当前 CTA 计算哪个输出 tile
当前 CTA 属于哪个 split-K slice
```

进行 swizzle 的主要原因是：

```text
重新组织物理上相邻 CTA 对逻辑 tile 的访问顺序，
提高 CTA 之间对 activation 或 filter 数据的 L2 cache 复用，
从而减少 global-memory/HBM 流量并改善 kernel 性能。
```

最简化的记忆方式是：

```text
get_grid_shape():
  逻辑网格 → 物理 CUDA 网格

get_tile_offset():
  物理 blockIdx → 逻辑 GEMM tile

swizzle:
  不改变计算内容，只改变 CTA 领取逻辑 tile 的顺序
```
