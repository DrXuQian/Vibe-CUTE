# Marlin

```cpp
THREAD_M_BLOCKS 1 (batch = 1)
THREAD_N_BLOCKS 8 (threadblock.n = 128)
THREAD_K_BLOCKS 8 (threadblock.k = 128)
STAGES 4
GROUP_BLOCKS 8 (suppose group = 128)
THREADS 256
```

# 成员函数

## Threadblock tile size

对于 k 和 n 的 tile size，采用 heuristic 的选择方法，这个是 marlin 原始的代码中的选择方案，但是 vllm 中也没有对于这个参数进行 tuning。可以认为这里就设置为这些固定值就可以了。

每个 tile 可以分为多个 block，每个 block 是 16x16x16。最终计算的时候用的是 m16n8k16，但是会用两个 mma 分别计算两个 n=8，也就是拼成了 m16n16k16。

Threadblock tile size 是单个 CTA 在单个 Stage 计算的数据量。

```python
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      // Small-M: favor a smaller N tile and larger K tile (better latency for GEMV-ish shapes).
      thread_k = 128;
      thread_n = 128;
    } else {
      // Larger-M: favor a larger N tile (better throughput).
      thread_k = 64;
      thread_n = 256;
    }
  }
```

thread_n_blocks 和 thread_k_blocks 可以根据上面的计算逻辑得出。也就是 m 比较小的时候，选择 thread_n_blocks=thread_k_blocks=8。当 m 比较大的时候，选择更大的 thread_n_blocks=16，thread_k_blocks=8。

对于 thread\_m\_blocks，thread\_m\_blocks 最大是 4。每个 warp 计算的时候会遍历所有的 thread_m_blocks 来计算所有的 m 的输出。也就是从 CTA 的 tile 切分到 warp 的时候不会切分 m 维度，只会切分 n 和 k 维度。

另外 prob_m 总共最多是 64 * max_par，也就是 64 * 16 = 16 个 thread_m_blocks。如果原本的 prob_m 超过 1024，那么会分成多个 kernel 来 launch，每个都是最大 1024 的 prob_m。

* why？
    * 因为更大的 thread\_m\_blocks 会导致寄存器爆炸
    * par 用来控制 M 维度上的 threadblock tile 的数量。为什么需要 max\_par，因为 workspace 需要 `n / 128 * max_par` 个 int 用于 lock 同步，`max_par` 太大会浪费内存，太小会限制并行度

然后是一个 heuristic 进行模板初始化的启动代码：

可以看到 nb 和 kb 只有两种配置，8，8 或者 16，4。

mb 可以是 1-4。gb 可以是 -1 或者 8。分别表示 per-column 和 groupsize=128。

```python
            mb  nb  kb  gb
    CALL_IF(1,  8,  8, -1)
    CALL_IF(1,  8,  8,  8)
    CALL_IF(1, 16,  4, -1)
    CALL_IF(1, 16,  4,  8)
    CALL_IF(2, 16,  4, -1)
    CALL_IF(2, 16,  4,  8)
    CALL_IF(3, 16,  4, -1)
    CALL_IF(3, 16,  4,  8)
    CALL_IF(4, 16,  4, -1)
    CALL_IF(4, 16,  4,  8)
```

其他的启动参数，都是固定写死的：

```c++
const int THREADS = 256;  // threads per CTA
const int STAGES = 4;  // cp.async pipeline stages
const int SHARED_MEM = 96 * 1024;  // 96 KB shared memory
```

ppu 的 shm 大小更大，可以思考一下这部分能不能优化。

```c++
Marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>      
  <<<blocks, THREADS, SHARED_MEM, stream>>>(A_ptr, B_ptr, C_ptr, s_ptr, prob_m, prob_n, prob_k, locks); 
```

## CTA Dispatcher

![](attachment/b675a2fd620fba40a04775a62545a99c.xlsx)

参考上面的 excel。这里总共 20 个 SM，每个 SM 均匀分配了 26 个 tile。每个 tile 是 thread_m_blocks x thread_n_blocks x thread_k_blocks 定义的块。每个块对应了 16x16x16 大小的矩阵计算。这个 excel 上的每个方框代表了一个 tile。这个 excel 总共的大小可以表示为 rest N * rest M 行，总共 rest K 列。

![](attachment/60ede3e1e15e6bed95819ccd457f9f43.png)

对于一些参数的解释可以看下面的表格。下面的表格以 n=k=2048 为例。假设 parallel=2，也就是 m 维度上有两个 thread_m_blocks（假设 thread_m_blocks=4,也就是一个 thread_m_blocks 对应了 64 个 m 维度，那么 prob_m=2* thread_m_blocks=128）。

第一次 init slice 的时候，

- slice_col_par:
	- 每个 CTA 对应的 slice_col_par 就是自己在第几行（总共 rest M * rest N 行）。每行是对应了所有的 rest K。可以看到，CTA0 对应了第 0 行，CTA1 在第一次 init slice 对应了第 1 行。CTA2 对应了第 3 行。依次类推。
- M 区域：
	- 可以看到在 CTA=9 的时候，跨过了两个 rest M。但是此时是第一次迭代，也就是 CTA9 这个时刻对应的计算区域还是 rest M=0。
	- CTA=10，对应了 rest M = 1。
- slice_col:
	- 对应了第几个 rest N。
- slice_row:
	- 对应了第几个 rest K。
- slice_iters:
	- 当前的 CTA 在当前的行上还剩余多少次迭代（每次迭代计算一个 tile）。
- slice_count:
	- 当前的行上总共需要多少次迭代。
- slice_idx:
	- 当前的 CTA 在行上是第几个 CTA。（用于后面的 CTA 间的同步）

![](attachment/9290f30d361e13d30eb9282cda9ad89f.png)

同理可得第二次 init slice 和第三次 init slice 的变化情况：

![](attachment/5318643c633a51417a7be972f24d265c.png)

### 代码走读

#### offset 初始化

计算每个 CTA 需要多少次 iteration 才能遍历完所有的 tiles。

```c++
int k_tiles = prob_k / 16 / thread_k_blocks;  // number of K tiles for this launch
int n_tiles = prob_n / 16 / thread_n_blocks;  // number of N tiles for this launch
int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);  // stripe length per CTA (in K-tiles)
```

然后计算上面的 dispatcher 需要的一些参数：

```c++
int slice_row = (iters * blockIdx.x) % k_tiles;  // tile row in K grid (idx%k_tiles)
int slice_col_par = (iters * blockIdx.x) / k_tiles;  // tile col in N grid (idx/k_tiles, incl parallel)
int slice_col = slice_col_par;  // tile col (N tile index)
int slice_iters;  // K-tiles this CTA will process for current N tile
int slice_count = 0;  // num blocks contributing to this col
int slice_idx;  // this CTA's index within slice_count (barrier order)
```

如果 m 维度的 parallel 大于 1，那么需要把对应的 A 和 C 矩阵的 offset 设置到对应的 m 维度。

```c++
if (slice_col_par >= n_tiles) {
	//slice_col_par / n_tiles 就是第几个parallel
	A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
	C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
	locks += (slice_col_par / n_tiles) * n_tiles;
	slice_col = slice_col_par % n_tiles;  // tile col (N tile index)
}
```

另外，这里的 locks 需要挪到对应的位置。slice_col 也需要更新。

#### init slice

从代码来看，`init_slice` 是一个 lambda 函数，它会修改以下**外部变量**（通过引用捕获 `[&]`）：

| 变量 | 作用 | 外部用途 |
|------|------|----------|
| `slice_iters` | 当前 CTA 在当前 N tile 处理的 K-tile 数量 | 控制主循环的迭代次数 |
| `slice_count` | 贡献到当前 N tile 的 CTA 总数 | 归约同步时判断需要等待多少个 CTA |
| `slice_idx` | 当前 CTA 在归约中的编号 | 决定归约顺序（谁先写、谁后归约）|
| `slice_col` | 当前 N tile 索引 | 用于计算 B、C 指针偏移和 locks 索引 |
| `A` | A 矩阵指针 | 可能在切换 M 区域时被偏移 |
| `C` | C 矩阵指针 | 可能在切换 M 区域时被偏移 |
| `locks` | 归约同步锁指针 | 可能在切换 M 区域时被偏移 |

最重要的三个输出是：

- **`slice_iters`**：告诉主循环 " 当前 CTA 在这一轮处理多少个 K-tile"，从图上可以看到，对于红色的块对应的 CTA，slice_iters=2。而对于绿色块对应的 CTA，当前的 N-tile 处理的 slice_iters=1。对于下一个 N-tile，绿色的 CTA 处理的 slice_iters 还是等于 1。
- **`slice_count`** 和 **`slice_idx`**：告诉归约逻辑 " 对于当前的 N-tile 的输出，有几个 CTA 参与，我是第几个 "。对于图上的 N-tile 和 N+1-tile，slice_count 都是 3。slice_idx 分别从 0 到 3 分配给不同的 CTA。

![](attachment/a240473cd8327681c58ce5003293db99.png)

一个具体的例子，对于 128 * 128 的 B 矩阵，20 个 SM（thor 的配置）下，init slice 之后的不同的 SM 分配到的任务是：

![](attachment/b675a2fd620fba40a04775a62545a99c.xlsx)

代码走读：

其中 slice_iters 和 slice_count 以及 slice_idx 可以从上面的图推测出来。

```c++
auto init_slice = [&] () {  // compute slice_iters/slice_count/slice_idx for this CTA
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);  // tiles this block does in current col
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;  // tiles this block does in current col
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;  // tiles this block does in current col
    slice_count = 1;  // num blocks contributing to this col
    slice_idx = 0;  // this block order within slice (for barrier)

    // If `iters < k_tiles`, multiple CTAs will land in the same slice_col_par and must be reduced.
    // This computes:
    //   - slice_count: total CTAs that touch this N tile
    //   - slice_idx:   this CTA's rank within that set (used by locks[slice_col])
    // col_first: 第一个完整落在当前 N tile 内的 CTA 的起始位置
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);  // num blocks contributing to this col
      if (col_off > 0)
        slice_count++;  // account for partial first segment
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
	        slice_idx = slice_count - 1;  // this block order within slice (for barrier)
      else {
        slice_idx = slice_count - 1 - delta_first / iters;  // this block order within slice (for barrier)
        if (col_off > 0)
          slice_idx--;  // adjust for non-zero col_off
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;  // tile col (N tile index)
    }
  };  // end init_slice
  init_slice();  // initialize slice state

```

另外，A 和 C 矩阵的 offset，如果遇到跨越 M-tile 的边界的时候，需要调整。例如下面的 CTA Dispatcher 中的 CTA9。locks 也需要对应修改。

## GMEM to SMEM

![](attachment/48dad17cc5fd5ed8f350b94192072e67.xlsx)

### A matrix gmem to smem

其中 smem 分配给 A 矩阵的大小是：

`a_sh_stride * (16 * thread_m_blocks);`

对于 gmem 到 smem 的拷贝，以及 smem 到 reg 的拷贝，有如下几个关键参数：

其中：

- a_gl_stride 是 gmem 上 A 的行 stride。也就是从一行到下一行，地址加多少。每行是一个完整的 K。这里的单位是 int4，注意是 4 个 32-bit，不是 4-bit。
- a_sh_stride 是 smem 上的 A 矩阵的行的 stride。也就是每行是多长。这里表示 16 个 int4。也就是 64B。或者说是一个 thread_k_block 的大小。
- a_gl_rd_delta_o 是读取一个 k-block 需要多少个 thread。每个 thread 读取的内容是 int4，也就是 4B。这里 16 个 thread 配合读取一行。outerloop 是多个 stage 之间的 gmem 的 offset。
- a_gl_rd_delta_i 是第二次迭代和第一次迭代在 gmem 上的间隔。注意每 16 个 thread 读取一行，256 个 thread 读取 16 行，那么 16 * 256 = 4096 就是多次读取之间的 gmem 的间隔。innerloop 是一个 stage 内部的多次 iteration 的 offset。
- a_sh_wr_delta 是多次写入 smem 的间隔。
- a_sh_wr_iters 是总共需要迭代多少次。这里需要迭代 4 次来写完所有的数据。

![](attachment/e3cfe56c412f44ca11d5378375cf220a.png)

然后，实际拷贝的时候，对于每个 CTA，总共是 8 个 warp。每个 warp 拷贝的内容是 2 * 16 个 int4。对应的也就是两行数据。对应拷贝到 smem 的时候也是类似的，但是添加了 swizzle。swizzle 的逻辑是一个通用逻辑，这里不赘述。

![](attachment/0c7e05e28e815bb995d1b7251cbf0c8b.png)

其中：

![](attachment/eb455892359b422795e18ab6017a194b.png)

### A matrix smem to reg

A 矩阵对应到 frag，对于每个 thread：

```c++
using FragA = Vec<half2, 4>;      // Input A fragment: 4x half2 = 16x FP16
using FragB = Vec<half2, 2>;      // Weight B fragment: 2x half2 = 4x FP16
using FragC = Vec<float, 4>;      // Accumulator fragment: 4x float
using FragS = Vec<half2, 1>;      // Scale fragment: 1x half2 = 2x FP16

FragA frag_a[2][thread_m_blocks];  // regs: A fragments (ping-pong over k)
I4 frag_b_quant[2];  // regs: packed INT4 B fragments (ping-pong over k)
FragC frag_c[thread_m_blocks][4][2];  // regs: accumulators [m_block][n_subtile][b_half]
FragS frag_s[2][4];  // regs: scale fragments (ping-pong over k)

```

smem 到 reg 的拷贝同样需要一些参数，下面讲解这些参数：

- 需要多少个 warp 来计算 n 维度：因为每个 warp 计算的 thread_n_blocks 的数量是固定的（代码写死），就是 4 个。对于 thread_n_blocks=8 来说，n 维度上分配的 warp 就是 2 个。然后分配到 m 维度的 warp 数量是所有的，因为所有的 warp 都会计算所有的 thread_m_blocks。那么分配到 k 上的 warp 数量就是 8/2=4。因为总共是 8 个 warp，每个 n 上分配两个 warp，那么每个 k 上就是 4 个 warp。
- 对照下面图更容易理解。

![](attachment/26ce69c592f7d83e2e5e87eb74facdcf.png)

- a_sh_rd_delta_o 是 2 * k 维度分配的 warp 数量，乘 2 是因为 ldsmx4 中会加载一行 16 个 fp16，对应了两个 int4。outer loop 的 offset。
- a_sh_rd_delta_i 是 shm 上下一个 m-block 的 stride。如图所示，就是 ldsmx4 的下一个加载的 offset。是 inner loop 的 offset。
- a_sh_stage 是每个流水线 stage 需要的 shm 的大小

![](attachment/4b755a5ab4a7ada9ee5358fa7423a732.png)

其中：

![](attachment/1ad28332ea3c5dc31f12f5dd5a9b8be7.png)

### B matrix gmem to smem

B 矩阵拷贝到 smem 比较特殊。因为 B 矩阵是 4-bit，所以有一些特殊的 layout。首先讲一下 4-bit 的数据如何 pack：

![](attachment/b270a09bb4328e08f3727eba4c118ae5.png)

上面是原始的数据，n 维度连续。下面的数据是 pack 之后的数据。可以看到，第一个 32-bit 的数值包括了 8 个原始的数据中的数值，就是标绿部分的 8 个数值。后面是黄色的部分的 8 个数值，依次类推。可以看到一个 32-bit 的数值中 pack 了 2n * 4k 的数据。第一个 int4 中 pack 了 2n * 16k 的数据。

B 的数据拷贝同样也存在多个关键参数：

![](attachment/bfcecbe859d2c7261d45ed38c80c0395.png)

- b_gl_stride 是 gmem 上的 stride 大小。一个 stride 包含了全部的 n。另外由于这里的格式的特殊性，这里包含了 16 个 k。
- b_sh_stride 是处理 16x thread_n_blocks 需要一个 CTA 的多少个 threads。或者说下一个 16x thread_n_blocks 的 offset。
- b_gl_rd_delta_o 是下一个 stage 的 offset。是 outer loop 的 offset。
- b_gl_rd_delta_i 是 CTA 内部 warp 的下一次 iteration 的 offset。是 inner loop 的 offset。
- b_sh_wr_delta 是 thread 数量。是 CTA 下一次写入 shm 的 offset。
- b_sh_rd_delta 同上。是 CTA 下一次读 shm 的 offset。
- b_sh_stage 是一个 stage 需要多大的 shm 的 storage。
- b_sh_wr_iters 是需要多少次迭代。
![](attachment/2cb2b0433031ed61e4108ad6c73b6485.png)

其中：

![](attachment/c93556943b26a8a2f6aac438751ce5f3.png)

### 代码走读

#### offset 初始化

下面的初始化可以参考上面的图的计算。这里首先是每个 warp 的计算内容：

```c++
// ========================================================================
  // Memory Access Parameters - A Matrix
  // ========================================================================
  int a_gl_stride = prob_k / 8;  // A row stride in int4 (K/8)
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;  // A shared stride per row in int4
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;  // A global delta (int4) per tid group
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);  // A global delta (int4) within a tile
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);  // A shared write delta (int4) per iter
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));  // A shared read delta across warps
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;  // A shared read delta across m-blocks
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);  // A shared bytes per stage (in int4)
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);  // A shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - B Matrix (Weights)
  // ========================================================================
  int b_gl_stride = 16 * prob_n / 32;  // B stride in int4 (layout-specific)
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;  // B shared stride in int4
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;  // B global delta (int4) per k-tile
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);  // B global delta (int4) per load-iter
  constexpr int b_sh_wr_delta = threads;  // B shared write delta (int4)
  constexpr int b_sh_rd_delta = threads;  // B shared read delta (int4)
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;  // B shared int4 per stage
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;  // B shared write iters per stage

  // ========================================================================
  // Memory Access Parameters - Scales
  // ========================================================================
  int s_gl_stride = prob_n / 8;  // scale row stride in int4 (N/8)
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;  // scale shared stride in int4
  constexpr int s_sh_stage = s_sh_stride;  // scale shared int4 per stage
  int s_gl_rd_delta = s_gl_stride;  // scale global delta per group
```

下面切分到 thread：

```c++
// A matrix indices
int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)
a_gl_rd += a_gl_rd_delta_o * slice_row;  // A gmem read index (int4)
int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A smem write index (int4)
int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;  // A smem read index (int4)
a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));  // A smem read index (int4)

// B matrix indices
int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);  // B gmem read base (int4)
b_gl_rd += b_sh_stride * slice_col;  // B gmem read base (int4)
b_gl_rd += b_gl_rd_delta_o * slice_row;  // B gmem read base (int4)
int b_sh_wr = threadIdx.x;  // B smem write base (int4)
int b_sh_rd = threadIdx.x;  // B smem read base (int4)

// Scale indices
int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
int s_sh_wr = threadIdx.x;  // scale smem write index (int4)
int s_sh_rd;  // scale smem read index (int4)
if (group_blocks != -1)
	s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;  // scale smem read index (int4)
else
	s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;  // scale smem read index (int4)
```

其中对于 a_gl_rd 和 a_sh_wr 都是类似的，256 个 thread 均匀分配到 256 个 int4，图上的数字是 tid：

![](attachment/0d44b2e747055c3dfbe8f9fd00bf995d.png)

另外的 a_sh_rd，就是一个简单的拷贝。每个 int4 对应了 8 个 fp16，需要一个 thread 来拷贝。

![](attachment/03fa4a7ba4131a4af6e4e41df8502432.png)

对于 b_gl_rd，也是类似的：

![](attachment/1d127df6fe90790c449807cf890edce9.png)

另外 b_sh_wr 和 b_sh_rd 就是直接写和读。跟上面也是一样。这是因为已经 offline 重新排序了。这里的每个 element，都对应了 16 个 k。

#### Predication

这部分跳过。

```c++
bool a_sh_wr_pred[a_sh_wr_iters];  // per-iter pred: A gmem loads are in-bounds
  #pragma unroll  // unroll pred init
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;
```

#### Fetch to Shared

注意，这里的 pipe 是 shm 上第几个 stage 的 index（循环使用），a_off 是 gmem 上的下一个 stage 的 index。其实最主要的区别是一个是循环使用的，一个是递增的。但是 marlin 里面的写法有点诡异。后面使用的时候是：

先 start pipe 的时候，load 其中 stage-1 的数据到 shared memory。这里 stage 和 a_off 都是 i。

```c++
	    for (int i = 0; i < stages - 1; i++)  // prefetch (stages-1) K-tiles into shared
	      fetch_to_shared(i, i, i < slice_iters);  // stage=i, a_off=i, pred guards tail tiles
```

后面还有一次调用：

```c++
fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
```

```c++
// Fetch tile to shared memory
  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {  // cp.async A/B(/S) into smem stage=pipe
    if (pred) {
      int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
      #pragma unroll  // unroll A cp.async
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],  // smem dst (swizzled)
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],  // gmem src
          a_sh_wr_pred[i]  // in-bounds pred
        );  // end cp_async4_pred
      }
      int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
      #pragma unroll  // unroll B cp.async
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
        B_ptr[i] += b_gl_rd_delta_o;  // advance B gmem ptr by one k-tile
      }
      // Fetch scales once per quantization group (groupsize = 16 * group_blocks).
      if constexpr (group_blocks != -1) {
        constexpr int GROUP_PIPES = group_blocks / thread_k_blocks;  // K-tiles per scale row
        if (pipe % GROUP_PIPES == 0) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;  // S stage base in smem
          if (s_sh_wr_pred)
            cp_async4_stream(&sh_s_stage[s_sh_wr], &s[s_gl_rd]);
          s_gl_rd += s_gl_rd_delta;
        }
      }
    }
    cp_async_fence();
  };  // end fetch_to_shared
```

注意，这里的 A 和 B 加载还不一样。A 需要用 a_off 去索引到对应的 a 的 gmem。B 用 B_ptr 去索引，然后加上 b_gl_rd_delta_o。

#### Fetch to registers

这里比较关键的逻辑主要是 ldsm4 去加载 frag_a，对应的 register 的映射如上面的分析所示。

另外 b 的加载涉及到解包的过程。这部分参考上面的 lop3。

```c++
// Fetch from shared to registers
  auto fetch_to_registers = [&] (int k, int pipe) {  // load A/B(/S) for k-subtile into regs
    if constexpr (group_blocks != -1) {
      // Scales are stored per-group; pin the stage to the group's first pipe.
      constexpr int GROUP_PIPES = group_blocks / thread_k_blocks;
      int4* sh_s_stage = sh_s + s_sh_stage * (GROUP_PIPES * (pipe / GROUP_PIPES));
      reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];  // load 2x half2 scales
    }
    int4* sh_a_stage = sh_a + a_sh_stage * pipe;  // A stage base in smem
    #pragma unroll  // unroll ldmatrix
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;  // B stage base in smem
    frag_b_quant[k % 2] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);  // load packed INT4 weights
  };  // end fetch_to_regi
```

## Swizzle

这部分也跳过。

```c++
// XOR-based layout to avoid bank conflicts
  auto transform_a = [&] (int i) {  // apply XOR swizzle for A shared layout
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };  // end transform_a
  
 int a_sh_wr_trans[a_sh_wr_iters];  // A smem write indices after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);

  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];  // A smem read indices [k-subtile][m_block] after swizzle
  #pragma unroll  // unroll swizzle init
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll  // unroll m-block init
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  } 
  
```

## 块间同步

### locks

下面是 locks 在全局 mem 上的申请的空间：

```c++
size_t workspace_size = (N / 128) * 16 * sizeof(int);
//                      └───────┘   └──┘
//                       n_tiles   max_par
```

可以看到，实际的 locks 大小就是 n_tiles * m_tiles。这里 reserve 了最多的 parallel 的大小。

### ThreadBlock Reduce

首先计算需要多少次 reduce：

```c++
constexpr int red_off = threads / b_sh_stride / 2;
```

这里计算的 threads / b_sh_stride 就是有多少个 warp 分配到了不同的 K，或者说多少个 warp 之间需要 reduce。在我们的例子中，应该是 4 个 warp 需要 reduce，red_off=2。

完整流程图

```
初始:
  frag_c:  [P0]     [P1]     [P2]     [P3]
           rep0     rep1     rep2     rep3

Step i=2 (rep 2,3 写入 shared):
  frag_c:  [P0]     [P1]     [P2]     [P3]
  shared:  [P2]     [P3]
            ↑        ↑
          rep2写   rep3写

Step i=1 (rep 1 累加并写入):
  frag_c:  [P0]   [P1+P2+P3]
  shared:  [P1+P2+P3]
               ↑
             rep1写

Final (rep 0 累加):
  frag_c:  [P0+P1+P2+P3]
               ↑
           最终结果
```

如果是 8 个 warp 需要 reduce:

 完整流程图

```
初始:
frag_c:  [P0] [P1] [P2] [P3] [P4] [P5] [P6] [P7]
          0    1    2    3    4    5    6    7

Step i=4 (rep 4,5,6,7 写入):
frag_c:  [P0] [P1] [P2] [P3] [P4] [P5] [P6] [P7]
shared:  [P4] [P5] [P6] [P7]
          ↑    ↑    ↑    ↑
        rep4 rep5 rep6 rep7 写入

Step i=2 (rep 2,3 累加并写入):
frag_c:  [P0] [P1] [P2+P4+P6] [P3+P5+P7]
shared:  [P2+P4+P6] [P3+P5+P7]
              ↑          ↑
            rep2       rep3 写入

Step i=1 (rep 1 累加并写入):
frag_c:  [P0] [P1+P2+…+P7]
shared:  [P1+P2+…+P7]
              ↑
            rep1 写入

Final (rep 0 累加):
frag_c:  [P0+P1+P2+P3+P4+P5+P6+P7]  ✓
```

这里的 shm 是复用了 A 的 shm 的空间。

```c++
auto thread_block_reduce = [&] () {  // CTA-local reduce over red_idx replicas
    constexpr int red_off = threads / b_sh_stride / 2;  // initial tree offset = (num_replicas / 2)
    if (red_off >= 1) {  // only needed when we actually have multiple replicas
      int red_idx = threadIdx.x / b_sh_stride;  // replica id in [0, threads/b_sh_stride)
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;  // int4s per replica buffer in shared (8 * b_sh_stride)
      constexpr int red_sh_delta = b_sh_stride;  // int4 stride between j-slices within one replica buffer
      int red_sh_rd = red_sh_stride * red_idx + (threadIdx.x % b_sh_stride);  // this thread's base int4 slot in shared

      #pragma unroll  // unroll over m-blocks (each m_block is 16 rows)
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {  // reduce per m_block independently
        #pragma unroll  // unroll tree steps
        for (int i = red_off; i > 0; i /= 2) {  // i = reduction distance between source/dest replica groups
          if (i <= red_idx && red_idx < 2 * i) {  // upper-half replica groups write/reduce into lower-half
            #pragma unroll  // unroll over (n_subtile * b_half) = 4 * 2 = 8 fragments
            for (int j = 0; j < 4 * 2; j++) {  // j enumerates the 8 FragC fragments per m_block
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);  // dest shared slot (red_idx -> red_idx-i)
              if (i < red_off) {  // not the first step: shared already contains partials to accumulate
                float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * j + red_sh_rd]);  // src partial (this red_idx)
                float* c_wr = reinterpret_cast<float*>(&sh[red_sh_wr]);  // dst partial (red_idx-i) from previous step
                #pragma unroll  // unroll 4-lane FragC accumulation
                for (int k = 0; k < 4; k++) {  // k enumerates the 4 float lanes inside one FragC
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];  // local += src + dst
                }
              }
              sh[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];  // write updated partial to shared
            }
          }
          __syncthreads();  // make partials visible before the next tree step
        }
        if (red_idx == 0) {  // replica-0 threads finalize by adding the last partial stored in shared
          #pragma unroll  // unroll over 8 fragments
          for (int i = 0; i < 4 * 2; i++) {  // i enumerates the 8 FragC fragments per m_block
            float* c_rd = reinterpret_cast<float*>(&sh[red_sh_delta * i + red_sh_rd]);  // reduced partial from other replicas
            #pragma unroll  // unroll 4-lane FragC accumulation
            for (int j = 0; j < 4; j++) {  // j enumerates the 4 float lanes inside one FragC
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];  // add reduced partial into replica-0
            }
          }
        }
        __syncthreads();  // keep shared reuse safe across m_block
      }
    }
  };
```

### Global Reduce

首先每个 CTA 只有一部分的 warp 需要参与这个 reduce 的过程，因为在上一步已经将归属于同一个 n 的不同 k 的 warp 进行了 reduce。

在我们的例子中，这里只需要 64 个 thread，也就是两个 warp。

```c++
constexpr int active_threads = 32 * thread_n_blocks / 4 = 32 * 8 / 4 = 64;
```

然后计算一些全局拷贝和 smem 的参数：

![](attachment/97bc4498c1254adb755c8cb5d06aafd0.png)

这里的拷贝到 gmem 的逻辑如下图所示。可以看到对于我们的两个 warp，首先拷贝 M 0-8，N 0-8。然后下一步拷贝 M 0-8，N 8-15。然后切换到下一行，以此类推。

![](attachment/56d614de4aa367f72fb63c11640f7c51.png)

但是需要注意的是对于 first 和 last 以及中间的 CTA，我们需要有不同的处理：

- Case 1: 第一个 CTA (first=true, last=false)

```cpp
if (!first) { … }  // 跳过加载

for (int i = 0; i < thread_m_blocks * 4; i++) {
    if (!first) { … }  // 跳过累加
    
    if (!last) {  // 执行
        // 将 frag_c 转换为 fp16 写入全局 C
        int4 c;
        for (int j = 0; j < 8; j++) {
            c[j] = fp32_to_fp16(frag_c[…]);
        }
        C[…] = c;
    }
}
```

```
操作: 只写入部分和到 C
C = P0
```

---

- Case 2: 中间 CTA (first=false, last=false)

```cpp
// 1. 从全局 C 加载之前的部分和到 shared
for (int i = …) {
    cp_async4_pred(&sh[…], &C[…], pred);
}
cp_async_fence();
cp_async_wait<0>();

// 2. 累加并写回
for (int i = …) {
    // 读取 shared 中的部分和，累加到 frag_c
    int4 c_red = sh[…];
    for (int j = 0; j < 8; j++) {
        frag_c[…] += fp16_to_fp32(c_red[j]);
    }
    
    // 写回新的部分和
    int4 c;
    for (int j = 0; j < 8; j++) {
        c[j] = fp32_to_fp16(frag_c[…]);
    }
    C[…] = c;
}
```

```
操作: 读取 → 累加 → 写回
C = C + P1 = P0 + P1
```

---

 - Case 3: 最后 CTA (first=false, last=true)

```cpp
// 1. 从全局 C 加载
for (int i = …) {
    cp_async4_pred(&sh[…], &C[…], pred);
}
cp_async_fence();
cp_async_wait<0>();

// 2. 只累加，不写回
for (int i = …) {
    // 累加
    int4 c_red = sh[…];
    for (int j = 0; j < 8; j++) {
        frag_c[…] += fp16_to_fp32(c_red[j]);
    }
    
    if (!last) { … }  // 跳过写回
}
```

```
操作: 读取 → 累加（结果留在 frag_c 中，由 write_result 写出）
frag_c = C + P2 = P0 + P1 + P2
```

## 快速反量化

* 先用 lop3 取出 int32 格式的 q 中的两个 4-bit 数据。分别对于低位和高位进行 lop3。这里 lop3 之后还对于 f 加了 1024，参考如下的观察，相当于转成了 fp16 的数据。但是额外增加了 1024，后续需要减去。
    * 【观察一】 对于任意 FP16 数 X，其中 1024 ≤ X <2048, 1024 将准确地表示在指数位，而 int(X − 1024) 部分将直接存储在尾数中（二进制 level 原封不动地存储）。例如，1027 的 FP16 表示（表示为 0x6403) 将整数 3 直接存储在其表示的尾数位。个人理解：0x6403 中的尾部部分的二进制表示为 0b0000000011(0x03)，这和整数 3 的 int 二进制表示完全一致。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/53f35fa3-4a51-4508-ba2b-04fab6f8727c.jpg)

    * 【观察二】对于任何整数 0 ≤ Y < 1024，我们可以构造 Y + 1024 的 FP16 表示，将指数设置为 1024 并存储 Y在 FP16 尾数中。这很容易通过 0x6400 | Y 实现 ，因为 0x6400 是FP16 中 1024 的十六进制表示。
        
    

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/9f504e53-ffeb-4063-8610-763f97c3bee8.jpg)

    这个可以在[https://evanw.github.io/float-toy/](https://evanw.github.io/float-toy/) 里面玩一下。

    

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/3a0eeecd-8ef4-4ea5-9740-2f6b0f8c6395.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/ad18c223-f182-44b5-89ce-d6824cc228de.png)

![image.png](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/e5f40e1a-72ac-465b-968c-1600f8678490.png)

* 然后我们把之前额外增加的 1024 减掉，这里减掉了 1032，额外的 8 是因为 uint4 本身需要减去 8。高位的处理相对特殊一点，因为需要额外除 16（MUL），对应的 ADD 数值也需要修改。

```c++
// Efficiently dequantize an int32 value into a full B-fragment of 4 fp16 values.
// We mostly follow the strategy in the link below, with some small changes:
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/cutlass_extensions/include/cutlass_extensions/interleaved_numeric_conversion.h
__device__ inline FragB dequant(int q) {
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  // Guarantee that the `(a & b) | c` operations are LOP3s.
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  // We want signed int4 outputs, hence we fuse the `-8` symmetric zero point directly into `SUB` and `ADD`.
  const int SUB = 0x64086408; //0x6408 = -1024 - 8 = -1032
  const int MUL = 0x2c002c00; //two 0.0625, means divide by 16
  const int ADD = 0xd480d480; //0xd480 = -72 = -1024/16 - 8 = -64 - 8 = -72
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  frag_b[1] = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL), *reinterpret_cast<const half2*>(&ADD)
  );
  return frag_b;
}

```

这里的 lop3 做了什么：

举个例子，对于 `int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);`

其实就是 `(q & LO) | EX;` 具体为什么可以参考 [《Marlin》](https://alidocs.dingtalk.com/i/nodes/nYMoO1rWxaLa1EMxtdvgXZqxV47Z3je9?utm_scene=team_space&iframeQuery=anchorId%3Duu_mhk9l7et4vj1gbex2ar)

假设原始的 q 是

```python
q = 0x00C300A5

  高16位        低16位
  00C3          00A5
  ││            ││
  │└─ n2=3      │└─ n0=5
  └── n3=12     └── n1=10
```

lo 取出来的是 n2 和 n0，并且加上了 1024，_// lo = { 1024 + n0, 1024 + n2 }//    = { 1029.0,    1027.0 }_

按照前面的逻辑，我们减去 1024，就完成了 int4 转 float16 的过程。额外减 8 从 uint4 变成 int4。

后面的代码就是分别对高位和低位进行减 1024 的操作。低位比较简单：

```python
  FragB frag_b;
  frag_b[0] = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
```

高位需要先除 16，然后减去 1024/16=64。额外减 8 从 uint4 变成 int4。

### PRMT

这一部分可以先忽略，W4A16 的 kernel 中没有用到。

参考 NV PTX ISA 8.1 文档，章节 9.7.8.7 Data Movement and Conversion Instructions: prmt 的内容。

* prmt：permute bytes from register pair

```nasm
prmt.b32{.mode} d, a, b, c; 
.mode = { .f4e, .b4e, .rc8, .ecl, .ecr, .rc16 }
```

PRMT 指令，会从两个 32 位寄存器 a, b,中选取四个任意字节，重新组成 32 位值，并保存在目标寄存器中。在通用形式（未指定模式）中，最终选取的 4 个字节，由四个 4 bit 的选择器组成。PRMT 指令会将两个源寄存器 a,b 中的字节编号为 0 到 7，具体形式为：

```nasm
{b,a} = {{b7,b6,b5,b4}，{b3，b2，b1，b0}}
```

对于目标寄存器中的每个字节，定义了一个 4 位选择器。选择值的 3 个 低位 lsb 指定应将 8 个源字节中的哪一个移至目标中位置。 msb 定义是否应直接复制原始字节值，或者是否应复制符号（即，是否进行符号扩展）；msb=0 表示直接复制原始的 bit 值，msb=1，则表示进行符号扩展。为简单起见，这里只关注 PRMT 指令的通用形式。（事实上，这个指令还有 f2e、b4e、rc8 等特殊模型）

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/843fa7fa-d1ec-4b5f-b9b1-f293fa78898c.jpg)

对于这条指令：

```nasm
prmt.b32{.mode} d, a, b, c; 
```

d 是目标操作数，a,b 分别是两个 32bit 的源操作数，c 是选择器。需要注意的是，c 中只有低 16 位是有用的（即使输入的是 32bit 寄存器），因为，d.b\_i 中只有四个待确定的字节，每个字节需要 4 bit 作为选择器，从 {b,a} = {{b7,b6,b5,b4}，{b3，b2，b1，b0}} 中选择对应的字节作为自己的值。之所以 4bit，是因为 3bit 就可以索引 0~7，覆盖了 8 个字节，还需要 1bit 表示是否进行符号扩展。举个例子：

```nasm
c[3:0] -> 0001 -> msb 0 lsb 001 -> 不进行符号扩展，并选择 index 1的字节，即 b1 -> d.b0 = b1
```

### LOP3

本节内容，参考 NV PTX ISA 8.1 文档 9.7.7.6 Logic and Shift Instructions: lop3 小节。

* lop3: Arbitrary logical operation on 3 inputs.

```nasm
lop3.b32 d, a, b, c, immLut; 
```

LOP3 指令对 3 个输入 a,b,c（都是 32 位寄存器）执行任意的逻辑操作，比如 (a & b) | c；并把逻辑运算后的结果保存在目的寄存器 d 中（也是一个 32 位寄存器）；操作数 immLut，指定了对 a,b,c 需要执行的操作。按照 NV PTX ISA 8.1 文档中的说明，immLut 和一个 look-up table 进行对应，immLut 的可选值范围为 0~255，每一个值，映射到一个特定的 F(a,b,c)，比如 immLut 为 0x80 时，LOP3 对 a,b,c 执行：d=(a & b & c)。

![image](https://alidocs.oss-cn-zhangjiakou.aliyuncs.com/res/WgZOZA8Mz8ZdeqLX/img/946f843b-a1bb-40c2-baa1-78d97e3a8fa1.jpg)

那么，对于某个操作，比如 F(a,b,c)=(a & b & c)，我们到底要怎么样指定这个 immLut 值呢？对于逻辑运算 F(a, b, c)，可以通过应用相同的方法来计算 immLut 的值，对三个预定义常数值（ta,tb,tc）的运算如下：

```nasm
# ta,tb,tc都是预定义的值，每个都是8bits
ta = 0xF0;  
tb = 0xCC;
tc = 0xAA;
immLut = F(ta, tb, tc);
```

举一些例子：

```nasm
If F = (a & b & c);
immLut = 0xF0 & 0xCC & 0xAA = 0x80
If F = (a | b | c);
immLut = 0xF0 | 0xCC | 0xAA = 0xFE
If F = (a & b & ~c);
immLut = 0xF0 & 0xCC & (~0xAA) = 0x40
If F = ((a & b | c) ^ a);
immLut = (0xF0 & 0xCC | 0xAA) ^ 0xF0 = 0x1A
```

比如，当你想要 LOP3 执行 (a & b & c) 操作时，这时 immLut 的值为 0xF0 & 0xCC & 0xAA = 0x80，也就是，对 ta,tb,tc 执行和 a,b,c 一样的逻辑操作后得到的值，即为 immLut；然后：

```text
lop3.b32 d, a, b, c, 0x80; // 得到的d=(a & b & c)
```

最终就是查表。

```python
 索引 | a | b | c
-----|---|---|---
  0  | 0 | 0 | 0
  1  | 0 | 0 | 1
  2  | 0 | 1 | 0
  3  | 0 | 1 | 1
  4  | 1 | 0 | 0
  5  | 1 | 0 | 1
  6  | 1 | 1 | 0
  7  | 1 | 1 | 1
```

构成了一个 8-bit 的查找表。然后 ta，tb，tc 这三个 int32 的数的每一个 bit 都去查找表内对应的结果。总共查找 32 次。得到最终的输出。

例如，对于我们的 case：

```python
 索引 | a | b | c | o |
-----|---|---|---|---|
  0  | 0 | 0 | 0 | 0 |
  1  | 0 | 0 | 1 | 1 |
  2  | 0 | 1 | 0 | 0 |
  3  | 0 | 1 | 1 | 1 |
  4  | 1 | 0 | 0 | 0 |
  5  | 1 | 0 | 1 | 1 |
  6  | 1 | 1 | 0 | 1 |
  7  | 1 | 1 | 1 | 1 |
```

对应的 a 列从高位到低位就是 0xF0，0xCC, 0xAA。输出是 (0xF0 & 0xCC) | 0xAA=0xEA。

## 矩阵乘法

```c++
  // Execute the actual tensor core matmul of a sub-tile. 
  auto matmul = [&] (int k) {
    // We have the m dimension as the inner loop in order to encourage overlapping dequantization and matmul operations.
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      int b_quant = frag_b_quant[k % 2][j];
      int b_quant_shift = b_quant >> 8;
      FragB frag_b0 = dequant(b_quant);
      // If there are no groups, we can just scale the final output once and can avoid doing so for each weight.
      if (group_blocks != -1)
        scale(frag_b0, frag_s[k % 2][j], 0);
      FragB frag_b1 = dequant(b_quant_shift);
      if (group_blocks != -1)
        scale(frag_b1, frag_s[k % 2][j], 1);
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
        mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
      }
    }
  };
```

## C 矩阵写回

```c++
// Write final result for this CTA's (M,N) tile.
  // Pack fp16 results into shared (as half2) and stream out as int4 (16B) stores.
  auto write_result = [&] () {  // final writeback of this CTA's tile to global C
    // Shared-memory writeback staging:
    // - First, a subset of threads packs f32 accumulators into shared as half2 (fp16), with padding to avoid bank conflicts.
    // - Then, all threads stream shared -> global as int4 (each int4 = 8 fp16 values = 16 bytes).
    int c_gl_stride = prob_n / 8;  // global C row stride in int4 (N/8)
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;  // shared row stride in int4 (tile width + 1 padding)
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));  // per-iteration global int4 stride for this thread
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));  // per-iteration shared int4 stride for this thread

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // base global int4 (row,col)
    c_gl_wr += (2 * thread_n_blocks) * slice_col;  // offset to this N tile (tile width = 2*thread_n_blocks int4)
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;  // base shared half2 index (warp-friendly layout)
    c_sh_wr += 32 * (threadIdx.x / 32);  // separate warps in half2 indexing (each warp writes its own columns)
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));  // base shared int4 index (contiguous stream order)

    int c_gl_wr_end = c_gl_stride * prob_m;  // end of valid global C region (in int4) for prob_m rows

    auto write = [&] (int idx, float c0, float c1, FragS& s) {  // pack 2 f32 -> half2, optionally scale, store to shared
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));  // f32 -> fp16 pair (2 columns)
      if (group_blocks == -1)  // per-column mode: apply scale at the end (grouped mode already scaled during matmul)
        res = __hmul2(res, s[0]);  // scale both halves
      ((half2*) sh)[idx] = res;  // shared write as half2 (4 bytes)
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {  // only the warps that own columns pack frag_c -> shared
      #pragma unroll  // unroll m_block packing
      for (int i = 0; i < thread_m_blocks; i++) {  // i selects the m_block (16 rows each)
        #pragma unroll  // unroll 16-col chunks per warp (4 * 16 = 64 cols per warp)
        for (int j = 0; j < 4; j++) {  // j selects a 16-column chunk within this warp's half-tile
          int wr = c_sh_wr + 8 * j;  // half2 index: 8 half2 = 16 columns per chunk
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 0.,  cols 0.
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 8., cols 0.
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 0.,  cols 8.
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 8., cols 8.
        }
        c_sh_wr += 16 * (4 * c_sh_stride);  // advance shared half2 base by 16 rows for the next m_block
      }
    }
    __syncthreads();  // ensure shared packing is complete before streaming out

    #pragma unroll  // unroll per-thread streaming loop
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {  // cover all rows in the CTA tile
      if (c_gl_wr < c_gl_wr_end) {  // guard the prob_m tail
        C[c_gl_wr] = sh[c_sh_rd];  // global store of one int4 (8 fp16) from shared
        c_gl_wr += c_gl_wr_delta;  // advance global int4 index for the next iteration
        c_sh_rd += c_sh_rd_delta;  // advance shared int4 index for the next iteration
      }
    }
  };
```

这里分成了两个阶段：

第一个阶段：frag_c to shm

```c++
if (threadIdx.x / 32 < thread_n_blocks / 4) {  // only the warps that own columns pack frag_c -> shared
      #pragma unroll  // unroll m_block packing
      for (int i = 0; i < thread_m_blocks; i++) {  // i selects the m_block (16 rows each)
        #pragma unroll  // unroll 16-col chunks per warp (4 * 16 = 64 cols per warp)
        for (int j = 0; j < 4; j++) {  // j selects a 16-column chunk within this warp's half-tile
          int wr = c_sh_wr + 8 * j;  // half2 index: 8 half2 = 16 columns per chunk
          write(wr + (4 * c_sh_stride) * 0 + 0, frag_c[i][j][0][0], frag_c[i][j][0][1], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 0.,  cols 0.
          write(wr + (4 * c_sh_stride) * 8 + 0, frag_c[i][j][0][2], frag_c[i][j][0][3], frag_s[j / 2][2 * (j % 2) + 0]);  // rows 8., cols 0.
          write(wr + (4 * c_sh_stride) * 0 + 4, frag_c[i][j][1][0], frag_c[i][j][1][1], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 0.,  cols 8.
          write(wr + (4 * c_sh_stride) * 8 + 4, frag_c[i][j][1][2], frag_c[i][j][1][3], frag_s[j / 2][2 * (j % 2) + 1]);  // rows 8., cols 8.
        }
        c_sh_wr += 16 * (4 * c_sh_stride);  // advance shared half2 base by 16 rows for the next m_block
      }
    }
```

这里的两层外围的循环，i 是遍历所有的 m block，j 是遍历 4 个 n block。对于每个 block 的输出，都是 16x16。

解析一下 frag_c，这里的 i 和 j 分别是第几个 m block 和第几个 n block。第三个方框是 n 的 idx（因为每次计算 16x8x16，这里的 idx 是这个 8 的 idx）。最后一个是 4 个输出的 float 的 idx。

注意到最后的 c 矩阵的输出对应了第 k 行和第 k+8 行的输出。所以可以看到上面的坐标。这里的 4 * c_sh_stride 是因为 c_sh_stride 是以 int4 为单位，而最终 write 的内部用了 half2。

同样的，wr + (4 * c_sh_stride) * 8 + 4 这里的加 4，也是因为 half2 为单位的时候，+4 相当于 half 的 +8。

![](attachment/b0ec18bf52454c1463d604666fc902f1.png)

上图可以看到 t0-31 对于第一个 16x16 的输出的拷贝。后面的 t32-64 也是类似的，拷贝下一个 n 的 half。然后通过外围的遍历，拷贝完整的一个 thread_m_blocks * 4 的数据。这里的 4 是一个 warp 对应的 n 上的 thread_n_blocks 的数据范围。

第二阶段：Shared Memory → 全局 C

```c++
for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
    // ceildiv(64, 16) = 4 次迭代
    if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = sh[c_sh_rd];     // 写 1 int4 = 8 fp16
        c_gl_wr += c_gl_wr_delta;     // +4096 (跳 16 行)
        c_sh_rd += c_sh_rd_delta;     // +272 (跳 16 行)
    }
}
```

迭代过程:

- 迭代 i=0: 写入 row 0-15   (256 线程各写 1 int4)
- 迭代 i=1: 写入 row 16-31
- 迭代 i=2: 写入 row 32-47
- 迭代 i=3: 写入 row 48-63

总共: 4 × 16 = 64 行 = thread_m_blocks × 16

## Mainloop

### Start pipe

```c++
	  // Start pipeline:
	  // Prefetch (stages-1) tiles so the main loop can overlap:
	  //   ldmatrix/dequant/mma on stage p  with  cp.async filling stage (p+stages-1).
	  auto start_pipes = [&] () {  // prologue: fill pipeline and prime registers for k=0
	    #pragma unroll  // unroll stage prefetches
	    for (int i = 0; i < stages - 1; i++)  // prefetch (stages-1) K-tiles into shared
	      fetch_to_shared(i, i, i < slice_iters);  // stage=i, a_off=i, pred guards tail tiles
	    zero_accums();  // clear frag_c accumulators before accumulation
	    wait_for_stage();  // wait until stage 0 data is ready in shared
	    fetch_to_registers(0, 0);  // load A/B(/S) for k=0 from stage 0 into regs
	    a_gl_rd += a_gl_rd_delta_o * (stages - 1);  // A gmem read index (int4)
	  };
	  start_pipes();  // kick off the pipeline for this slice
```

首先取 stage-1 个 stage 的数据到 shm。然后初始化 frag c 为空。

然后等待至少一个 stage 的数据 ready。

然后取 stage0 的数据到 register。切换 a 矩阵的 gmem 到下一个 stage。

### Mainloop

```c++
	  while (slice_iters) {  // 只要slice_iters还有剩余就继续迭代
```

首先看到外围的大循环，只要 slice_iters 还有数值，就继续迭代。这里的 slice_iters 是在当前的 n 上还剩余多少个 tile。

然后 pipeline 遍历 stages。对于每个 stage，又分成了多个 k 上的 iterations。注意，这里的 ldmatrix 也进行了 double buffer 的 pingpong 操作。这里加载的是下一个 ldmatrix 对应的数据到 register。最后计算的是前一个 k 的 matmul。

当我们计算到最后一个 subtile，需要加载下一个 stage 的数据。注意，这里加载下一个 stage 的数据之后，需要把 stage++，另外还需要 wait_for_stage，因为前面加载的是 stage-1 的数据。

```
时间 →
────────────────────────────────────────────────────────────────────────────────→

共享内存 stage:
stage 0: [k-tile 0 已加载]────[计算中]────[释放]────────[k-tile 4 加载中]────
stage 1: [k-tile 1 已加载]──────────────[计算中]────[释放]────────────────────
stage 2: [k-tile 2 已加载]────────────────────────[计算中]────[释放]──────────
stage 3: [k-tile 3 加载中]────[就绪]────────────────────────[计算中]──────────

寄存器缓冲:
frag[0]:  [k=0]────[计算]────[k=0]────[计算]────[k=0]────[计算]────
frag[1]:  [空]────[k=1 加载]────[计算]────[k=1 加载]────[计算]────

计算:
          matmul(0) matmul(1) matmul(0) matmul(1) matmul(0) matmul(1)
          └─k-tile 0─┘        └─k-tile 1─┘        └─k-tile 2─┘
```

然后计算完成 matmul，slice_iters --，如果不为 0，继续计算下一个 stage。如果是 0，break。

```c++
	    for (int pipe = 0; pipe < stages;) {  // pipe is the logical K-tile index within this unrolled chunk
	      #pragma unroll  // unroll B-subtiles inside one stage
	      for (int k = 0; k < b_sh_wr_iters; k++) {  // k iterates over B sub-tiles within a stage
	        fetch_to_registers(k + 1, pipe % stages);  // prefetch next k-subtile into regs (ping-pong buffers)
	        if (k == b_sh_wr_iters - 2) {  // when we're about to use the last subtile, rotate the shared stage
	          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);  // cp.async next K-tile into the "oldest" stage
	          pipe++;  // advance logical pipe (stage consumer)
	          wait_for_stage();  // wait until the newly-fetched stage is ready for ldmatrix/loads
	        }
	        matmul(k);  // dequant/scale (if needed) + tensorcore MMA for subtile k
	      }
	      slice_iters--;  // one K-tile fully consumed
	      if (slice_iters == 0)  // slice complete
	        break;  // exit stage loop early
	    }
```

这里的问题：fetch_to_registers 会多取一次。

然后挪动 a_gl_rd 的 addr 到下一个 stages：

```c++
	    a_gl_rd += a_gl_rd_delta_o * stages;  // A gmem read index (int4)
```

如果 slice_iters 不等于 0，继续上面的循环。

如果 slice_iters 等于 0。进入下面的处理逻辑：

- 首先进行 block reduce：thread_block_reduce();
- 如果 slice_count > 1，也就是需要 global reduce，那么我们需要等到对应的 slice_idx。然后进行 reduce。注意这里的问题。由 last 进行 reduce。

 ```c++
		  if (slice_count > 1) {  // multiple CTAs contribute to the same N tile => global reduction needed
	        barrier_acquire(&locks[slice_col], slice_idx);  // wait until it's this CTA's turn to reduce
	        global_reduce(slice_idx == 0, last);  // first writes partial, middle reduces+writes, last reduces only
	        barrier_release(&locks[slice_col], last);  // signal next CTA (or reset on last)
	      }
 ```

- 如果是 last，还需要 write data out:

```c++
		  if (last)  // only last CTA writes the final, fully reduced output tile
	        write_result();  // pack to shared and store to global C
```

下一步，进入下一个 N 的当前 CTA 的处理：

```c++
	      slice_row = 0;  // tile row in K grid (idx%k_tiles)
	      slice_col_par++;  // advance to the next N-tile (including parallel slices)
	      slice_col++;  // advance N-tile within this parallel slice
	      init_slice();  // compute next slice_iters/slice_count/slice_idx
```

对应的 ptr 进行重置，之后再次循环：

```c++
	      if (slice_iters) {  // if there is another slice to process, rewind pointers and restart pipeline
	        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)

	        // Move B pointers to the next N tile and rewind K back to slice_row=0 for the new slice.
	        #pragma unroll
	        for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;  // move to next N tile and rewind K by k_tiles
	        if (slice_col == 0) {  // wrapped around to the next parallel slice
	          #pragma unroll
	          for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	            B_ptr[i] -= b_gl_stride;  // move B back by one row stride to align slice_col wrap
	        }

	        // Reset scale pointer for the new N tile (grouped mode will advance it inside fetch_to_shared()).
	        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
	        start_pipes();  // restart prologue for the next slice
	      }
```

完整的代码：

```c++
	  // Main loop
	  while (slice_iters) {  // 只要slice_iters还有剩余就继续迭代
	    #pragma unroll  // unroll over stages (small fixed count)
	    for (int pipe = 0; pipe < stages;) {  // pipe is the logical K-tile index within this unrolled chunk
	      #pragma unroll  // unroll B-subtiles inside one stage
	      for (int k = 0; k < b_sh_wr_iters; k++) {  // k iterates over B sub-tiles within a stage
	        fetch_to_registers(k + 1, pipe % stages);  // prefetch next k-subtile into regs (ping-pong buffers)
	        if (k == b_sh_wr_iters - 2) {  // when we're about to use the last subtile, rotate the shared stage
	          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);  // cp.async next K-tile into the "oldest" stage
	          pipe++;  // advance logical pipe (stage consumer)
	          wait_for_stage();  // wait until the newly-fetched stage is ready for ldmatrix/loads
	        }
	        matmul(k);  // dequant/scale (if needed) + tensorcore MMA for subtile k
	      }
	      slice_iters--;  // one K-tile fully consumed
	      if (slice_iters == 0)  // slice complete
	        break;  // exit stage loop early
	    }
	    a_gl_rd += a_gl_rd_delta_o * stages;  // A gmem read index (int4)

	    if (slice_iters == 0) {  // finished all K-tiles for this N tile => reduce + writeback
	      cp_async_wait<0>();  // ensure no pending cp.async reads are outstanding
	      bool last = slice_idx == slice_count - 1;  // last CTA for this N tile (does final writeback)

	      if (group_blocks == -1 && last) {  // per-column mode: fetch scales once at the end (applied in write_result)
	        if (s_sh_wr_pred)  // only threads within the scale tile issue the load
	          cp_async4_stream(&sh_s[s_sh_wr], &s[s_gl_rd]);  // stage per-column scales into sh_s
	        cp_async_fence();  // commit scale cp.async
	      }

	      thread_block_reduce();  // CTA-local reduce (collapse red_idx replicas into red_idx==0)

	      if (group_blocks == -1 && last) {  // per-column mode: bring scales into regs for write_result()
	        cp_async_wait<0>();  // wait for scale cp.async
	        __syncthreads();  // make sh_s visible to all threads
	        if (threadIdx.x / 32 < thread_n_blocks / 4) {  // only active writer warps need scales
	          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];  // load scales for cols 0..?
	          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];  // load scales for cols +8..?
	        }
	      }

	      if (slice_count > 1) {  // multiple CTAs contribute to the same N tile => global reduction needed
	        barrier_acquire(&locks[slice_col], slice_idx);  // wait until it's this CTA's turn to reduce
	        global_reduce(slice_idx == 0, last);  // first writes partial, middle reduces+writes, last reduces only
	        barrier_release(&locks[slice_col], last);  // signal next CTA (or reset on last)
	      }

	      if (last)  // only last CTA writes the final, fully reduced output tile
	        write_result();  // pack to shared and store to global C

	      slice_row = 0;  // tile row in K grid (idx%k_tiles)
	      slice_col_par++;  // advance to the next N-tile (including parallel slices)
	      slice_col++;  // advance N-tile within this parallel slice
	      init_slice();  // compute next slice_iters/slice_count/slice_idx

	      if (slice_iters) {  // if there is another slice to process, rewind pointers and restart pipeline
	        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);  // A gmem read index (int4)

	        // Move B pointers to the next N tile and rewind K back to slice_row=0 for the new slice.
	        #pragma unroll
	        for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	          B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;  // move to next N tile and rewind K by k_tiles
	        if (slice_col == 0) {  // wrapped around to the next parallel slice
	          #pragma unroll
	          for (int i = 0; i < b_sh_wr_iters; i++)  // adjust each per-iter B pointer
	            B_ptr[i] -= b_gl_stride;  // move B back by one row stride to align slice_col wrap
	        }

	        // Reset scale pointer for the new N tile (grouped mode will advance it inside fetch_to_shared()).
	        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;  // scale gmem read index (int4)
	        start_pipes();  // restart prologue for the next slice
	      }
	    }
	  }
```

## 调度解析

我们以 20 个 SM，处理 2048 * 2048 的 n 和 k 为例：

![](attachment/b644083d75755f519bf93d5cb00e24b2.png)

实际 reduce 的时候不会有同步的问题：

![](attachment/3d0a29ccb5798a1c19eb8d688db33f8a.png)
