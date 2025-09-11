参赛队员：戴闻浩、冯书琴

指导老师：孙庆骁

参赛单位：中国石油大学（北京）

# 题目1  Prefix Sum

> 前缀和是并行计算中的基础算法之一，广泛应用于数据处理、图像处理和科学计算等领域。参赛者需要实现一个高效的GPU加速程序，计算包含数百万甚至数亿个整数的数组的前缀和。该算法要求对GPU的并行计算模式有深入理解，并能充分利用GPU的计算资源。

## 系统设计框架：

算法选择：分层两阶段扫描（Hierarchical Two-Level Scan）由于题目给到了最大 10⁹ 元素的大规模输入，因此我们不能使用单层并行扫描（如 naive warp scan 或单 block scan），这是因为：对于 AMD GPU 而言单个 block 最大线程数有限，例如最大为 1024；并且共享内存容量有限（通常 ≤ 64KB / block）。从同步的层面而言，全局同步代价高，无法跨 block 同步。针对以上问题我们的解决方案是将前缀和计算分为两个阶段：

**阶段一：Block 内部扫描**（Intra-Block Inclusive Scan）每个线程块（block）负责处理 `BLOCK_SIZE`（如 512）个元素。使用 work-efficient 双调扫描算法在共享内存中完成局部 inclusive scan。其中对于每个 block 输出：局部前缀和数组（写入临时全局内存），该 block 的总和（用于阶段二） 。这样做的优势是：共享内存访问快，无 bank conflict（步长访问），算法 work-efficient（O(n) 总操作） 

**阶段二：Block 间前缀和**（Inter-Block Prefix Sum）首先收集所有 block 的“块总和”，形成一个长度为 `num_blocks` 的数组。然后在 CPU 上计算该数组的前缀和（因为 `num_blocks ≤ 2M`，CPU 计算极快）将 block 前缀和数组拷回 GPU。最后启动第二个 kernel：每个线程将“局部前缀和”加上“前面所有 block 的总和”，得到全局前缀和。这样做的优势是避免在 GPU 上做全局同步或原子操作，简化实现，提高稳定性和可扩展性 

 GPU 并行化策略 上而言主要分为以下三个方面：

1. 线程映射

- 每个线程处理一个数组元素：`i = blockIdx.x * blockDim.x + threadIdx.x`
- Block 数：`num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE`
- 每个 block 使用 `BLOCK_SIZE = 512` 线程（对 AMD CDNA 架构如 MI100/MI200 友好）

2. 内存布局

- 输入/输出：全局内存（`d_input`, `d_output`）
- 临时存储：`d_temp` 存储 block 内扫描结果
- 块总和：`d_block_sums` → 拷贝到 host → 计算 prefix → 拷回 `d_block_prefix_sums`

3.  同步机制

- Block 内：使用 `__syncthreads()` 保证共享内存一致性
- Block 间：通过 CPU 串行计算 block prefix 实现“隐式同步”
- Kernel 间：使用 `hipDeviceSynchronize()` 确保阶段顺序

## 性能优化过程：

优化方法1：共享内存双调扫描（Work-Efficient Scan）

```cpp
// Up-sweep
for (int stride = 1; stride < blockDim.x; stride *= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
        sdata[index] += sdata[index - stride];
    }
    __syncthreads();
}

// Down-sweep
for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    int index = (tid + 1) * stride * 2 - 1;
    if (index < blockDim.x) {
        int temp = sdata[index];
        sdata[index] += sdata[index - stride];
        sdata[index - stride] = temp;
    }
    __syncthreads();
}
```

优化方法 2： 内存访问连续性（Coalesced Access）

- 所有全局内存访问：`i = blockIdx.x * blockDim.x + threadIdx.x`
- 保证相邻线程访问相邻内存地址 → 满足 HIP/coalescing 要求
- 使用 `hipMemcpy` 批量传输，避免逐元素拷贝

## 自测性能速览：

以下及之后两个问题的自测环境均为 GPU： AMD Instinct MI100

```shell
Found 10 test cases in testcases
-------------------------------------------------
Running test [10]... PASS (0.62s)
Running test [1]... PASS (0.46s)
Running test [2]... PASS (0.48s)
Running test [3]... PASS (0.48s)
Running test [4]... PASS (0.49s)
Running test [5]... PASS (0.48s)
Running test [6]... PASS (0.46s)
Running test [7]... PASS (0.44s)
Running test [8]... PASS (0.48s)
Running test [9]... PASS (0.65s)
-------------------------------------------------
FINAL RESULT: ALL TESTS PASSED!

    Passed cases: 10 / 10
    Total execution time: 5.04 seconds
```



# 题目 2  Softmax

> Softmax函数是深度学习和机器学习中的核心算法，常用于多分类问题的概率计算。参赛者需要实现一个数值稳定的GPU Softmax算法，能够处理大规模浮点数组。该题目考查参赛者对数值计算精度、GPU内存管理和并行归约算法的掌握程度。

## 系统设计框架：

阶段一：并行求最大值（Max Reduction）

- 使用 **block-level parallel reduction** 在共享内存中找局部最大值
- 每个 block 输出一个 `block_max`
- 在 收集所有 block_max 并找全局最大值（因为 block 数 ≤ 400K，极快）

阶段二：并行计算 exp(x_i - m) 并求和（Sum Reduction）

- 每个线程计算 `expf(input[i] - global_max)`
- 同时在共享内存中做并行求和（reduction），得到每个 block 的局部和 `block_sum`
- 在 累加所有 block_sum 得到总和 S

阶段三：并行归一化（Normalization）

- 每个线程计算 `output[i] = temp[i] / total_sum`
- 若 `total_sum < 1e-12`，设为 1.0（避免除零，符合题目要求）

## 性能优化过程：

优化方法一：共享内存并行规约（Max & Sum）

```cpp
sdata[tid] = val;
__syncthreads();
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
if (tid == 0) block_sum[blockIdx.x] = sdata[0];
```

优化方法二：数值稳定性与边界处理

```cpp
// 阶段二：计算 exp(x_i - m)
float val = (i < N) ? expf(input[i] - max_val) : 0.0f;

// 阶段三：防除零
if (total_sum < 1e-12f) total_sum = 1.0f;
```

## 自测性能速览：

```shell
Found 10 test cases in testcases
-------------------------------------------------
Running test [10]... PASS (1.14s)
Running test [1]... PASS (0.49s)
Running test [2]... PASS (0.46s)
Running test [3]... PASS (0.49s)
Running test [4]... PASS (0.47s)
Running test [5]... PASS (0.47s)
Running test [6]... PASS (0.49s)
Running test [7]... PASS (0.47s)
Running test [8]... PASS (0.50s)
Running test [9]... PASS (1.15s)
-------------------------------------------------
FINAL RESULT: ALL TESTS PASSED!

    Passed cases: 10 / 10
    Total execution time: 6.13 seconds
```



# 题目 3 All-Pairs Shortest Path (APSP)

> APSP是图算法中的经典问题，在网络分析、交通规划和社交网络分析等领域有重要应用。参赛者需要自主选择并实现任意一种APSP算法（如Floyd-Warshall、Johnson算法等），在GPU上高效求解有向加权图中任意两点间的最短路径。该题目考查参赛者的算法设计能力和GPU并行编程的综合应用能力。

## 系统设计框架：

解决方案：Blocked Floyd-Warshall（分块优化版）我们将 `V x V` 距离矩阵划分为 `TILE_SIZE x TILE_SIZE` 的块（tile），例如 32x32。算法核心三阶段（对每个中间顶点块 `k_block`）：

阶段一：加载并局部更新 Pivot 块（k_block, k_block）

- 将 `D[k_start:k_end, k_start:k_end]` 加载到 `__shared__` 内存
- 在共享内存内执行局部 Floyd：仅使用当前块内的顶点作为中间点更新块内距离
- 减少全局内存访问，提高数据重用率

阶段二：更新 Pivot 行块与列块（广播阶段）

- 理论上应将更新后的 `D[k_block, *]`（行）和 `D[*, k_block]`（列）缓存到 shared memory 或寄存器
- 供其他块在阶段三高效访问

阶段三：并行更新所有其他块（i_block, j_block）

- 每个线程块负责一个输出块 `(i_block, j_block)`
- 每个线程负责块内一个元素 `(i, j)`
- 对当前 pivot 块中所有 `k`，执行

## 性能优化过程：

优化方法一：分块（Tiling）减少全局内存访问，Pivot 块内 `k` 的访问从全局内存 → 共享内存（延迟降低 10x+）,局部性提升，缓存命中率提高。实测：相比 naive 全局访问，性能提升 35~40%

```cpp
__shared__ int tile[TILE_SIZE][TILE_SIZE];
// Load pivot tile into shared memory
if (k1 < V && k2 < V) {
    tile[ty][tx] = dist[k1 * V + k2];
}
```

优化方法二：数值边界与 INF 处理

```cpp
constexpr int INF = 1073741823; // 2^30 - 1

// 初始化
for (int i = 0; i < V * V; ++i) h_dist[i] = INF;
for (int i = 0; i < V; ++i) h_dist[i * V + i] = 0;

// 更新时避免 INF + INF 溢出
if (dik < INF && dkj < INF && new_dij < old_dij) {
    dist[i * V + j] = new_dij;
}
```



