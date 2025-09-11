#include "main.h"

constexpr int BLOCK_SIZE = 512;

__global__ void block_scan_kernel(const int* input, int* temp_output, int* block_sums, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    int val = (i < N) ? input[i] : 0;
    sdata[tid] = val;
    __syncthreads();

    // Up-sweep phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            sdata[index] += sdata[index - stride];
        }
        __syncthreads();
    }

    // Save last element for exclusive scan
    if (tid == 0) {
        sdata[blockDim.x - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int temp = sdata[index];
            sdata[index] += sdata[index - stride];
            sdata[index - stride] = temp;
        }
        __syncthreads();
    }

    // Convert exclusive scan to inclusive scan: add original input[i]
    if (i < N) {
        temp_output[i] = sdata[tid] + val; // inclusive = exclusive + original
    }

    // Last thread in block saves the block's total sum (last inclusive value in block)
    if (tid == blockDim.x - 1 && i < N) {
        block_sums[blockIdx.x] = temp_output[i]; // 直接取 inclusive 的最后一个值
    }
}

__global__ void add_block_prefix_kernel(int* temp_output, int* output, const int* block_prefix_sums, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int block_id = blockIdx.x;
    if (block_id > 0) {
        output[i] = temp_output[i] + block_prefix_sums[block_id - 1];
    } else {
        output[i] = temp_output[i];
    }
}

extern "C" void solve(const int* input, int* output, int N) {
    if (N <= 0) return;

    int *d_input, *d_output, *d_temp, *d_block_sums, *d_block_prefix_sums;

    hipMalloc(&d_input, N * sizeof(int));
    hipMalloc(&d_output, N * sizeof(int));
    hipMalloc(&d_temp, N * sizeof(int));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipMalloc(&d_block_sums, num_blocks * sizeof(int));
    hipMalloc(&d_block_prefix_sums, num_blocks * sizeof(int));

    hipMemcpy(d_input, input, N * sizeof(int), hipMemcpyHostToDevice);

    size_t shared_mem_size = BLOCK_SIZE * sizeof(int);
    hipLaunchKernelGGL(block_scan_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), shared_mem_size, 0,
                       d_input, d_temp, d_block_sums, N);
    hipDeviceSynchronize();

    // Copy block sums to host and compute prefix sums
    std::vector<int> h_block_sums(num_blocks);
    hipMemcpy(h_block_sums.data(), d_block_sums, num_blocks * sizeof(int), hipMemcpyDeviceToHost);

    std::vector<int> h_block_prefix_sums(num_blocks);
    if (num_blocks > 0) {
        h_block_prefix_sums[0] = h_block_sums[0];
        for (int i = 1; i < num_blocks; ++i) {
            h_block_prefix_sums[i] = h_block_prefix_sums[i - 1] + h_block_sums[i];
        }
    }

    hipMemcpy(d_block_prefix_sums, h_block_prefix_sums.data(), num_blocks * sizeof(int), hipMemcpyHostToDevice);

    hipLaunchKernelGGL(add_block_prefix_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0,
                       d_temp, d_output, d_block_prefix_sums, N);
    hipDeviceSynchronize();

    hipMemcpy(output, d_output, N * sizeof(int), hipMemcpyDeviceToHost);

    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_temp);
    hipFree(d_block_sums);
    hipFree(d_block_prefix_sums);
}