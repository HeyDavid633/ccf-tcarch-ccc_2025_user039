#include "main.h"

// Block size (tunable, 256 is safe for most GPUs)
constexpr int BLOCK_SIZE = 256;

// Kernel 1: Find block-level max
__global__ void find_block_max(const float* input, float* block_max, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < N) ? input[i] : -FLT_MAX;
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write block max to global memory
    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Compute exp(x_i - max_val) and reduce sum per block
__global__ void compute_exp_and_block_sum(const float* input, float* temp, float* block_sum, int N, float max_val) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (i < N) ? expf(input[i] - max_val) : 0.0f;
    temp[i] = val; // save for later normalization

    sdata[tid] = val;
    __syncthreads();

    // Reduce sum in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Normalize: output[i] = temp[i] / total_sum
__global__ void normalize_kernel(float* temp, float* output, int N, float total_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = temp[i] / total_sum;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    // Device pointers
    float *d_input, *d_output, *d_temp;
    float *d_block_max, *d_block_sum;

    // Allocate device memory
    hipMalloc(&d_input, N * sizeof(float));
    hipMalloc(&d_output, N * sizeof(float));
    hipMalloc(&d_temp, N * sizeof(float));

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipMalloc(&d_block_max, num_blocks * sizeof(float));
    hipMalloc(&d_block_sum, num_blocks * sizeof(float));

    // Copy input to device
    hipMemcpy(d_input, input, N * sizeof(float), hipMemcpyHostToDevice);

    // Step 1: Find global max
    hipLaunchKernelGGL(find_block_max, dim3(num_blocks), dim3(BLOCK_SIZE), BLOCK_SIZE * sizeof(float), 0, d_input, d_block_max, N);
    hipDeviceSynchronize();

    // Copy block_max to host and find global max
    std::vector<float> h_block_max(num_blocks);
    hipMemcpy(h_block_max.data(), d_block_max, num_blocks * sizeof(float), hipMemcpyDeviceToHost);

    float global_max = -FLT_MAX;
    for (float val : h_block_max) {
        if (val > global_max) global_max = val;
    }

    // Step 2: Compute exp(x_i - global_max) and per-block sum
    hipLaunchKernelGGL(compute_exp_and_block_sum, dim3(num_blocks), dim3(BLOCK_SIZE), BLOCK_SIZE * sizeof(float), 0,
                       d_input, d_temp, d_block_sum, N, global_max);
    hipDeviceSynchronize();

    // Copy block_sum to host and compute total sum
    std::vector<float> h_block_sum(num_blocks);
    hipMemcpy(h_block_sum.data(), d_block_sum, num_blocks * sizeof(float), hipMemcpyDeviceToHost);

    float total_sum = 0.0f;
    for (float val : h_block_sum) {
        total_sum += val;
    }

    // Avoid division by zero (per README requirement)
    if (total_sum < 1e-12f) total_sum = 1.0f;

    // Step 3: Normalize
    hipLaunchKernelGGL(normalize_kernel, dim3(num_blocks), dim3(BLOCK_SIZE), 0, 0, d_temp, d_output, N, total_sum);
    hipDeviceSynchronize();

    // Copy result back to host
    hipMemcpy(output, d_output, N * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_input);
    hipFree(d_output);
    hipFree(d_temp);
    hipFree(d_block_max);
    hipFree(d_block_sum);
}