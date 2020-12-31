#include<iostream>
#include "utils.h"

static void __global__ mathKernel1(float *c){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if (tid % 2 == 0){
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

static void __global__ mathKernel2(float *c){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if ((tid / warpSize) % 2 == 0){
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

static void __global__ mathKernel3(float *c){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float a, b;
    a = b = 0.0f;
    
    bool ipred = (tid % 2 == 0);
    if (ipred){
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

static void __global__ warmingup(float *c){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    float a, b;
    a = b = 0.0f;
    
    if ((tid / warpSize) % 2 == 0){
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[tid] = a + b;
}

//nvprof --metrics branch_efficiency ./sample07_simpledivergence
//======== Warning: Skipping profiling on device 0 since profiling is not supported on devices with compute capability 7.5 or higher. Profiling features on these devices are supported in the next generation GPU profiling tool NVIDIA Nsight Compute. Refer https://developer.nvidia.com/nsight-compute for more details.
int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // set up data size
    int size = 64; 
    int blocksize = 64; 
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size = atoi(argv[2]);
    printf("Data size %d ", size);

    // set up execution configuration
    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate gpu memory
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // run a warmup kernel(kernel2) to remove overhead
    double iStart, iElaps;
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    warmingup<<< grid, block >>>(d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    printf("warmingup    <<<%4d,%4d>>> elapsed %f millsec\n", grid.x, block.x, iElaps);

    // run kernel1
    iStart = cpuTimerInMS();
    mathKernel1<<< grid, block >>>(d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    printf("mathKernel1  <<<%4d,%4d>>> elapsed %f millsec\n", grid.x, block.x, iElaps);

    // run kernel2
    iStart = cpuTimerInMS();
    mathKernel2<<< grid, block >>>(d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    printf("mathKernel2  <<<%4d,%4d>>> elapsed %f millsec\n", grid.x, block.x, iElaps);

    // run kernel3
    iStart = cpuTimerInMS();
    mathKernel3<<< grid, block >>>(d_C);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    printf("mathKernel3  <<<%4d,%4d>>> elapsed %f millsec\n", grid.x, block.x, iElaps);

    CUDA_CHECK(cudaFree(d_C));

    // reset device
    cudaDeviceReset();
    return 0;
}
