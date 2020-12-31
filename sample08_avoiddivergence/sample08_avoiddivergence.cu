#include<iostream>
#include "utils.h"

// not only for add but any operation that satisfies the commutative law and associative law
int recursiveReduce(int *data, int const size){
    // terminate check
    if (size == 1) return data[0];
    // renew the stride
    int const stride = size /2;

    // in-place reducetion
    for (int i = 0; i< stride; i++){
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// using odd thread 
static void __global__ reduceNeighbored(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    if (idx >= n) return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *=2) {
        if((tid % (2 * stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// using first half thread with less divergence
static void __global__ reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    if (idx >= n) return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *=2) {
        int index = 2 * stride * tid;
        if( index < blockDim.x ){
            idata[index] += idata[index + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// interleaved pair implementation with less divergence
static void __global__ reduceInterleaved(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x;

    // boundary check
    if (idx >= n) return;
    
    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if( tid < stride ){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling2
static void __global__ reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 2 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 2;

    // unrolling 2 data blocks
    if(idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if( tid < stride ){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling4
static void __global__ reduceUnrolling4(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 4 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 4;

    // unrolling 4 data blocks
    if (idx + 3 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if( tid < stride ){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrolling8
static void __global__ reduceUnrolling8(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 8 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    // unrolling 8 data blocks
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>=1) {
        if( tid < stride ){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// unrollwarps8
static void __global__ reduceUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 8 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    // unrolling 8 data blocks
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>=1) {
        if( tid < stride ){
            idata[tid] += idata[tid + stride];
        }
        // synchronize within block
        __syncthreads();
    }

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// complete unrollwarps8
static void __global__ reduceCompleteUnrollWarps8(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 8 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    // unrolling 8 data blocks
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if( blockDim.x>=1024 && tid < 512 ) idata[tid] += idata[tid + 512];
    __syncthreads();

    if( blockDim.x>=512 && tid < 256 ) idata[tid] += idata[tid + 256];
    __syncthreads();

    if( blockDim.x>=256 && tid < 128 ) idata[tid] += idata[tid + 128];
    __syncthreads();

    if( blockDim.x>=128 && tid < 64 ) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// complete unrollwarps8 with template
template <unsigned int iBlockSize>
static void __global__ reduceCompleteUnrollWarps8T(int *g_idata, int *g_odata, unsigned int n){
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + 8 * blockDim.x * blockIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x * 8;

    // unrolling 8 data blocks
    if (idx + 7 * blockDim.x < n)
    {
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int a5 = g_idata[idx + 4 * blockDim.x];
        int a6 = g_idata[idx + 5 * blockDim.x];
        int a7 = g_idata[idx + 6 * blockDim.x];
        int a8 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8;
    }
    __syncthreads();

    // in-place reduction and complete unroll
    if( iBlockSize>=1024 && tid < 512 ) idata[tid] += idata[tid + 512];
    __syncthreads();

    if( iBlockSize>=512 && tid < 256 ) idata[tid] += idata[tid + 256];
    __syncthreads();

    if( iBlockSize>=256 && tid < 128 ) idata[tid] += idata[tid + 128];
    __syncthreads();

    if( iBlockSize>=128 && tid < 64 ) idata[tid] += idata[tid + 64];
    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid +  8];
        vmem[tid] += vmem[tid +  4];
        vmem[tid] += vmem[tid +  2];
        vmem[tid] += vmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
int main(int argc, char **argv) {
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s\n starting reduction at ", argv[0]);
    printf("Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1<<24;   // total number of elements to reduce 
    printf("with array size %d\n", size);

    // set up execution configuration
    int blocksize = 512;   // initial block size 
    if(argc > 1) blocksize = atoi(argv[1]);

    dim3 block(blocksize, 1);
    dim3 grid((size+block.x-1)/block.x, 1);
    printf("Execution Configure (block %d grid %d)\n", block.x, grid.x);

    // allocate cpu memory
    size_t nBytes = size * sizeof(int);
    int *h_idata = (int *) malloc(nBytes);
    int *h_odata = (int *) malloc(grid.x*sizeof(int));
    int *tmp     = (int *) malloc(nBytes);

    // initialize the array
    for(int i = 0; i<size; i++){
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, nBytes);

    double iStart, iElaps;
    int gpu_sum = 0;
    // allocate gpu memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void**)&d_idata, nBytes);
    cudaMalloc((void**)&d_odata, grid.x*sizeof(int));

    // cpu reduction
    iStart = cpuTimerInMS();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = cpuTimerInMS() - iStart;
    printf("[cpu] reduce               elapsed %5.2f millsec cpu_sum: %d\n", iElaps, cpu_sum);

    // run reduceNeighbored to warm up
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceNeighbored<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    //printf("warmingup    <<<%5d,%4d>>> elapsed %f millsec\n", grid.x, block.x, iElaps);

    // run reduceNeighbored
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceNeighbored<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceNeighbored     elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceNeighboredLess
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceNeighboredLess<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceNeighboredLess elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceInterleaved
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceInterleaved<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceInterleaved    elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceUnrolling2
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceUnrolling2<<< grid.x/2, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/2*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/2; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceUnrolling2     elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/2, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceUnrolling4
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceUnrolling4<<< grid.x/4, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/4*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/4; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceUnrolling4     elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/4, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceUnrolling8
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceUnrolling8<<< grid.x/8, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceUnrolling8     elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/8, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceUnrollWraps8
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceUnrollWarps8<<< grid.x/8, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceUnrollWarps8   elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/8, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceCompleteUnrollWraps8
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    reduceCompleteUnrollWarps8<<< grid.x/8, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceCompleteUW8    elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/8, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduceCompleteUnrollTemplate
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    switch (blocksize)
    {
    case 1024:
        reduceCompleteUnrollWarps8T<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 512:
        reduceCompleteUnrollWarps8T<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reduceCompleteUnrollWarps8T<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reduceCompleteUnrollWarps8T<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reduceCompleteUnrollWarps8T<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x/8; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduceCompleteUW8T   elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x/8, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

     
     

    // free host memory
    free(h_idata);
    free(h_odata);
    free(tmp);
    cudaFree(d_idata);
    cudaFree(d_odata);
    // reset device
    cudaDeviceReset();
    

    return 0;
}
