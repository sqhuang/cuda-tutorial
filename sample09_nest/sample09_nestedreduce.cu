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

// gpu recursive
static void __global__ gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned int isize){
    // set thread ID
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x;
    int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation
    isize >>= 1;
    if(isize > 1 && tid < isize){
        // in place reduction
        idata[tid] += idata[tid + isize];
    }
    // sync at block level
    __syncthreads();

    // nested invocation to generate child grids
    if(tid==0){
        gpuRecursiveReduce<<<1, isize>>>(idata, odata, isize);
        //sync all child grids launched in this block
        cudaDeviceSynchronize();
    }
    // sync at block level again
    __syncthreads();
}

// gpu recursive no sync
static void __global__ gpuRecursiveReduceNosync(int *g_idata, int *g_odata, unsigned int isize){
    // set thread ID
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockDim.x * blockIdx.x;
    int *odata = &g_odata[blockIdx.x];

    // stop condition
    if (isize == 2 && tid == 0) {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    // nested invocation
    isize >>= 1;
    if(isize > 1 && tid < isize){
        // in place reduction
        idata[tid] += idata[tid + isize];
    }

    // nested invocation to generate child grids
    if(tid==0){
        gpuRecursiveReduce<<<1, isize>>>(idata, odata, isize);
    }
}

// gpu recursive no sync
static void __global__ gpuRecursiveReduce2(int *g_idata, int *g_odata, unsigned int istride, int const iDim){
    // set thread ID
    unsigned int tid = threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + iDim * blockIdx.x;

    // stop condition
    if (istride == 1 && tid == 0) {
        g_odata[tid] = idata[0] + idata[1];
        return;
    }

    // in place reduction
    idata[tid] += idata[tid + istride];

    // nested invocation to generate child grids
    if(tid==0 && blockIdx.x==0){
        gpuRecursiveReduce2<<<gridDim.x, istride/2>>>(g_idata, g_odata, istride/2, iDim);
    }
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

    // run reduce recursive gpu
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    gpuRecursiveReduce<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduce nested        elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduce recursive gpu
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    gpuRecursiveReduceNosync<<< grid, block >>>(d_idata, d_odata, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduce nested nosync elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
    //check the results
    bResult = (gpu_sum == cpu_sum);
    if(!bResult) printf("Test failed!\n");

    // run reduce recursive gpu
    CUDA_CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    iStart = cpuTimerInMS();
    gpuRecursiveReduce2<<< grid, block >>>(d_idata, d_odata, size, block.x);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    CUDA_CHECK(cudaMemcpy(h_odata, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
    gpu_sum = 0;
    for (int i = 0; i<grid.x; i++) gpu_sum += h_odata[i];
    printf("[gpu] reduce nested 2      elapsed %5.2f millsec gpu_sum: %d <<<%5d,%4d>>> \n", iElaps, gpu_sum, grid.x, block.x);
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
