#include<iostream>
#include "utils.h"

void sumMatrixOnHost(float *MatA, float *MatB, float *MatC, const int nx, const int ny){
    float *ia = MatA;
    float *ib = MatB;
    float *ic = MatC;

    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx; ib += nx; ic += nx;
    }
}

static void __global__ sumMatrixOnGPU2D(const float *MatA, const float *MatB, float *MatC, const int nx, const int ny){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    // map thread idx to global linear memory idx
    unsigned int idx = iy * nx + ix;
    if(ix<nx && iy<ny)
        MatC[idx] = MatA[idx] + MatB[idx];
}


int main() {
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy =  nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory
    float* h_MatA  = (float *)malloc(nBytes);
    float* h_MatB  = (float *)malloc(nBytes);
    float* hostRef  = (float *)malloc(nBytes);
    float* gpuRef  = (float *)malloc(nBytes);

    // initialize host matrix
    initialFloat(h_MatA, nxy);
    initialFloat(h_MatB, nxy);
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    double iStart = cpuTimerInMS();
    sumMatrixOnHost(h_MatA, h_MatB, hostRef, nx, ny);
    double iElaps = cpuTimerInMS() - iStart;
    printf("sumMatrixOnHost elapsed %f millsec\n", iElaps);

    // malloc
    float* d_MatA;
    CUDA_CHECK(cudaMalloc(&d_MatA, nBytes));
    float* d_MatB;
    CUDA_CHECK(cudaMalloc(&d_MatB, nBytes));
    float* d_MatC;
    CUDA_CHECK(cudaMalloc(&d_MatC, nBytes));

    // copy in
    CUDA_CHECK(cudaMemcpy(d_MatA, h_MatA, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MatB, h_MatB, nBytes, cudaMemcpyHostToDevice));
    
    // set up execution configuration
    // int dimx = 32;
    // int dimy = 32;
    // int dimx = 32;
    // int dimy = 16;
    int dimx = 16;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // invoke the kernel
    iStart = cpuTimerInMS();
    sumMatrixOnGPU2D<<< grid, block >>>(d_MatA, d_MatB, d_MatC, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInMS() - iStart;
    printf("sumMatrixOnGPU2D<<<(%d,%d), (%d,%d)>>> elapsed %f millsec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // copy out
    CUDA_CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
    
    // check results
    checkResult(hostRef, gpuRef, nxy);

    // free host and device memory
    CUDA_CHECK(cudaFree(d_MatA));
    CUDA_CHECK(cudaFree(d_MatB));
    CUDA_CHECK(cudaFree(d_MatC));
    free(h_MatA);
    free(h_MatB);
    free(hostRef);
    free(gpuRef);

    // reset device
    cudaDeviceReset();
    return 0;
}
