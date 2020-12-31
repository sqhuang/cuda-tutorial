#include<iostream>
#include "utils.h"

/*
 * This example demonstrates using explicit CUDA memory transfer to implement
 * matrix addition. This code contrasts with sumMatrixGPUManaged.cu, where CUDA
 * managed memory is used to remove all explicit memory transfers and abstract
 * away the concept of physicall separate address spaces.
 */

void initialData(float *ip, const int size)
{
    initialFloat(ip, size);
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }

        ia += nx;
        ib += nx;
        ic += nx;
    }

    return;
}

// grid 2D block 2D
__global__ void sumMatrixGPU(float *MatA, float *MatB, float *MatC, int nx,
                             int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting ", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx, ny;
    int ishift = 12;

    if  (argc > 1) ishift = atoi(argv[1]);

    nx = ny = 1 << ishift;

    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = cpuTimerInS();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuTimerInS() - iStart;

    printf("initialization: \t %f sec\n", iElaps);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = cpuTimerInS();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuTimerInS() - iStart;
    printf("sumMatrix on host:\t %f sec\n", iElaps);

    // malloc device global memory
    float *d_MatA, *d_MatB, *d_MatC;
    CUDA_CHECK(cudaMalloc((void **)&d_MatA, nBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_MatB, nBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_MatC, nBytes));

    // invoke kernel at host side
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    // init device data to 0.0f, then warm-up kernel to obtain accurate timing
    // result
    CUDA_CHECK(cudaMemset(d_MatA, 0.0f, nBytes));
    CUDA_CHECK(cudaMemset(d_MatB, 0.0f, nBytes));
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, 1, 1);


    // transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    iStart =  cpuTimerInS();
    sumMatrixGPU<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);

    CUDA_CHECK(cudaDeviceSynchronize());
    iElaps = cpuTimerInS() - iStart;
    printf("sumMatrix on gpu :\t %f sec <<<(%d,%d), (%d,%d)>>> \n", iElaps,
            grid.x, grid.y, block.x, block.y);

    CUDA_CHECK(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));

    // check kernel error
    CUDA_CHECK(cudaGetLastError());

    // check device results
    checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CUDA_CHECK(cudaFree(d_MatA));
    CUDA_CHECK(cudaFree(d_MatB));
    CUDA_CHECK(cudaFree(d_MatC));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return (0);
}