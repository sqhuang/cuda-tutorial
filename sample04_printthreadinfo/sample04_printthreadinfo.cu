#include<iostream>
#include "utils.h"

static void __global__ printThreadIndex(int *A, const int nx, const int ny)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    unsigned int idx = iy * nx + ix;
    printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global index %2d ival %2d\n", 
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

void initialInt(int *ip, int size){
    for(int i=0; i<size; i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            printf("%3d", ic[ix]); 
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

int main() {
    // get device information
    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));

    // set matrix dimension
    int nx = 8;
    int ny = 6;
    int nxy =  nx * ny;
    int nBytes = nxy * sizeof(int);

    // malloc host memory
    int* h_A  = (int *)malloc(nBytes);

    // initialize host matrix with interger
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    // malloc
    int* d_A;
    CUDA_CHECK(cudaMalloc(&d_A, nBytes));

    // copy
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    
    // set up execution configuration
    dim3 block(4, 2);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // invoke the kernel
    printThreadIndex<<< grid, block >>>(d_A, nx, ny);
    CUDA_CHECK(cudaDeviceSynchronize());

    // free host and device memory
    CUDA_CHECK(cudaFree(d_A));
    free(h_A);

    // reset device
    cudaDeviceReset();
    return 0;
}
