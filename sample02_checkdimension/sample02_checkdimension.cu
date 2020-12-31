#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

__global__ void checkIndex(void){
    printf("[KERNEL side] threadIdx:(%d, %d, %d)  blockIdx:(%d, %d, %d)  blockDim:(%d, %d, %d)  gridDim:(%d, %d, %d)\n",
        threadIdx.x, threadIdx.y, threadIdx.z,
        blockIdx.x, blockIdx.y, blockIdx.z, 
        blockDim.x, blockDim.y, blockDim.z,
        gridDim.x, gridDim.y, gridDim.z);
}

int main(){
    // define total data element
    int nElem = 6;

    // define grid and block structure
    dim3 block(3);
    dim3 grid ((nElem+block.x-1)/block.x);
    
    // check grid and block dimension from host side
    std::cout<< "[HOST side] grid.x: "  << grid.x   << " grid.y: "  << grid.y   <<" grid.z: " << grid.z   <<std::endl;
    std::cout<< "[HOST side] block.x: " << block.x  << " block.y: " << block.y  <<" block.z: "<< block.z  <<std::endl;
    
    // check grid and block dimension from device side
    checkIndex<<<grid, block>>>();

    //reset device before you leave
    cudaDeviceReset();
    return 0;
}
