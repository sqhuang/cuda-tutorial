#include<iostream>
#include "utils.h"


// 这里的变量 devData 作为一个标识符，并不是设备全局内存的变量地址
// 不能在主机端的设备变量中使用运算符 “&”，因为它只是一个在 GPU 上表示的物理符号
// 可以用 CUDA API cudaError_t cudaGetSymbolAddress 获取一个全局变量的地址
__device__ float devData;

void __global__ checkGlobalVariable(){
    // display the original value
    int tid = threadIdx.x;
    printf("Device: the value of the global variable is\n", devData);
    // alter the value
    devData += 2.0f;
}

int main(int argc, char **argv)
{
    // initialize the global variable
    float value = 3.14f;
    CUDA_CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable <<<1, 1>>>();
    
    // copy the global variable back to the host
    CUDA_CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}