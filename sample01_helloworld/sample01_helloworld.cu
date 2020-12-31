#include <stdio.h>
__global__ void cuda_hello(){
    if(threadIdx.x==1)
    printf("Hello World from GPU!\n");
}

int main() {
    cuda_hello<<<10,10>>>();
    cudaDeviceSynchronize();
    return 0;
}
