#include<iostream>
#include "utils.h"

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize = 1 << 22;
    unsigned int nbytes = isize * sizeof(float);

    // get device information
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    if (!deviceProp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CUDA_CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    printf("%s starting at ", argv[0]);
    printf("device %d: %s memory size %d nbyte %5.2fMB canMap %d\n", dev,
           deviceProp.name, isize, nbytes / (1024.0f * 1024.0f),
           deviceProp.canMapHostMemory);

    // allocate pinned host memory
    float *h_a;
    CUDA_CHECK(cudaMallocHost ((float **)&h_a, nbytes));

    // allocate device memory
    float *d_a;
    CUDA_CHECK(cudaMalloc((float **)&d_a, nbytes));

    // initialize host memory
    memset(h_a, 0, nbytes);

    for (int i = 0; i < isize; i++) h_a[i] = 100.10f;

    // transfer data from the host to the device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CUDA_CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));

    // free memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFreeHost(h_a));

    // reset device
    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}