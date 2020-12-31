#include<iostream>
#include "utils.h"

// for multi GPUS
void setBestDevice(){
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if(numDevices > 1){
        int maxMultiprocessors = 0, maxDevice = 0;
        for(int device=0; device<numDevices; device++){
            cudaDeviceProp props;
            CUDA_CHECK(cudaGetDeviceProperties(&props, device));
            if(maxMultiprocessors < props.multiProcessorCount){
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
    CUDA_CHECK(cudaSetDevice(maxDevice));
    }
}

int main(int argc, char **argv) {
    printf("%s Starting...\n", argv[0]);
    // get device information
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount ==0){
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
    int dev = 0 , driverVersion = 0, runtimeVersion = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaSetDevice(dev));
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));
    printf("  CUDA Driver Version / Runtime Version:         %d.%d / %d.%d\n", 
    driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", 
    deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.2f MBytes (%llu bytes)\n",
    (float)deviceProp.totalGlobalMem/(pow(1024.0, 3)), (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n",
    deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                             %.0f MHz)\n",
    deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
    deviceProp.memoryBusWidth);
    if(deviceProp.l2CacheSize){
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z):            1D=(%d)\n", 
    deviceProp.maxTexture1D);
    printf("                                                 2D=(%d, %d)\n", 
    deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
    printf("                                                 3D=(%d, %d, %d)\n", 
    deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[1]);
    printf("  Max Layered Texture Size (dim) x layers:       1D=(%d) x %d\n", 
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture2DLayered[1]);
    printf("                                                 2D=(%d, %d) x %d\n", 
    deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
    printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
    printf("  Total amount of constant memory:               %lu bytes\n", 
    deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n", 
    deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n", 
    deviceProp.regsPerBlock);
    printf("  Warp Size:                                     %d\n", 
    deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n", 
    deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of wraps per multiprocessor:    %d\n", 
    deviceProp.maxThreadsPerMultiProcessor/deviceProp.warpSize);
    printf("  Maximum number of threads per block:           %d\n", 
    deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n", 
    deviceProp.maxThreadsDim[0],
    deviceProp.maxThreadsDim[1],
    deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n", 
    deviceProp.maxGridSize[0],
    deviceProp.maxGridSize[1],
    deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n", 
    deviceProp.memPitch);

    // reset device
    cudaDeviceReset();
    return 0;
}
