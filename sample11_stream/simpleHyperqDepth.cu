#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "utils.h"

/*
 * This example demonstrates submitting work to a CUDA stream in depth-first
 * order. Work submission in depth-first order may introduce false-dependencies
 * between unrelated tasks in different CUDA streams, limiting the parallelism
 * of a CUDA application. kernel_1, kernel_2, kernel_3, and kernel_4 simply
 * implement identical, dummy computation. Separate kernels are used to make the
 * scheduling of these kernels simpler to visualize in the Visual Profiler.
 */

#define N 300000
#define NSTREAM 4

__global__ void kernel_1()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_2()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_3()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

__global__ void kernel_4()
{
    double sum = 0.0;

    for(int i = 0; i < N; i++)
    {
        sum = sum + tan(0.1) * tan(0.1);
    }
}

int main(int argc, char **argv)
{
    int n_streams = NSTREAM;
    int isize = 1;
    int iblock = 1;
    int bigcase = 0;

    // get argument from command line
    if (argc > 1) n_streams = atoi(argv[1]);

    if (argc > 2) bigcase = atoi(argv[2]);

    float elapsed_time;

    // set up max connectioin
    // 并行硬件连接数量，每个 CUDA 流都会被映射到单一 CUDA 设备连接中。
    // 如果流的数量超过了硬件连接数量，多个流将共享一个连接。
    // 当多个流共享相同的硬件工作队列时，肯呢个会产生虚假的依赖关系。

    char* iname = "CUDA_DEVICE_MAX_CONNECTIONS";
    setenv (iname, "32", 1);
    /*
    NAME
           setenv - change or add an environmentvariable      //改变或添加一个环境变量
    SYNOPSIS
           #include <stdlib.h>
           int setenv(const char *name, const char*value, int overwrite);
               int unsetenv(const char *name);
    
    overwrite参数：
        非0表示覆盖原有环境变量，0表示不覆盖。
    */
    char *ivalue =  getenv (iname);
    printf ("%s = %s\n", iname, ivalue);

    int dev = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("> Using Device %d: %s with num_streams=%d\n", dev, deviceProp.name,
           n_streams);
    CUDA_CHECK(cudaSetDevice(dev));

    // check if device support hyper-q
    if (deviceProp.major < 3 || (deviceProp.major == 3 && deviceProp.minor < 5))
    {
        if (deviceProp.concurrentKernels == 0)
        {
            printf("> GPU does not support concurrent kernel execution (SM 3.5 "
                    "or higher required)\n");
            printf("> CUDA kernel runs will be serialized\n");
        }
        else
        {
            printf("> GPU does not support HyperQ\n");
            printf("> CUDA kernel runs will have limited concurrency\n");
        }
    }

    printf("> Compute Capability %d.%d hardware with %d multi-processors\n",
           deviceProp.major, deviceProp.minor, deviceProp.multiProcessorCount);

    // Allocate and initialize an array of stream handles
    cudaStream_t *streams = (cudaStream_t *) malloc(n_streams * sizeof(
                                cudaStream_t));

    for (int i = 0 ; i < n_streams ; i++)
    {
        CUDA_CHECK(cudaStreamCreate(&(streams[i])));
    }

    // run kernel with more threads
    if (bigcase == 1)
    {
        iblock = 512;
        isize = 1 << 12;
    }

    // set up execution configuration
    dim3 block (iblock);
    dim3 grid  (isize / iblock);
    printf("> grid %d block %d\n", grid.x, block.x);

    // creat events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // dispatch job with depth first ordering
    for (int i = 0; i < n_streams; i++)
    {
        kernel_1<<<grid, block, 0, streams[i]>>>();
        kernel_2<<<grid, block, 0, streams[i]>>>();
        kernel_3<<<grid, block, 0, streams[i]>>>();
        kernel_4<<<grid, block, 0, streams[i]>>>();
    }

    // record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // calculate elapsed time
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
    printf("Measured time for parallel execution = %.3fs\n",
           elapsed_time / 1000.0f);

    // release all stream
    for (int i = 0 ; i < n_streams ; i++)
    {
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    free(streams);

    // destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // reset device
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}