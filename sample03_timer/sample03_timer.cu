#include<iostream>
#include "utils.h"

template <class T>
static void __global__ _cuda_relu(const T* a, unsigned total, T* b)
{
    const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    const unsigned skip = blockDim.x * gridDim.x;
    for (unsigned i = tid; i < total; i += skip)
    {
        T v = a[i];
        b[i] = max(v, T(0));
    }
}


int main() {
    unsigned int size = 100;
    float* input  = (float*)malloc(size * sizeof(float));
    float* output = (float*)malloc(size * sizeof(float));
    for(int i = 0; i < size; i++){
        input[i] = rand()/double(RAND_MAX) - 0.5;
    }

    // malloc
    float* pinput;
    CUDA_CHECK(cudaMalloc(&pinput,  size * sizeof(float)));
    float* poutput;
    CUDA_CHECK(cudaMalloc(&poutput, size * sizeof(float)));

    // copy
    CUDA_CHECK(cudaMemcpy(pinput, input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    int blockCount  = 1;
    int threadCount = 10;

    double timestart_cpu = cpuTimerInMS();
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    _cuda_relu<<<blockCount, threadCount, 0, 0>>>(pinput, size, poutput);
    CUDA_CHECK(cudaMemcpy(output, poutput, size*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(start));
    CUDA_CHECK(cudaEventSynchronize(stop));
    // calculate the elapsed time between two events

    float timeelaps_cuda;
    CUDA_CHECK(cudaEventElapsedTime(&timeelaps_cuda, start, stop));
    double timeelaps_cpu = cpuTimerInMS() - timestart_cpu;
    std::cout<<"[CPU]   time elapsed in millisecond: "<< timeelaps_cpu <<"" << std::endl;
    std::cout<<"[CUDA]  time elapsed in millisecond: "<< timeelaps_cuda <<"" << std::endl;
//    for(int i=0; i<size; i++){
//        std::cout<<"in[" << i <<"]: "<<input[i]<<"\t out[" << i <<"]: "<<output[i]<<std::endl;
//    }
    // free
    CUDA_CHECK(cudaFree(pinput));
    CUDA_CHECK(cudaFree(poutput));
    free(input);
    free(output);
    return 0;
}
