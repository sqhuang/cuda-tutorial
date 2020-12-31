#ifndef __UTILS__
#define __UTILS__

#include <sys/time.h>

#define CUDA_CHECK(call)                                                         \
{                                                                                \
    const cudaError_t error = call;                                              \
    if(error != cudaSuccess)                                                     \
    {                                                                            \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
        printf("code: %d, reason: %s, ", error, cudaGetErrorString(error));      \
        exit(1);                                                                 \
    }                                                                            \
}


// cpu timer
// microsecond from 1970.01.01
double cpuTimerInUS(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec * 1.e6 + (double)tp.tv_usec);
}

// millisecond from 1970.01.01
double cpuTimerInMS(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec * 1.e3 + (double)tp.tv_usec * 1.e-3);
}

// second from 1970.01.01
double cpuTimerInS(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif
