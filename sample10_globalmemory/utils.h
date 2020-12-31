#ifndef __UTILS__
#define __UTILS__

#include <sys/time.h>
#include <stdio.h>

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

// data

void initialInt(int *ip, int size){
    for(int i=0; i<size; i++){
        ip[i] = i;
    }
}

void initialFloat(float *ip, int size){
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for(int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF)/10.0f;
    }
}

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = true;
    for(int i=0; i<N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = false;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current idx %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
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
