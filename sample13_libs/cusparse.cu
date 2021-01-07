#include <stdio.h>
#include <stdlib.h>
#include <cusparse_v2.h>
#include <cuda.h>
#include "utils.h"

/*
 * This is an example demonstrating usage of the cuSPARSE library to perform a
 * sparse matrix-vector multiplication on randomly generated data.
 */

/*
 * M = # of rows
 * N = # of columns
 */
int M = 1024;
int N = 1024;

/*
 * Generate a vector of length N with random single-precision floating-point
 * values between 0 and 100.
 */
void generate_random_vector(int N, float **outX)
{
    int i;
    double rMax = (double)RAND_MAX;
    float *X = (float *)malloc(sizeof(float) * N);

    for (i = 0; i < N; i++)
    {
        int r = rand();
        double dr = (double)r;
        X[i] = (dr / rMax) * 100.0;
    }

    *outX = X;
}

/*
 * Generate random dense matrix A in column-major order, while rounding some
 * elements down to zero to ensure it is sparse.
 */
int generate_random_dense_matrix(int M, int N, float **outA)
{
    int i, j;
    double rMax = (double)RAND_MAX;
    float *A = (float *)malloc(sizeof(float) * M * N);
    int totalNnz = 0;

    for (j = 0; j < N; j++)
    {
        for (i = 0; i < M; i++)
        {
            int r = rand();
            float *curr = A + (j * M + i);

            if (r % 3 > 0)
            {
                *curr = 0.0f;
            }
            else
            {
                double dr = (double)r;
                *curr = (dr / rMax) * 100.0;
            }

            if (*curr != 0.0f)
            {
                totalNnz++;
            }
        }
    }

    *outA = A;
    return totalNnz;
}

int main(int argc, char **argv)
{
    int row;
    float *A, *dA;
    int *dNnzPerRow;
    float *dCsrValA;
    int *dCsrRowPtrA;
    int *dCsrColIndA;
    int totalNnz;
    float alpha = 3.0f;
    float beta = 4.0f;
    float *dX, *X;
    float *dY, *Y;
    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    // Generate input
    srand(9384);
    int trueNnz = generate_random_dense_matrix(M, N, &A);
    generate_random_vector(N, &X);
    generate_random_vector(M, &Y);

    // Create the cuSPARSE handle
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Allocate device memory for vectors and the dense form of the matrix A
    CUDA_CHECK(cudaMalloc((void **)&dX, sizeof(float) * N));
    CUDA_CHECK(cudaMalloc((void **)&dY, sizeof(float) * M));
    CUDA_CHECK(cudaMalloc((void **)&dA, sizeof(float) * M * N));
    CUDA_CHECK(cudaMalloc((void **)&dNnzPerRow, sizeof(int) * M));

    // Construct a descriptor of the matrix A
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // Transfer the input vectors and dense matrix A to the device
    CUDA_CHECK(cudaMemcpy(dX, X, sizeof(float) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, Y, sizeof(float) * M, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    // Compute the number of non-zero elements in A
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, M, N, descr, dA,
                                M, dNnzPerRow, &totalNnz));

    if (totalNnz != trueNnz)
    {
        fprintf(stderr, "Difference detected between cuSPARSE NNZ and true "
                "value: expected %d but got %d\n", trueNnz, totalNnz);
        return 1;
    }

    // Allocate device memory to store the sparse CSR representation of A
    CUDA_CHECK(cudaMalloc((void **)&dCsrValA, sizeof(float) * totalNnz));
    CUDA_CHECK(cudaMalloc((void **)&dCsrRowPtrA, sizeof(int) * (M + 1)));
    CUDA_CHECK(cudaMalloc((void **)&dCsrColIndA, sizeof(int) * totalNnz));

    // Convert A from a dense formatting to a CSR formatting, using the GPU
    CHECK_CUSPARSE(cusparseSdense2csr(handle, M, N, descr, dA, M, dNnzPerRow,
                                      dCsrValA, dCsrRowPtrA, dCsrColIndA));

    // Perform matrix-vector multiplication with the CSR-formatted matrix A
    CHECK_CUSPARSE(cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  M, N, totalNnz, &alpha, descr, dCsrValA,
                                  dCsrRowPtrA, dCsrColIndA, dX, &beta, dY));

    // Copy the result vector back to the host
    CUDA_CHECK(cudaMemcpy(Y, dY, sizeof(float) * M, cudaMemcpyDeviceToHost));

    for (row = 0; row < 10; row++)
    {
        printf("%2.2f\n", Y[row]);
    }

    printf("...\n");

    free(A);
    free(X);
    free(Y);

    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dY));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dNnzPerRow));
    CUDA_CHECK(cudaFree(dCsrValA));
    CUDA_CHECK(cudaFree(dCsrRowPtrA));
    CUDA_CHECK(cudaFree(dCsrColIndA));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));


    return 0;
}