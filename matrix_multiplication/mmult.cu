#include "RAJA/RAJA.hxx"
#include <cuda.h>
#include <cuda_runtime.h>
#include "mmult.cuh"

__global__ void mmultKernel(const double* __restrict__ left, 
                            const double* __restrict__ right, 
                            double* __restrict__ result, 
                            const size_t numRows, const size_t numCols)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int stepSize = blockDim.x * gridDim.x;


    for (int i = threadIndex; i < numRows * numCols; i += stepSize) {
        int row = (i - numCols) / numRows;
        int col = (i - numCols) % numRows;
        
        double tmp = 0.0;
        for (int k = 0; k < numCols; ++k) {
            tmp += left[row * numCols + k] * right[k * numCols + col];
        }
        result[row * numCols + col] = tmp;
    }
}

template <size_t BLOCK_SIZE>
double* mmultGpu(double* left, double* right, size_t numRows, size_t numCols)
{
    double* result;
    cudaMallocManaged((void**)&result,
                                 numRows * numCols * sizeof(double),
                                 cudaMemAttachGlobal);
    cudaMemset(result, 0, numRows * numCols * sizeof(double));
    cudaDeviceSynchronize();

    size_t gridSize = RAJA_DIVIDE_CEILING_INT(numRows * numCols, BLOCK_SIZE);
    gridSize = RAJA_MIN(gridSize, RAJA_CUDA_MAX_NUM_BLOCKS);

    mmultKernel<<<gridSize, BLOCK_SIZE>>>(left, right, result, numRows, numCols);
    cudaDeviceSynchronize();

    return result;
}

