#include "RAJA/RAJA.hxx"
#include <random>
#include <vector>
#include "mmult.cuh"
#include "caliper/Annotation.h"

template < typename T >
T* alloc(size_t size)
{
    T* result;
    cudaErrchk(cudaMalloc((void**)&result, size * sizeof(T)));
    cudaErrchk(cudaMemset(result, 0, size * sizeof(T)));
    cudaErrchk(cudaDeviceSynchronize());
    return result;
}

template < typename T>
T* toGPU(size_t size, T* src)
{
    T* result = alloc<T>(size);
    cudaErrchk(cudaMemcpy(result, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    return result;
}

template < typename T >
T* toCPU(size_t size, T* src)
{
    T* result = (double*)malloc(size * sizeof(T));
    cudaErrchk(cudaMemcpy(result, src, size * sizeof(T), cudaMemcpyHostToDevice));
    return result;
}

template < typename Policy, typename View >
double* mmult1(double* left, double* right, size_t numRows, size_t numCols)
{
    double* result = alloc<double>(numRows * numCols);

    View leftView(left, numRows, numCols);
    View rightView(right, numCols, numRows);
    View resultView(result, numRows, numCols);

    RAJA::forall<Policy>(0, numRows * numCols,
        [=] __device__ (RAJA::Index_type rowCol) {
            int row = (rowCol - numCols) / numRows;
            int col = (rowCol - numCols) % numRows;
            double tmpResult = 0;
            for (int k = 0; k < numCols; ++k) {
                tmpResult += leftView(row, k) * rightView(k, col);
            }
            resultView(row, col) = tmpResult;
        });

    return result;
}

template < typename Policy, typename View >
double* mmult2(double* left, double* right, size_t numRows, size_t numCols)
{
    double* result = alloc<double>(numRows * numCols);

    View leftView(left, numRows, numCols);
    View rightView(right, numCols, numRows);
    View resultView(result, numRows, numCols);

    RAJA::forall<Policy>(0, numRows,
        [=] __device__ (RAJA::Index_type i) {
            for (int j = 0; j < numCols; ++j) {
                double tmpResult = 0;
                for (int k = 0; k < numCols; ++k) {
                    tmpResult += leftView(i, k) * rightView(k, j);
                }
                resultView(i, j) = tmpResult;
            }
        });

    return result;
}


int main()
{
    using View = RAJA::View<double, RAJA::Layout<int, RAJA::PERM_IJ, int, int>>;
    const int NUM_TRIALS = 10;

    std::mt19937 randomNumberEngine;
    std::uniform_real_distribution<double> rng(0, 10000);

    const int sizes[] = {100, 500, 1000, 2500, 5000, 10000};

    auto sizeAnn = cali::Annotation("size");
    auto iteration = cali::Annotation("iteration");

    for (int i = 0; i < NUM_TRIALS; ++i) {
        std::cout << "Beginning iteration " << i << "\n";
        iteration.set(i);
        
        for (int size : sizes) {
            std::cout << "Current size: " << size << "\n";
            sizeAnn.set(size);

            std::vector<double> left(size * size);
            std::vector<double> right(size * size);
            std::vector<double> result(size * size);

            for (int i = 0; i < size; ++i) {
                for (int j = 0; j < size; ++j) {
                    left[i * size + j] = rng(randomNumberEngine);
                    right[i * size + j] = rng(randomNumberEngine);
                }
            }

            double* dev_left;
            double* dev_right;
            double* dev_result;
            cudaErrchk(cudaMalloc((void**)&dev_left, size*size*sizeof(double)));
            cudaErrchk(cudaMalloc((void**)&dev_right, size*size*sizeof(double)));
            cudaErrchk(cudaMemcpy(dev_left, &left[0], size*size*sizeof(double), cudaMemcpyHostToDevice));
            cudaErrchk(cudaMemcpy(dev_right, &right[0], size*size*sizeof(double), cudaMemcpyHostToDevice));

            std::cout << "Mmult1 cuda\n";
            auto v1cuda = cali::Annotation("mode");
            v1cuda.set("Mmult1_cuda");
            dev_result = mmult1<RAJA::cuda_exec<128>, View>(dev_left, dev_right, size, size);
            v1cuda.end();
            cudaErrchk(cudaMemcpy(&result[0], dev_result, size*size*sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << result[0] << "\n";
            cudaFree((void*)dev_result);

            std::cout << "Mmult1 agency\n";
            auto v1agency = cali::Annotation("mode");
            v1agency.set("Mmult1_agency");
            dev_result = mmult1<RAJA::agency_cuda_exec<128>, View>(dev_left, dev_right, size, size);
            v1agency.end();
            cudaErrchk(cudaMemcpy(&result[0], dev_result, size*size*sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << result[0] << "\n";
            cudaFree((void*)dev_result);

            std::cout << "Mmult2 cuda\n";
            auto v2cuda = cali::Annotation("mode");
            v2cuda.set("Mmult2_cuda");
            dev_result = mmult2<RAJA::cuda_exec<128>, View>(dev_left, dev_right, size, size);
            v2cuda.end();
            cudaErrchk(cudaMemcpy(&result[0], dev_result, size*size*sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << result[0] << "\n";
            cudaFree((void*)dev_result);

            std::cout << "Mmult2 agency\n";
            auto v2agency = cali::Annotation("mode");
            v2agency.set("Mmult2_agency");
            dev_result = mmult2<RAJA::agency_cuda_exec<128>, View>(dev_left, dev_right, size, size);
            v2agency.end();
            cudaErrchk(cudaMemcpy(&result[0], dev_result, size*size*sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << result[0] << "\n";
            cudaFree((void*)dev_result);

            //auto raw = cali::Annotation("Mmult_raw");
            //result = mmultGpu<1024>(left, right, size, size);
            //v2cuda.end();
            //printf("%f", result[0]);
            //cudaFree(result);

            cudaFree(dev_left);
            cudaFree(dev_right);
        }
    }
    iteration.end();
    sizeAnn.end();
}

