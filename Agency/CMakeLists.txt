cmake_minimum_required(VERSION 3.1)

set(
        CUDA_TOOLKIT_ROOT_DIR
        /clinic/2016/llnl16/test/cuda)

find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3;
        -arch=sm_32;
        -std=c++11;
        --expt-extended-lambda
)

# Specify include directories
include_directories(
        /clinic/2016/llnl16/test/agency
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        mmultAgency
        mmultAgency.cu
)
