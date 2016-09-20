#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <agency/experimental.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <ctime>

// allocate data in GPU memory
using Matrix = std::vector<size_t, agency::cuda::managed_allocator<size_t>>;

void sequentialMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_rows = agency::experimental::tile_evenly(right_data, n);

    agency::bulk_invoke(agency::seq(n), [=](agency::sequenced_agent& outer)
    {
        auto left_row = left_rows[outer.index()];
        agency::bulk_invoke(agency::seq(n), [=](agency::sequenced_agent& inner)
        {
            auto right_row = right_rows[inner.index()];
            for (int k = 0; k < n; ++k) {
                out_ptr[n * outer.index() + inner.index()] += left_row[k] * right_row[k];
            }
        });
    });
}

void parallelCpuMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_rows = agency::experimental::tile_evenly(right_data, n);

    agency::bulk_invoke(agency::par(n), [=](agency::parallel_agent& outer)
    {
        auto left_row = left_rows[outer.index()];
        agency::bulk_invoke(agency::par(n), [=](agency::parallel_agent& inner)
        {
            auto right_row = right_rows[inner.index()];
            for (int k = 0; k < n; ++k) {
                out_ptr[n * outer.index() + inner.index()] += left_row[k] * right_row[k];
            }
        });
    });
}

void parallelSingleGpuMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_rows = agency::experimental::tile_evenly(right_data, n);
    agency::cuda::grid_executor gpu;

    agency::bulk_invoke(agency::par(n).on(gpu), [=] __device__ (agency::parallel_agent& outer)
    {
        auto left_row = left_rows[outer.index()];
        agency::bulk_invoke(agency::par(n).on(gpu), [=] __device__ (agency::parallel_agent& inner)
        {
            auto right_row = right_rows[inner.index()];
            for (int k = 0; k < n; ++k) {
                out_ptr[n * outer.index() + inner.index()] += left_row[k] * right_row[k];
            }
        });
    });
}

void parallelAllGpuMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_rows = agency::experimental::tile_evenly(right_data, n);
    agency::cuda::multidevice_executor all_gpus;

    agency::bulk_invoke(agency::par(n).on(all_gpus), [=] __device__ (agency::parallel_agent& outer)
    {
        auto left_row = left_rows[outer.index()];
        agency::bulk_invoke(agency::par(n).on(all_gpus), [=] __device__ (agency::parallel_agent& inner)
        {
            auto right_row = right_rows[inner.index()];
            for (int k = 0; k < n; ++k) {
                out_ptr[n * outer.index() + inner.index()] += left_row[k] * right_row[k];
            }
        });
    });
}

int main()
{
    size_t n = 1 << 10;

    Matrix a(n*n, 1);
    Matrix b(n*n, 1);
    Matrix c(n*n, 0);

    Matrix reference(n*n, n);
    
    // execute sequentially in the current thread
    clock_t begin_time = clock();
    sequentialMultiply(a, b, c, n);
    clock_t difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Sequential Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on the CPU
    begin_time = clock();
    parallelCpuMultiply(a, b, c, n);
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel CPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on a GPU
    begin_time = clock();
    parallelSingleGpuMultiply(a, b, c, n);
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel single GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on all GPUs in the system
    begin_time = clock();
    parallelAllGpuMultiply(a, b, c, n);
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel All GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // Success!
    std::cout << "OK" << std::endl;
    return 0;
}
