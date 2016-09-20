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
    auto right_cols = agency::experimental::tile_evenly(right_data, n);

    agency::size2 shape{n,n};

    agency::bulk_invoke(agency::seq(shape), [=](agency::sequenced_agent_2d& self)
    {
        size_t row = self.index()[0];
        size_t col = self.index()[1];
        auto left_row = left_rows[row];
        auto right_col = right_cols[col];

        for (int k = 0; k < n; ++k) {
            out_ptr[n * row + col] += left_row[k] * right_col[k];
        }
    });
}

void parallelCpuMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_cols = agency::experimental::tile_evenly(right_data, n);

    agency::size2 shape{n,n};

    agency::bulk_invoke(agency::par(shape), [=](agency::parallel_agent_2d& self)
    {
        size_t row = self.index()[0];
        size_t col = self.index()[1];
        auto left_row = left_rows[row];
        auto right_col = right_cols[col];

        for (int k = 0; k < n; ++k) {
            out_ptr[n * row + col] += left_row[k] * right_col[k];
        }
    });
}

void parallelGpuMultiply(Matrix& left, Matrix& right, Matrix& out, size_t n) {
    agency::experimental::span<size_t> left_data(left.data(), n);
    agency::experimental::span<size_t> right_data(right.data(), n);
    size_t* out_ptr = out.data();

    auto left_rows = agency::experimental::tile_evenly(left_data, n);
    auto right_cols = agency::experimental::tile_evenly(right_data, n);

    agency::size2 shape{n,n};
    agency::cuda::parallel_executor gpu;

    agency::bulk_invoke(agency::par(shape).on(gpu), [=] __device__ (agency::parallel_agent_2d& self)
    {
        size_t row = self.index()[0];
        size_t col = self.index()[1];
        auto left_row = left_rows[row];
        auto right_col = right_cols[col];

        for (int k = 0; k < n; ++k) {
            out_ptr[n * row + col] += left_row[k] * right_col[k];
        }
    });
}

int main()
{
    size_t n = 1 << 10;

    // N.B. All multiply functions expect the right matrix to be transposed before being called.
    // However, we are only using symmetric matrices here so that doesn't matter
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
    parallelGpuMultiply(a, b, c, n);
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // Success!
    std::cout << "OK" << std::endl;
    return 0;
}
