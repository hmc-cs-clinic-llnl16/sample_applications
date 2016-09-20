#include <agency/agency.hpp>
#include <agency/cuda.hpp>
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
/*

    // execute in parallel on the CPU
    begin_time = clock();
    bulk_invoke(par(n*n), [=](parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[n*row + col] += a_ptr[n*row + k] *
                                  b_ptr[n*k + col];
        }
    });
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel CPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on a GPU
    begin_time = clock();
    cuda::grid_executor gpu;
    bulk_invoke(par(n*n).on(gpu), [=] __device__ (parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[n*row + col] += a_ptr[n*row + k] *
                                  b_ptr[n*k + col];
        }
    });
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel Single GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on all GPUs in the system
    begin_time = clock();
    cuda::multidevice_executor all_gpus;
    bulk_invoke(par(n).on(all_gpus), [=] __device__ (parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[n*row + col] += a_ptr[n*row + k] *
                                  b_ptr[n*k + col];
        }
    });
    difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Parallel All GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);
*/
    std::cout << "OK" << std::endl;
    return 0;
}
