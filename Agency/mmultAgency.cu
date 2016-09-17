#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <ctime>

int main()
{
    using namespace agency;
    // allocate data in GPU memory
    using matrix = std::vector<size_t, cuda::managed_allocator<size_t>>;
    
    size_t n = 1 << 10;

    matrix a(n*n, 1);
    matrix b(n*n, 1);
    matrix c(n*n, 0);

    matrix reference(n*n, n);

    size_t* a_ptr = a.data();
    size_t* b_ptr = b.data();
    size_t* c_ptr = c.data();
    
    // execute sequentially in the current thread
    clock_t begin_time = clock();
    bulk_invoke(seq(n*n), [=](sequenced_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[n*row + col] += a_ptr[n*row + k] *
                                  b_ptr[n*k + col];
        }
    });
    clock_t difference = clock() - begin_time;

    assert(c == reference);
    std::fill(c.begin(), c.end(), 0);

    printf("Sequential Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

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

    std::cout << "OK" << std::endl;
    return 0;
}
