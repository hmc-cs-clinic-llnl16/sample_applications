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
    using vector = std::vector<size_t, cuda::managed_allocator<size_t>>;
    using matrix = std::vector<vector, cuda::managed_allocator<vector>>;
    
    size_t n = 1 << 10;
    
    matrix a(n, vector(n, 1));
    matrix b(n, vector(n, 1));
    matrix c(n, vector(n, 0));
    
    vector* a_ptr = a.data();
    vector* b_ptr = b.data();
    vector* c_ptr = c.data();
    
    // execute sequentially in the current thread
    clock_t begin_time = clock();
    bulk_invoke(seq(n*n), [=](sequenced_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[row].data()[col] += a_ptr[row].data()[k] * 
                                      b_ptr[k].data()[col];
        }
    });
    clock_t difference = clock() - begin_time;

    for (auto i = c.begin(); i < c.end(); ++i) {
        for (auto j = i -> begin(); j < i -> end(); ++j) {
            assert(*j == n);
            *j = 0;
        };
    }

    printf("Sequential Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on the CPU
    begin_time = clock();
    bulk_invoke(par(n*n), [=](parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[row].data()[col] += a_ptr[row].data()[k] * 
                                      b_ptr[k].data()[col];
        }
    });
    difference = clock() - begin_time;

    for (auto i = c.begin(); i < c.end(); ++i) {
        for (auto j = i -> begin(); j < i -> end(); ++j) {
            assert(*j == n);
            *j = 0;
        };
    }

    printf("Parallel CPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on a GPU
    begin_time = clock();
    cuda::grid_executor gpu;
    bulk_invoke(par(n*n).on(gpu), [=] __device__ (parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[row].data()[col] += a_ptr[row].data()[k] *
                                      b_ptr[k].data()[col];
        }
    });
    difference = clock() - begin_time;

    for (auto i = c.begin(); i < c.end(); ++i) {
        for (auto j = i -> begin(); j < i -> end(); ++j) {
            assert(*j == n);
            *j = 0;
        };
    }

    printf("Parallel Single GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    // execute in parallel on all GPUs in the system
    begin_time = clock();
    cuda::multidevice_executor all_gpus;
    bulk_invoke(par(n).on(all_gpus), [=] __device__ (parallel_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[row].data()[col] += a_ptr[row].data()[k] *
                                      b_ptr[k].data()[col];
        }
    });
    difference = clock() - begin_time;

    for (auto i = c.begin(); i < c.end(); ++i) {
        for (auto j = i -> begin(); j < i -> end(); ++j) {
            assert(*j == n);
            *j = 0;
        };
    }

    printf("Parallel All GPU Execution took %ld clicks (%f seconds).\n", difference, ((float) difference)/CLOCKS_PER_SEC);

    std::cout << "OK" << std::endl;
    return 0;
}
