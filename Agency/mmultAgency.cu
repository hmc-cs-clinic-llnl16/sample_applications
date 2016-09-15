#include <agency/agency.hpp>
#include <agency/cuda.hpp>
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

int main()
{
    using namespace agency;
    // allocate data in GPU memory
    using vector = std::vector<int, cuda::managed_allocator<int>>;
    using matrix = std::vector<vector, cuda::managed_allocator<vector>>;
    size_t n = 1 << 20;
    matrix a(n, vector(n, 1));
    matrix b(n, vector(n, 1));
    matrix c(n, vector(n, 0));
    vector* a_ptr = a.data();
    vector* b_ptr = b.data();
    vector* c_ptr = c.data();
    // execute sequentially in the current thread
    bulk_invoke(seq(n*n), [=](sequenced_agent& self)
    {
        int row = self.index() / n;
        int col = self.index() % n;

        for (int k = 0; k < n; ++k) {
            c_ptr[row].data()[col] += a_ptr[row].data()[k] + b_ptr[k].data()[col];
        }
    });
    for (auto i = c.begin(); i < c.end(); ++i) {
        for (auto j = i -> begin(); j < i -> end(); ++j) {
            assert(*j == n);
            *j = 0;
        };
    }
    /*// execute in parallel on the CPU
    bulk_invoke(par(n), [=](parallel_agent& self)
    {
        int i = self.index();
        z_ptr[i] = a * x_ptr[i] + y_ptr[i];
    });
    assert(z == reference);
    std::fill(z.begin(), z.end(), 0);
    // execute in parallel on a GPU
    cuda::grid_executor gpu;
    bulk_invoke(par(n).on(gpu), [=] __device__ (parallel_agent& self)
    {
        int i = self.index();
        z_ptr[i] = a * x_ptr[i] + y_ptr[i];
    });
    assert(z == reference);
    std::fill(z.begin(), z.end(), 0);
    // execute in parallel on all GPUs in the system
    cuda::multidevice_executor all_gpus;
    bulk_invoke(par(n).on(all_gpus), [=] __device__ (parallel_agent& self)
    {
        int i = self.index();
        z_ptr[i] = a * x_ptr[i] + y_ptr[i];
    });
    assert(z == reference);
    std::fill(z.begin(), z.end(), 0);*/
    std::cout << "OK" << std::endl;
    return 0;
}