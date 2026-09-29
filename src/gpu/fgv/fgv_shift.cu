#include "gpu/kernels.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdexcept>
#include <string>

namespace
{
    __global__ void initialize_kernel(int num_vertices,int radius,float probability,int* shifts,int* distances,int* centers,int* parents,unsigned long long seed)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        curandState state;
        curand_init(seed,static_cast<unsigned long long>(vertex),0,&state);

        int shift = radius;

        for(int level = 0;level < radius;++level)
        {
            const float value = curand_uniform(&state);

            if(value < probability)
            {
                shift = level;
                break;
            }
        }

        shifts[vertex] = shift;
        distances[vertex] = radius - shift;
        centers[vertex] = vertex;
        parents[vertex] = -1;
    }
}

namespace spanner
{
    void initialize_fgv(int num_vertices,int radius,float probability,int* shifts,int* distances,int* centers,int* parents)
    {
        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        initialize_kernel<<<grid_size,block_size>>>(num_vertices,radius,probability,shifts,distances,centers,parents,1234567ULL);

        cudaError_t error = cudaGetLastError();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV initialization launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV initialization failed: ") + cudaGetErrorString(error));
        }
    }
}