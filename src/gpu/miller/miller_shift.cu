#include "gpu/kernels.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace
{
    __global__ void initialize_kernel(int num_vertices,float beta,float* shifts,int* distances,int* centers,int* parents,unsigned long long seed)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        curandState state;
        curand_init(seed, static_cast<unsigned long long>(vertex), 0, &state);

        const float uniform_value = curand_uniform(&state);
        const float safe_u = fmaxf(1.0e-7f, 1.0f - uniform_value);
        const float delta = -std::log(safe_u) / beta;

        shifts[vertex] = delta;
        distances[vertex] = static_cast<int>(std::floor(delta));
        centers[vertex] = vertex;
        parents[vertex] = -1;
    }
}

namespace spanner
{
    void initialize_miller(int num_vertices,float beta,float* shifts,int* distances,int* centers,int* parents)
    {
        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        initialize_kernel<<<grid_size,block_size>>>(num_vertices,beta,shifts,distances,centers,parents,1234567ULL);

        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Miller initialization launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Miller initialization failed: ") + cudaGetErrorString(error));
        }
    }
}
