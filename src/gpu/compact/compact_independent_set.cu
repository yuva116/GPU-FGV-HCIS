#include "gpu/compact_kernels.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace
{
    void check_cuda(cudaError_t status,const char* message)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
        }
    }

    __global__ void initialize_active_kernel(int num_vertices,int* active,int* active_neighbors)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        active[vertex] = 1;
        active_neighbors[vertex] = 0;
    }

    __global__ void compute_active_neighbors_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* active,int* active_neighbors)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            active_neighbors[vertex] = 0;
            return;
        }

        int count = 0;

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];

            if(active[neighbor] != 0)
            {
                ++count;
            }
        }

        active_neighbors[vertex] = count;
    }

    __global__ void initialize_frontier_kernel(int num_vertices,const int* is_center,int* frontier,int* visited)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(is_center[vertex] != 0)
        {
            frontier[vertex] = 1;
            visited[vertex] = 1;
        }
        else
        {
            frontier[vertex] = 0;
            visited[vertex] = 0;
        }
    }

    __global__ void expand_frontier_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* frontier,int* next_frontier,int* visited)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(frontier[vertex] == 0)
        {
            return;
        }

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];

            if(atomicCAS(&visited[neighbor],0,1) == 0)
            {
                next_frontier[neighbor] = 1;
            }
        }
    }

    __global__ void deactivate_visited_kernel(int num_vertices,const int* visited,int* active)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(visited[vertex] != 0)
        {
            active[vertex] = 0;
        }
    }

    __global__ void clear_center_flags_kernel(int num_vertices,int* is_center)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex < num_vertices)
        {
            is_center[vertex] = 0;
        }
    }
}

namespace spanner
{
    void initialize_compact_active_set(int num_vertices,int* active,int* active_neighbors)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        initialize_active_kernel<<<grid_size,block_size>>>(num_vertices,active,active_neighbors);

        check_cuda(cudaGetLastError(),"Failed to launch compact active-set initialization");
        check_cuda(cudaDeviceSynchronize(),"Compact active-set initialization failed");
    }

    void update_compact_active_neighbors(int num_vertices,const int* offsets,const int* neighbors,const int* active,int* active_neighbors)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        compute_active_neighbors_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,active,active_neighbors);

        check_cuda(cudaGetLastError(),"Failed to launch compact active-neighbor update");
        check_cuda(cudaDeviceSynchronize(),"Compact active-neighbor update failed");
    }

    void deactivate_compact_radius(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* is_center,int* active)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        if(radius < 0)
        {
            throw std::invalid_argument("Compact radius cannot be negative");
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;
        const std::size_t bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);

        int* frontier = nullptr;
        int* next_frontier = nullptr;
        int* visited = nullptr;

        try
        {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&frontier),bytes),"Failed to allocate compact frontier");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next_frontier),bytes),"Failed to allocate compact next frontier");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&visited),bytes),"Failed to allocate compact visited");

            check_cuda(cudaMemset(frontier,0,bytes),"Failed to initialize compact frontier");
            check_cuda(cudaMemset(next_frontier,0,bytes),"Failed to initialize compact next frontier");
            check_cuda(cudaMemset(visited,0,bytes),"Failed to initialize compact visited");

            initialize_frontier_kernel<<<grid_size,block_size>>>(num_vertices,is_center,frontier,visited);

            check_cuda(cudaGetLastError(),"Failed to initialize compact center frontier");
            check_cuda(cudaDeviceSynchronize(),"Compact center frontier initialization failed");

            for(int level = 0;level < radius;++level)
            {
                check_cuda(cudaMemset(next_frontier,0,bytes),"Failed to clear compact frontier");

                expand_frontier_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,frontier,next_frontier,visited);

                check_cuda(cudaGetLastError(),"Failed to expand compact frontier");
                check_cuda(cudaDeviceSynchronize(),"Compact frontier expansion failed");

                std::swap(frontier,next_frontier);
            }

            deactivate_visited_kernel<<<grid_size,block_size>>>(num_vertices,visited,active);

            check_cuda(cudaGetLastError(),"Failed to deactivate compact vertices");
            check_cuda(cudaDeviceSynchronize(),"Compact vertex deactivation failed");
        }
        catch(...)
        {
            cudaFree(frontier);
            cudaFree(next_frontier);
            cudaFree(visited);
            throw;
        }

        cudaFree(frontier);
        cudaFree(next_frontier);
        cudaFree(visited);
    }

    void clear_compact_center_flags(int num_vertices,int* is_center)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        clear_center_flags_kernel<<<grid_size,block_size>>>(num_vertices,is_center);

        check_cuda(cudaGetLastError(),"Failed to clear compact center flags");
        check_cuda(cudaDeviceSynchronize(),"Compact center-flag clearing failed");
    }
}