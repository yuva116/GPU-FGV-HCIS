#include "gpu/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

// Algorithm 1: ESTCluster uses a virtual super-source connected to each vertex u with
// edge weight delta_u. We relax labels in increasing distance, packing each label as
// (distance << 32) | center_id so the shortest path tree is discovered with a single
// 64-bit atomicMin on the GPU.
namespace
{
    __device__ __forceinline__ std::uint64_t pack_label(int distance, int center)
    {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(distance)) << 32) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(center));
    }

    __global__ void initialize_labels_kernel(int num_vertices,const float* shifts,std::uint64_t* labels)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int distance = static_cast<int>(std::floor(shifts[vertex]));
        labels[vertex] = pack_label(distance, vertex);
    }

    __global__ void relax_cluster_kernel(int num_vertices,const int* offsets,const int* neighbors,const std::uint64_t* current_labels,std::uint64_t* next_labels,int* changed)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const std::uint64_t current = current_labels[vertex];
        const int current_distance = static_cast<int>(current >> 32);
        const int current_center = static_cast<int>(current & 0xffffffffULL);

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin; index < end; ++index)
        {
            const int neighbor = neighbors[index];
            const std::uint64_t candidate = pack_label(current_distance + 1, current_center);
            const std::uint64_t previous = atomicMin(reinterpret_cast<unsigned long long*>(&next_labels[neighbor]), static_cast<unsigned long long>(candidate));

            if(previous > candidate)
            {
                atomicExch(changed, 1);
            }
        }
    }

    __global__ void unpack_labels_kernel(int num_vertices,const std::uint64_t* labels,int* distances,int* centers)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const std::uint64_t label = labels[vertex];
        distances[vertex] = static_cast<int>(label >> 32);
        centers[vertex] = static_cast<int>(label & 0xffffffffULL);
    }

    __global__ void parent_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,int* parents)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(centers[vertex] == vertex)
        {
            parents[vertex] = -1;
            return;
        }

        int parent = -1;
        const int distance = distances[vertex];

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin; index < end; ++index)
        {
            const int neighbor = neighbors[index];
            if(centers[neighbor] != centers[vertex])
            {
                continue;
            }
            if(distances[neighbor] != distance - 1)
            {
                continue;
            }
            if(parent == -1 || neighbor < parent)
            {
                parent = neighbor;
            }
        }

        parents[vertex] = parent;
    }
}

namespace spanner
{
    void build_miller_clusters(int num_vertices,int /*unused*/,const int* offsets,const int* neighbors,const float* shifts,int* distances,int* centers,int* parents)
    {
        std::uint64_t* labels = nullptr;
        std::uint64_t* next_labels = nullptr;
        int* changed = nullptr;

        const std::size_t bytes = static_cast<std::size_t>(num_vertices) * sizeof(std::uint64_t);

        cudaError_t error = cudaMalloc(&labels,bytes);
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to allocate Miller labels: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&next_labels,bytes);
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            throw std::runtime_error(std::string("Failed to allocate next Miller labels: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&changed,sizeof(int));
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            throw std::runtime_error(std::string("Failed to allocate Miller changed flag: ") + cudaGetErrorString(error));
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        initialize_labels_kernel<<<grid_size,block_size>>>(num_vertices,shifts,labels);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller label initialization launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller label initialization failed: ") + cudaGetErrorString(error));
        }

        while(true)
        {
            error = cudaMemcpy(next_labels,labels,bytes,cudaMemcpyDeviceToDevice);
            if(error != cudaSuccess)
            {
                cudaFree(labels);
                cudaFree(next_labels);
                cudaFree(changed);
                throw std::runtime_error(std::string("Failed to copy current Miller labels: ") + cudaGetErrorString(error));
            }

            error = cudaMemset(changed,0,sizeof(int));
            if(error != cudaSuccess)
            {
                cudaFree(labels);
                cudaFree(next_labels);
                cudaFree(changed);
                throw std::runtime_error(std::string("Failed to reset Miller changed flag: ") + cudaGetErrorString(error));
            }

            relax_cluster_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,labels,next_labels,changed);
            error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                cudaFree(labels);
                cudaFree(next_labels);
                cudaFree(changed);
                throw std::runtime_error(std::string("Miller relaxation launch failed: ") + cudaGetErrorString(error));
            }

            error = cudaDeviceSynchronize();
            if(error != cudaSuccess)
            {
                cudaFree(labels);
                cudaFree(next_labels);
                cudaFree(changed);
                throw std::runtime_error(std::string("Miller relaxation failed: ") + cudaGetErrorString(error));
            }

            int host_changed = 0;
            error = cudaMemcpy(&host_changed,changed,sizeof(int),cudaMemcpyDeviceToHost);
            if(error != cudaSuccess)
            {
                cudaFree(labels);
                cudaFree(next_labels);
                cudaFree(changed);
                throw std::runtime_error(std::string("Failed to read Miller changed flag: ") + cudaGetErrorString(error));
            }

            std::uint64_t* label_temp = labels;
            labels = next_labels;
            next_labels = label_temp;

            if(host_changed == 0)
            {
                break;
            }
        }

        unpack_labels_kernel<<<grid_size,block_size>>>(num_vertices,labels,distances,centers);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller unpack launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller unpack failed: ") + cudaGetErrorString(error));
        }

        parent_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,distances,centers,parents);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller parent kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            cudaFree(labels);
            cudaFree(next_labels);
            cudaFree(changed);
            throw std::runtime_error(std::string("Miller parent construction failed: ") + cudaGetErrorString(error));
        }

        cudaFree(labels);
        cudaFree(next_labels);
        cudaFree(changed);
    }
}
