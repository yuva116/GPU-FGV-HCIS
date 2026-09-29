#include "gpu/kernels.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{
    __device__ bool better_label(int candidate_distance,int candidate_center)
    {
        return candidate_distance >= 0 && candidate_center >= 0;
    }

    __global__ void relax_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* previous_distances,const int* previous_centers,int* next_distances,int* next_centers,int* changed)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        int best_distance = previous_distances[vertex];
        int best_center = previous_centers[vertex];

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin;index < end;++index)
        {
            const int neighbor = neighbors[index];
            const int neighbor_distance = previous_distances[neighbor];
            const int neighbor_center = previous_centers[neighbor];

            if(!better_label(neighbor_distance,neighbor_center))
            {
                continue;
            }

            const int candidate_distance = neighbor_distance + 1;
            const int candidate_center = neighbor_center;

            if(candidate_distance < best_distance)
            {
                best_distance = candidate_distance;
                best_center = candidate_center;
            }
            else if(candidate_distance == best_distance && candidate_center < best_center)
            {
                best_center = candidate_center;
            }
        }

        next_distances[vertex] = best_distance;
        next_centers[vertex] = best_center;

        if(best_distance != previous_distances[vertex] || best_center != previous_centers[vertex])
        {
            atomicExch(changed,1);
        }
    }

    __global__ void parent_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,int* parents)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int distance = distances[vertex];
        const int center = centers[vertex];

        if(distance <= 0 || center < 0)
        {
            parents[vertex] = -1;
            return;
        }

        int parent = -1;

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin;index < end;++index)
        {
            const int neighbor = neighbors[index];

            if(centers[neighbor] != center)
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
    void build_fgv_clusters(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* shifts,int* distances,int* centers,int* parents)
    {
        (void)shifts;

        int* previous_distances = nullptr;
        int* next_distances = nullptr;
        int* previous_centers = nullptr;
        int* next_centers = nullptr;
        int* changed = nullptr;

        const std::size_t bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);

        cudaError_t error = cudaMalloc(&previous_distances,bytes);

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to allocate previous distances: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&next_distances,bytes);

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            throw std::runtime_error(std::string("Failed to allocate next distances: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&previous_centers,bytes);

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            cudaFree(next_distances);
            throw std::runtime_error(std::string("Failed to allocate previous centers: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&next_centers,bytes);

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            cudaFree(next_distances);
            cudaFree(previous_centers);
            throw std::runtime_error(std::string("Failed to allocate next centers: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&changed,sizeof(int));

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            cudaFree(next_distances);
            cudaFree(previous_centers);
            cudaFree(next_centers);
            throw std::runtime_error(std::string("Failed to allocate changed flag: ") + cudaGetErrorString(error));
        }

        error = cudaMemcpy(previous_distances,distances,bytes,cudaMemcpyDeviceToDevice);

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            cudaFree(next_distances);
            cudaFree(previous_centers);
            cudaFree(next_centers);
            cudaFree(changed);
            throw std::runtime_error(std::string("Failed to initialize FGV distances: ") + cudaGetErrorString(error));
        }

        error = cudaMemcpy(previous_centers,centers,bytes,cudaMemcpyDeviceToDevice);

        if(error != cudaSuccess)
        {
            cudaFree(previous_distances);
            cudaFree(next_distances);
            cudaFree(previous_centers);
            cudaFree(next_centers);
            cudaFree(changed);
            throw std::runtime_error(std::string("Failed to initialize FGV centers: ") + cudaGetErrorString(error));
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        for(int round = 0;round <= radius;++round)
        {
            error = cudaMemset(changed,0,sizeof(int));

            if(error != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to reset FGV changed flag: ") + cudaGetErrorString(error));
            }

            relax_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,previous_distances,previous_centers,next_distances,next_centers,changed);

            error = cudaGetLastError();

            if(error != cudaSuccess)
            {
                throw std::runtime_error(std::string("FGV relaxation launch failed: ") + cudaGetErrorString(error));
            }

            error = cudaDeviceSynchronize();

            if(error != cudaSuccess)
            {
                throw std::runtime_error(std::string("FGV relaxation failed: ") + cudaGetErrorString(error));
            }

            int host_changed = 0;

            error = cudaMemcpy(&host_changed,changed,sizeof(int),cudaMemcpyDeviceToHost);

            if(error != cudaSuccess)
            {
                throw std::runtime_error(std::string("Failed to read FGV changed flag: ") + cudaGetErrorString(error));
            }

            int* distance_temp = previous_distances;
            previous_distances = next_distances;
            next_distances = distance_temp;

            int* center_temp = previous_centers;
            previous_centers = next_centers;
            next_centers = center_temp;

            if(host_changed == 0)
            {
                break;
            }
        }

        error = cudaMemcpy(distances,previous_distances,bytes,cudaMemcpyDeviceToDevice);

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to copy final FGV distances: ") + cudaGetErrorString(error));
        }

        error = cudaMemcpy(centers,previous_centers,bytes,cudaMemcpyDeviceToDevice);

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to copy final FGV centers: ") + cudaGetErrorString(error));
        }

        parent_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,distances,centers,parents);

        error = cudaGetLastError();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV parent kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV parent construction failed: ") + cudaGetErrorString(error));
        }

        cudaFree(previous_distances);
        cudaFree(next_distances);
        cudaFree(previous_centers);
        cudaFree(next_centers);
        cudaFree(changed);
    }
}