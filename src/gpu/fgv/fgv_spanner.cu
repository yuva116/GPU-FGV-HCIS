#include "gpu/kernels.hpp"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace
{
    __global__ void build_tree_kernel(int num_vertices,const int* parents,int* tree_count,int* sources,int* destinations,int max_edges)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int parent = parents[vertex];

        if(parent < 0)
        {
            return;
        }

        const int position = atomicAdd(tree_count,1);

        if(position >= max_edges)
        {
            return;
        }

        sources[position] = vertex;
        destinations[position] = parent;
    }

    __global__ void collect_f_candidates_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,std::uint64_t* keys,int* destinations)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int vertex_distance = distances[vertex];
        const int vertex_center = centers[vertex];

        if(vertex_distance < 0 || vertex_center < 0)
        {
            return;
        }

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin;index < end;++index)
        {
            const int neighbor = neighbors[index];
            const int neighbor_distance = distances[neighbor];
            const int neighbor_center = centers[neighbor];

            if(neighbor_distance < 0 || neighbor_center < 0)
            {
                continue;
            }

            const int candidate_distance = neighbor_distance + 1;

            const bool same_level = candidate_distance == vertex_distance;
            const bool lower_level = candidate_distance == vertex_distance + 1 && neighbor_center < vertex_center;

            if(!same_level && !lower_level)
            {
                continue;
            }

            const std::uint64_t key = static_cast<std::uint64_t>(vertex) * static_cast<std::uint64_t>(num_vertices) + static_cast<std::uint64_t>(neighbor_center);

            keys[index] = key;
            destinations[index] = neighbor;
        }
    }

    __global__ void decode_f_candidates_kernel(int count,const std::uint64_t* keys,const int* destinations,int num_vertices,int tree_count,int* sources,int* output_destinations,int* f_count)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;

        if(index >= count)
        {
            return;
        }

        const std::uint64_t key = keys[index];

        if(key == UINT64_MAX)
        {
            return;
        }

        const int source = static_cast<int>(key / static_cast<std::uint64_t>(num_vertices));
        const int position = atomicAdd(f_count,1);

        sources[tree_count + position] = source;
        output_destinations[tree_count + position] = destinations[index];
    }
}

namespace spanner
{
    void build_fgv_spanner(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,const int* parents,int* edge_count,int* spanner_sources,int* spanner_destinations,int max_edges)
    {
        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        cudaError_t error = cudaMemset(edge_count,0,sizeof(int));

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to reset FGV edge count: ") + cudaGetErrorString(error));
        }

        build_tree_kernel<<<grid_size,block_size>>>(num_vertices,parents,edge_count,spanner_sources,spanner_destinations,max_edges);

        error = cudaGetLastError();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV tree kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("FGV tree construction failed: ") + cudaGetErrorString(error));
        }

        int tree_count = 0;

        error = cudaMemcpy(&tree_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost);

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to read FGV tree count: ") + cudaGetErrorString(error));
        }

        std::uint64_t* candidate_keys = nullptr;
        int* candidate_destinations = nullptr;
        int* f_count = nullptr;

        error = cudaMalloc(&candidate_keys,static_cast<std::size_t>(max_edges) * sizeof(std::uint64_t));

        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to allocate FGV candidate keys: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&candidate_destinations,static_cast<std::size_t>(max_edges) * sizeof(int));

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            throw std::runtime_error(std::string("Failed to allocate FGV candidate destinations: ") + cudaGetErrorString(error));
        }

        error = cudaMalloc(&f_count,sizeof(int));

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            throw std::runtime_error(std::string("Failed to allocate FGV F count: ") + cudaGetErrorString(error));
        }

        error = cudaMemset(candidate_keys,0xFF,static_cast<std::size_t>(max_edges) * sizeof(std::uint64_t));

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("Failed to initialize FGV candidate keys: ") + cudaGetErrorString(error));
        }

        error = cudaMemset(f_count,0,sizeof(int));

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("Failed to reset FGV F count: ") + cudaGetErrorString(error));
        }

        collect_f_candidates_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,distances,centers,candidate_keys,candidate_destinations);

        error = cudaGetLastError();

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("FGV F-candidate kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("FGV F-candidate construction failed: ") + cudaGetErrorString(error));
        }

        thrust::device_ptr<std::uint64_t> key_begin(candidate_keys);
        thrust::device_ptr<std::uint64_t> key_end(candidate_keys + max_edges);
        thrust::device_ptr<int> destination_begin(candidate_destinations);

        thrust::sort_by_key(key_begin,key_end,destination_begin);

        auto unique_result = thrust::unique_by_key(key_begin,key_end,destination_begin);

        const int unique_count = static_cast<int>(unique_result.first - key_begin);

        int valid_count = unique_count;

        while(valid_count > 0)
        {
            std::uint64_t key = 0;

            error = cudaMemcpy(&key,candidate_keys + valid_count - 1,sizeof(std::uint64_t),cudaMemcpyDeviceToHost);

            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(f_count);
                throw std::runtime_error(std::string("Failed to inspect FGV candidate keys: ") + cudaGetErrorString(error));
            }

            if(key != UINT64_MAX)
            {
                break;
            }

            --valid_count;
        }

        if(valid_count > 0)
        {
            const int f_grid_size = (valid_count + block_size - 1) / block_size;

            decode_f_candidates_kernel<<<f_grid_size,block_size>>>(valid_count,candidate_keys,candidate_destinations,num_vertices,tree_count,spanner_sources,spanner_destinations,f_count);

            error = cudaGetLastError();

            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(f_count);
                throw std::runtime_error(std::string("FGV F-output kernel launch failed: ") + cudaGetErrorString(error));
            }

            error = cudaDeviceSynchronize();

            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(f_count);
                throw std::runtime_error(std::string("FGV F-output construction failed: ") + cudaGetErrorString(error));
            }
        }

        int host_f_count = 0;

        error = cudaMemcpy(&host_f_count,f_count,sizeof(int),cudaMemcpyDeviceToHost);

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("Failed to read FGV F count: ") + cudaGetErrorString(error));
        }

        const int total_count = tree_count + host_f_count;

        if(total_count > max_edges)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error("FGV spanner exceeded allocated edge capacity");
        }

        error = cudaMemcpy(edge_count,&total_count,sizeof(int),cudaMemcpyHostToDevice);

        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(f_count);
            throw std::runtime_error(std::string("Failed to update FGV edge count: ") + cudaGetErrorString(error));
        }

        cudaFree(candidate_keys);
        cudaFree(candidate_destinations);
        cudaFree(f_count);
    }
}
