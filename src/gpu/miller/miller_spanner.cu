#include "gpu/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

// Algorithm 2: UnweightedSpanner builds the forest H from the ESTCluster tree and then
// stitches boundary edges from a vertex to each adjacent higher-ID cluster. A key-based
// sort and unique pass retains one representative edge for each (vertex, cluster) pair.
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

        const int position = atomicAdd(tree_count, 1);
        if(position >= max_edges)
        {
            return;
        }

        sources[position] = vertex;
        destinations[position] = parent;
    }

    __global__ void collect_boundary_candidates_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* centers,std::uint64_t* keys,int* destinations)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= num_vertices)
        {
            return;
        }

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int index = begin; index < end; ++index)
        {
            const int neighbor = neighbors[index];
            const int vertex_cluster = centers[vertex];
            const int neighbor_cluster = centers[neighbor];
            if(neighbor == vertex || vertex_cluster >= neighbor_cluster)
            {
                continue;
            }

            keys[index] = (static_cast<std::uint64_t>(vertex) << 32) |
                          static_cast<std::uint64_t>(static_cast<std::uint32_t>(neighbor_cluster));
            destinations[index] = neighbor;
        }
    }

    __global__ void decode_boundary_kernel(int count,const std::uint64_t* keys,const int* candidate_destinations,int* sources,int* destinations,int tree_count,int* output_count)
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

        const int source = static_cast<int>(key >> 32);
        const int destination = candidate_destinations[index];
        const int position = atomicAdd(output_count, 1);
        sources[tree_count + position] = source;
        destinations[tree_count + position] = destination;
    }
}

namespace spanner
{
    void build_miller_spanner(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,const int* parents,int* edge_count,int* spanner_sources,int* spanner_destinations,int max_edges)
    {
        (void)distances;

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;

        cudaError_t error = cudaMemset(edge_count,0,sizeof(int));
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to reset Miller edge count: ") + cudaGetErrorString(error));
        }

        build_tree_kernel<<<grid_size,block_size>>>(num_vertices,parents,edge_count,spanner_sources,spanner_destinations,max_edges);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Miller tree kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Miller tree construction failed: ") + cudaGetErrorString(error));
        }

        int tree_count = 0;
        error = cudaMemcpy(&tree_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost);
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to read Miller tree count: ") + cudaGetErrorString(error));
        }

        std::uint64_t* candidate_keys = nullptr;
        error = cudaMalloc(&candidate_keys,static_cast<std::size_t>(max_edges) * sizeof(std::uint64_t));
        if(error != cudaSuccess)
        {
            throw std::runtime_error(std::string("Failed to allocate Miller candidate keys: ") + cudaGetErrorString(error));
        }

        error = cudaMemset(candidate_keys,0xFF,static_cast<std::size_t>(max_edges) * sizeof(std::uint64_t));
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            throw std::runtime_error(std::string("Failed to initialize Miller candidate keys: ") + cudaGetErrorString(error));
        }

        int* candidate_destinations = nullptr;
        error = cudaMalloc(&candidate_destinations,static_cast<std::size_t>(max_edges) * sizeof(int));
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            throw std::runtime_error(std::string("Failed to allocate Miller candidate destinations: ") + cudaGetErrorString(error));
        }

        error = cudaMemset(candidate_destinations,0xFF,static_cast<std::size_t>(max_edges) * sizeof(int));
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            throw std::runtime_error(std::string("Failed to initialize Miller candidate destinations: ") + cudaGetErrorString(error));
        }

        collect_boundary_candidates_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,centers,candidate_keys,candidate_destinations);
        error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            throw std::runtime_error(std::string("Miller boundary kernel launch failed: ") + cudaGetErrorString(error));
        }

        error = cudaDeviceSynchronize();
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            throw std::runtime_error(std::string("Miller boundary collection failed: ") + cudaGetErrorString(error));
        }

        thrust::device_ptr<std::uint64_t> key_begin(candidate_keys);
        thrust::device_ptr<std::uint64_t> key_end(candidate_keys + max_edges);
        thrust::device_ptr<int> destination_begin(candidate_destinations);
        thrust::sort_by_key(key_begin, key_end, destination_begin);
        auto unique_result = thrust::unique_by_key(key_begin, key_end, destination_begin);

        const int unique_count = static_cast<int>(unique_result.first - key_begin);
        int boundary_count = 0;
        int* d_boundary_count = nullptr;
        error = cudaMalloc(&d_boundary_count,sizeof(int));
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            throw std::runtime_error(std::string("Failed to allocate Miller boundary count: ") + cudaGetErrorString(error));
        }

        error = cudaMemset(d_boundary_count,0,sizeof(int));
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(d_boundary_count);
            throw std::runtime_error(std::string("Failed to reset Miller boundary count: ") + cudaGetErrorString(error));
        }

        int valid_count = unique_count;
        while(valid_count > 0)
        {
            std::uint64_t key = 0;
            error = cudaMemcpy(&key,candidate_keys + valid_count - 1,sizeof(std::uint64_t),cudaMemcpyDeviceToHost);
            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(d_boundary_count);
                throw std::runtime_error(std::string("Failed to inspect Miller boundary keys: ") + cudaGetErrorString(error));
            }

            if(key != UINT64_MAX)
            {
                break;
            }
            --valid_count;
        }

        if(valid_count > 0)
        {
            const int decode_grid = (valid_count + block_size - 1) / block_size;
            decode_boundary_kernel<<<decode_grid,block_size>>>(valid_count,candidate_keys,candidate_destinations,spanner_sources,spanner_destinations,tree_count,d_boundary_count);
            error = cudaGetLastError();
            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(d_boundary_count);
                throw std::runtime_error(std::string("Miller decode kernel launch failed: ") + cudaGetErrorString(error));
            }

            error = cudaDeviceSynchronize();
            if(error != cudaSuccess)
            {
                cudaFree(candidate_keys);
                cudaFree(candidate_destinations);
                cudaFree(d_boundary_count);
                throw std::runtime_error(std::string("Miller decode failed: ") + cudaGetErrorString(error));
            }
        }

        error = cudaMemcpy(&boundary_count,d_boundary_count,sizeof(int),cudaMemcpyDeviceToHost);
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(d_boundary_count);
            throw std::runtime_error(std::string("Failed to read Miller boundary count: ") + cudaGetErrorString(error));
        }

        const int total_count = tree_count + boundary_count;
        if(total_count > max_edges)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(d_boundary_count);
            throw std::runtime_error("Miller spanner exceeded allocated edge capacity");
        }

        error = cudaMemcpy(edge_count,&total_count,sizeof(int),cudaMemcpyHostToDevice);
        if(error != cudaSuccess)
        {
            cudaFree(candidate_keys);
            cudaFree(candidate_destinations);
            cudaFree(d_boundary_count);
            throw std::runtime_error(std::string("Failed to update Miller edge count: ") + cudaGetErrorString(error));
        }

        cudaFree(candidate_keys);
        cudaFree(candidate_destinations);
        cudaFree(d_boundary_count);
    }
}
