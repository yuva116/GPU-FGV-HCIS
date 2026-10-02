#include "gpu/compact_kernels.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace
{
    constexpr int block_size = 256;

    void check_cuda(cudaError_t status,const char* message)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
        }
    }

    __global__ void initialize_edge_count_kernel(int* edge_count)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            *edge_count = 0;
        }
    }

    __global__ void add_tree_edges_kernel(
        int num_vertices,
        const int* centers,
        const int* parents,
        int* edge_count,
        int* spanner_sources,
        int* spanner_destinations,
        int max_edges
    )
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int parent = parents[vertex];

        if(parent < 0 || parent == vertex)
        {
            return;
        }

        if(centers[vertex] < 0 || centers[parent] < 0)
        {
            return;
        }

        const int position = atomicAdd(edge_count,1);

        if(position >= max_edges)
        {
            return;
        }

        spanner_sources[position] = vertex;
        spanner_destinations[position] = parent;
    }

    __global__ void add_inter_cluster_edges_kernel(
        int num_vertices,
        const int* offsets,
        const int* neighbors,
        const int* distances,
        const int* centers,
        int* edge_count,
        int* spanner_sources,
        int* spanner_destinations,
        int max_edges
    )
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int vertex_center = centers[vertex];

        if(vertex_center < 0)
        {
            return;
        }

        const int vertex_distance = distances[vertex];

        int previous_cluster = -1;

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        for(int edge_index = begin;edge_index < end;++edge_index)
        {
            const int neighbor = neighbors[edge_index];

            const int neighbor_center = centers[neighbor];

            if(neighbor_center < 0 || neighbor_center == vertex_center)
            {
                continue;
            }

            if(neighbor_center == previous_cluster)
            {
                continue;
            }

            previous_cluster = neighbor_center;

            const int neighbor_distance = distances[neighbor];

            bool select = false;

            if(vertex_distance > neighbor_distance)
            {
                select = true;
            }
            else if(vertex_distance == neighbor_distance && vertex_center > neighbor_center)
            {
                select = true;
            }

            if(!select)
            {
                continue;
            }

            const int position = atomicAdd(edge_count,1);

            if(position >= max_edges)
            {
                // Capacity is exhausted for the whole kernel launch, not
                // just this vertex; there is nothing further to add.
                return;
            }

            spanner_sources[position] = vertex;
            spanner_destinations[position] = neighbor;

            // BUGFIX: do NOT return here. MPVX-Rule requires a boundary
            // vertex to add one bridge edge for EVERY distinct adjacent
            // cluster it "wins" against (smaller/greater comparison),
            // not just the first one encountered in adjacency order.
            // Returning early here silently drops the apposite-set
            // requirement of Lemma 2.1 for every additional adjacent
            // cluster beyond the first, which is what produced
            // unbounded-stretch edges like (357,34922) in testing.
        }
    }
}

namespace spanner
{
    void build_compact_spanner(
        int num_vertices,
        const int* offsets,
        const int* neighbors,
        const int* distances,
        const int* centers,
        const int* parents,
        int* edge_count,
        int* spanner_sources,
        int* spanner_destinations,
        int max_edges
    )
    {
        if(num_vertices <= 0 || max_edges <= 0)
        {
            return;
        }

        initialize_edge_count_kernel<<<1,1>>>(
            edge_count
        );

        check_cuda(cudaGetLastError(),"Failed to initialize compact edge count");

        check_cuda(cudaDeviceSynchronize(),"Compact edge count initialization failed");

        const int grid_size = (num_vertices + block_size - 1) / block_size;

        add_tree_edges_kernel<<<grid_size,block_size>>>(
            num_vertices,
            centers,
            parents,
            edge_count,
            spanner_sources,
            spanner_destinations,
            max_edges
        );

        check_cuda(cudaGetLastError(),"Failed to launch compact tree edge kernel");

        check_cuda(cudaDeviceSynchronize(),"Compact tree edge construction failed");

        add_inter_cluster_edges_kernel<<<grid_size,block_size>>>(
            num_vertices,
            offsets,
            neighbors,
            distances,
            centers,
            edge_count,
            spanner_sources,
            spanner_destinations,
            max_edges
        );

        check_cuda(cudaGetLastError(),"Failed to launch compact inter-cluster edge kernel");

        check_cuda(cudaDeviceSynchronize(),"Compact inter-cluster edge construction failed");

        int host_edge_count = 0;

        check_cuda(
            cudaMemcpy(
                &host_edge_count,
                edge_count,
                sizeof(int),
                cudaMemcpyDeviceToHost
            ),
            "Failed to read compact edge count"
        );

        if(host_edge_count > max_edges)
        {
            host_edge_count = max_edges;

            check_cuda(
                cudaMemcpy(
                    edge_count,
                    &host_edge_count,
                    sizeof(int),
                    cudaMemcpyHostToDevice
                ),
                "Failed to clamp compact edge count"
            );
        }
    }
}