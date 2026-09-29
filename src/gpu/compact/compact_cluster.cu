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

    __global__ void initialize_clusters_kernel(
        int num_vertices,
        const int* is_center,
        int* distances,
        int* centers,
        int* parents
    )
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        distances[vertex] = is_center[vertex] != 0 ? 0 : -1;

        centers[vertex] = is_center[vertex] != 0 ? vertex : -1;

        parents[vertex] = is_center[vertex] != 0 ? vertex : -1;
    }

    __global__ void relax_clusters_kernel(
        int num_vertices,
        int radius,
        const int* offsets,
        const int* neighbors,
        int* distances,
        int* centers,
        int* parents,
        int* changed
    )
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        const int current_distance = distances[vertex];

        // BUGFIX: previously this also returned early whenever
        // current_distance < 0 (i.e. the vertex has not yet been
        // reached by any center). That made it impossible for an
        // unclustered vertex to ever adopt a neighbor's label here,
        // silently pushing all real cluster growth onto the one-shot,
        // 1-hop-only repair_unclustered_kernel below. Only a vertex
        // that is already a center (distance == 0) has nothing left to
        // improve; every other vertex - including unclustered ones -
        // must keep looking at its neighbors each round.
        if(current_distance == 0)
        {
            return;
        }

        const int begin = offsets[vertex];
        const int end = offsets[vertex + 1];

        int best_center = centers[vertex];
        int best_parent = parents[vertex];
        int best_distance = current_distance;

        for(int edge_index = begin;edge_index < end;++edge_index)
        {
            const int neighbor = neighbors[edge_index];

            const int neighbor_distance = distances[neighbor];

            if(neighbor_distance < 0 || neighbor_distance >= radius)
            {
                continue;
            }

            const int candidate_distance = neighbor_distance + 1;

            if(best_distance < 0 || candidate_distance < best_distance)
            {
                best_distance = candidate_distance;
                best_center = centers[neighbor];
                best_parent = neighbor;
            }
            else if(candidate_distance == best_distance)
            {
                const int candidate_center = centers[neighbor];

                if(candidate_center >= 0 && (best_center < 0 || candidate_center < best_center))
                {
                    best_center = candidate_center;
                    best_parent = neighbor;
                }
            }
        }

        if(best_distance != distances[vertex] || best_center != centers[vertex] || best_parent != parents[vertex])
        {
            distances[vertex] = best_distance;
            centers[vertex] = best_center;
            parents[vertex] = best_parent;
            *changed = 1;
        }
    }

    __global__ void repair_unclustered_kernel(
        int num_vertices,
        const int* offsets,
        const int* neighbors,
        int* distances,
        int* centers,
        int* parents
    )
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices || centers[vertex] >= 0)
        {
            return;
        }

        int best_center = -1;
        int best_parent = -1;

        for(int edge_index = offsets[vertex];edge_index < offsets[vertex + 1];++edge_index)
        {
            const int neighbor = neighbors[edge_index];

            if(centers[neighbor] < 0)
            {
                continue;
            }

            if(best_center < 0 || centers[neighbor] < best_center)
            {
                best_center = centers[neighbor];
                best_parent = neighbor;
            }
        }

        if(best_center >= 0)
        {
            centers[vertex] = best_center;
            parents[vertex] = best_parent;
            distances[vertex] = distances[best_parent] + 1;
        }
    }
}

namespace spanner
{
    void grow_compact_clusters(
        int num_vertices,
        int radius,
        const int* offsets,
        const int* neighbors,
        const int* is_center,
        int* distances,
        int* centers,
        int* parents
    )
    {
        if(num_vertices <= 0)
        {
            return;
        }

        if(radius < 1)
        {
            check_cuda(
                cudaMemset(distances,-1,static_cast<std::size_t>(num_vertices) * sizeof(int)),
                "Failed to initialize compact distances"
            );

            check_cuda(
                cudaMemset(centers,-1,static_cast<std::size_t>(num_vertices) * sizeof(int)),
                "Failed to initialize compact centers"
            );

            check_cuda(
                cudaMemset(parents,-1,static_cast<std::size_t>(num_vertices) * sizeof(int)),
                "Failed to initialize compact parents"
            );

            return;
        }

        int* changed = nullptr;

        try
        {
            check_cuda(
                cudaMalloc(
                    reinterpret_cast<void**>(&changed),
                    sizeof(int)
                ),
                "Failed to allocate cluster change flag"
            );

            const int grid_size = (num_vertices + block_size - 1) / block_size;

            initialize_clusters_kernel<<<grid_size,block_size>>>(
                num_vertices,
                is_center,
                distances,
                centers,
                parents
            );

            check_cuda(cudaGetLastError(),"Failed to launch cluster initialization");

            check_cuda(cudaDeviceSynchronize(),"Cluster initialization failed");

            for(int round = 0;round < radius;++round)
            {
                check_cuda(
                    cudaMemset(changed,0,sizeof(int)),
                    "Failed to reset cluster change flag"
                );

                relax_clusters_kernel<<<grid_size,block_size>>>(
                    num_vertices,
                    radius,
                    offsets,
                    neighbors,
                    distances,
                    centers,
                    parents,
                    changed
                );

                check_cuda(cudaGetLastError(),"Failed to launch cluster relaxation");

                check_cuda(cudaDeviceSynchronize(),"Cluster relaxation failed");

                int host_changed = 0;

                check_cuda(
                    cudaMemcpy(
                        &host_changed,
                        changed,
                        sizeof(int),
                        cudaMemcpyDeviceToHost
                    ),
                    "Failed to read cluster change flag"
                );

                if(host_changed == 0)
                {
                    break;
                }
            }

            repair_unclustered_kernel<<<grid_size,block_size>>>(
                num_vertices,
                offsets,
                neighbors,
                distances,
                centers,
                parents
            );

            check_cuda(cudaGetLastError(),"Failed to launch cluster repair");

            check_cuda(cudaDeviceSynchronize(),"Cluster repair failed");
        }
        catch(...)
        {
            cudaFree(changed);
            throw;
        }

        cudaFree(changed);
    }
}