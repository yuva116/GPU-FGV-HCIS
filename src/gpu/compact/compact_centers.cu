#include "gpu/compact_kernels.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{
    void check_cuda(cudaError_t status,const char* message)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
        }
    }

    __global__ void initialize_coverage_kernel(int num_vertices,const int* active,const int* active_neighbors,int* coverage)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            coverage[vertex] = 0;
            return;
        }

        coverage[vertex] = active_neighbors[vertex];
    }

    __global__ void compute_next_coverage_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* active,const int* previous_coverage,int* next_coverage)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            next_coverage[vertex] = 0;
            return;
        }

        int value = 0;

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];

            if(active[neighbor] != 0)
            {
                value += previous_coverage[neighbor];
            }
        }

        next_coverage[vertex] = value;
    }

    // NOTE (bugfix): the max-coverage "champion" must be propagated over
    // the full r-hop neighborhood, not re-derived from a 1-hop scan. We
    // track both the champion's coverage VALUE and its VERTEX ID (owner)
    // together, because the paper's tie-break rule ("equal coverage,
    // smaller id wins") is only well-defined if you know *which* vertex
    // achieved the maximum, not just the maximum value.
    __global__ void initialize_max_kernel(int num_vertices,const int* active,const int* coverage,int* maximum,int* owner)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            maximum[vertex] = 0;
            owner[vertex] = -1;
            return;
        }

        maximum[vertex] = coverage[vertex];
        owner[vertex] = vertex;
    }

    __global__ void compute_max_kernel(int num_vertices,const int* offsets,const int* neighbors,const int* active,const int* previous_maximum,const int* previous_owner,int* next_maximum,int* next_owner)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            next_maximum[vertex] = 0;
            next_owner[vertex] = -1;
            return;
        }

        int maximum = previous_maximum[vertex];
        int maximum_owner = previous_owner[vertex];

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];

            if(active[neighbor] == 0)
            {
                continue;
            }

            const int neighbor_value = previous_maximum[neighbor];
            const int neighbor_owner = previous_owner[neighbor];

            if(neighbor_owner < 0)
            {
                continue;
            }

            if(neighbor_value > maximum || (neighbor_value == maximum && neighbor_owner < maximum_owner))
            {
                maximum = neighbor_value;
                maximum_owner = neighbor_owner;
            }
        }

        next_maximum[vertex] = maximum;
        next_owner[vertex] = maximum_owner;
    }

    // A vertex is a local coverage maximum over its FULL r-hop
    // neighborhood iff, after `radius` rounds of the propagation above,
    // it is still its own champion (owner[vertex] == vertex).
    __global__ void compute_max_flag_kernel(int num_vertices,const int* active,const int* coverage,const int* maximum,const int* owner,int* max_flag)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] == 0)
        {
            max_flag[vertex] = 0;
            return;
        }

        max_flag[vertex] = (owner[vertex] == vertex && maximum[vertex] == coverage[vertex]) ? 1 : 0;
    }

    __global__ void select_centers_kernel(int num_vertices,const int* active,const int* max_flag,int* is_center,int* newly_selected)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        newly_selected[vertex] = 0;

        if(active[vertex] != 0 && max_flag[vertex] != 0)
        {
            is_center[vertex] = 1;
            newly_selected[vertex] = 1;
        }
    }

    // Modified HCIS-r (Sec. 4.1): a rejected batch is dropped from consideration.
    __global__ void reject_batch_kernel(int num_vertices,const int* batch,int* is_center,int* active)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex < num_vertices && batch[vertex] != 0)
        {
            is_center[vertex] = 0;
            active[vertex] = 0;
        }
    }

    __global__ void count_active_kernel(int num_vertices,const int* active,int* count)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex] != 0)
        {
            atomicAdd(count,1);
        }
    }

    __global__ void count_selected_kernel(int num_vertices,const int* selected,int* count)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(selected[vertex] != 0)
        {
            atomicAdd(count,1);
        }
    }
}

namespace spanner
{
    void run_hcis_r(int num_vertices,int radius,const int* offsets,const int* neighbors,int* is_center,int max_iterations,long long max_boundary_edges)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        if(radius < 1)
        {
            throw std::invalid_argument("HCIS-r requires radius >= 1");
        }

        constexpr int block_size = 256;
        const int grid_size = (num_vertices + block_size - 1) / block_size;
        const std::size_t bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);

        int* active = nullptr;
        int* active_neighbors = nullptr;
        int* coverage = nullptr;
        int* next_coverage = nullptr;
        int* maximum = nullptr;
        int* next_maximum = nullptr;
        int* owner = nullptr;
        int* next_owner = nullptr;
        int* max_flag = nullptr;
        int* newly_selected = nullptr;
        int* active_count = nullptr;
        int* selected_count = nullptr;

        try
        {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&active),bytes),"Failed to allocate HCIS active");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&active_neighbors),bytes),"Failed to allocate HCIS active neighbors");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&coverage),bytes),"Failed to allocate HCIS coverage");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next_coverage),bytes),"Failed to allocate HCIS next coverage");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&maximum),bytes),"Failed to allocate HCIS maximum");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next_maximum),bytes),"Failed to allocate HCIS next maximum");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&owner),bytes),"Failed to allocate HCIS maximum owner");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&next_owner),bytes),"Failed to allocate HCIS next maximum owner");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&max_flag),bytes),"Failed to allocate HCIS max flags");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&newly_selected),bytes),"Failed to allocate HCIS selected centers");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&active_count),sizeof(int)),"Failed to allocate HCIS active count");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&selected_count),sizeof(int)),"Failed to allocate HCIS selected count");

            check_cuda(cudaMemset(is_center,0,bytes),"Failed to clear HCIS centers");

            initialize_compact_active_set(num_vertices,active,active_neighbors);

            update_compact_active_neighbors(num_vertices,offsets,neighbors,active,active_neighbors);

            int round = 0;

            while(true)
            {
                ++round;

                // Modified HCIS-r (Sec. 4.1): at most alpha phases.
                if(max_iterations > 0 && round > max_iterations)
                {
                    break;
                }

                check_cuda(cudaMemset(active_count,0,sizeof(int)),"Failed to reset HCIS active count");

                count_active_kernel<<<grid_size,block_size>>>(num_vertices,active,active_count);

                check_cuda(cudaGetLastError(),"Failed to count active HCIS vertices");
                check_cuda(cudaDeviceSynchronize(),"HCIS active count failed");

                int host_active_count = 0;

                check_cuda(cudaMemcpy(&host_active_count,active_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy HCIS active count");

                if(host_active_count == 0)
                {
                    std::cout << "HCIS round " << round << ": active=0 selected=0\n";
                    break;
                }

                initialize_coverage_kernel<<<grid_size,block_size>>>(num_vertices,active,active_neighbors,coverage);

                check_cuda(cudaGetLastError(),"Failed to initialize HCIS coverage");
                check_cuda(cudaDeviceSynchronize(),"HCIS coverage initialization failed");

                for(int level = 2;level <= radius;++level)
                {
                    compute_next_coverage_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,active,coverage,next_coverage);

                    check_cuda(cudaGetLastError(),"Failed to compute HCIS coverage");
                    check_cuda(cudaDeviceSynchronize(),"HCIS coverage computation failed");

                    std::swap(coverage,next_coverage);
                }

                initialize_max_kernel<<<grid_size,block_size>>>(num_vertices,active,coverage,maximum,owner);

                check_cuda(cudaGetLastError(),"Failed to initialize HCIS maximum");
                check_cuda(cudaDeviceSynchronize(),"HCIS maximum initialization failed");

                for(int level = 1;level <= radius;++level)
                {
                    compute_max_kernel<<<grid_size,block_size>>>(num_vertices,offsets,neighbors,active,maximum,owner,next_maximum,next_owner);

                    check_cuda(cudaGetLastError(),"Failed to compute HCIS maximum");
                    check_cuda(cudaDeviceSynchronize(),"HCIS maximum computation failed");

                    std::swap(maximum,next_maximum);
                    std::swap(owner,next_owner);
                }

                compute_max_flag_kernel<<<grid_size,block_size>>>(num_vertices,active,coverage,maximum,owner,max_flag);

                check_cuda(cudaGetLastError(),"Failed to compute HCIS max flags");
                check_cuda(cudaDeviceSynchronize(),"HCIS max-flag computation failed");

                select_centers_kernel<<<grid_size,block_size>>>(num_vertices,active,max_flag,is_center,newly_selected);

                check_cuda(cudaGetLastError(),"Failed to select HCIS centers");
                check_cuda(cudaDeviceSynchronize(),"HCIS center selection failed");

                check_cuda(cudaMemset(selected_count,0,sizeof(int)),"Failed to reset HCIS selected count");

                count_selected_kernel<<<grid_size,block_size>>>(num_vertices,newly_selected,selected_count);

                check_cuda(cudaGetLastError(),"Failed to count HCIS selected centers");
                check_cuda(cudaDeviceSynchronize(),"HCIS selected-center count failed");

                int host_selected_count = 0;

                check_cuda(cudaMemcpy(&host_selected_count,selected_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy HCIS selected count");

                std::cout << "HCIS round " << round << ": active=" << host_active_count << " selected=" << host_selected_count << '\n';

                if(host_selected_count == 0)
                {
                    throw std::runtime_error("HCIS selected zero centers while active vertices remain");
                }

                // Modified HCIS-r (Sec. 4.1): only keep this phase's centers if the
                // inter-cluster edges they can contribute are bounded by beta.
                if(max_boundary_edges >= 0)
                {
                    const long long boundary_edges = count_compact_boundary_edges(num_vertices,radius,offsets,neighbors,newly_selected);

                    if(boundary_edges > max_boundary_edges)
                    {
                        std::cout << "HCIS round " << round << ": rejected (boundary edges=" << boundary_edges << " > beta=" << max_boundary_edges << ")\n";

                        reject_batch_kernel<<<grid_size,block_size>>>(num_vertices,newly_selected,is_center,active);

                        check_cuda(cudaGetLastError(),"Failed to reject HCIS centers");
                        check_cuda(cudaDeviceSynchronize(),"HCIS center rejection failed");

                        update_compact_active_neighbors(num_vertices,offsets,neighbors,active,active_neighbors);

                        continue;
                    }
                }

                deactivate_compact_radius(num_vertices,radius,offsets,neighbors,newly_selected,active);

                update_compact_active_neighbors(num_vertices,offsets,neighbors,active,active_neighbors);
            }
        }
        catch(...)
        {
            cudaFree(active);
            cudaFree(active_neighbors);
            cudaFree(coverage);
            cudaFree(next_coverage);
            cudaFree(maximum);
            cudaFree(next_maximum);
            cudaFree(owner);
            cudaFree(next_owner);
            cudaFree(max_flag);
            cudaFree(newly_selected);
            cudaFree(active_count);
            cudaFree(selected_count);
            throw;
        }

        cudaFree(active);
        cudaFree(active_neighbors);
        cudaFree(coverage);
        cudaFree(next_coverage);
        cudaFree(maximum);
        cudaFree(next_maximum);
        cudaFree(owner);
        cudaFree(next_owner);
        cudaFree(max_flag);
        cudaFree(newly_selected);
        cudaFree(active_count);
        cudaFree(selected_count);
    }
}