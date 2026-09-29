#include "gpu/mpvx_cs.hpp"

#include "gpu/compact_kernels.hpp"
#include "gpu/edge_finalize.hpp"
#include "gpu/kernels.hpp"
#include "compact_internal.hpp"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace
{
    using spanner::detail::DeviceBuffer;
    using spanner::detail::check_cuda;
    using spanner::detail::grid_size;
    using spanner::detail::kBlockSize;

    template<typename T>
    thrust::device_ptr<T> device_ptr(T* pointer)
    {
        return thrust::device_pointer_cast(pointer);
    }

    __global__ void init_growth_kernel(int n,const int* is_center,int* distance,int* center,int* parent)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex < n)
        {
            const bool root = is_center[vertex] != 0;
            distance[vertex] = root ? 0 : -1;
            center[vertex] = root ? vertex : -1;
            parent[vertex] = root ? vertex : -1;
        }
    }

    __global__ void grow_level_kernel(int n,int level,const int* offsets,const int* neighbors,int* distance,int* center,int* parent,int* changed)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= n || distance[vertex] >= 0)
        {
            return;
        }

        int best_parent = -1;
        int best_center = INT_MAX;

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];
            if(distance[neighbor] == level - 1 && center[neighbor] < best_center)
            {
                best_center = center[neighbor];
                best_parent = neighbor;
            }
        }

        if(best_parent >= 0)
        {
            distance[vertex] = level;
            center[vertex] = best_center;
            parent[vertex] = best_parent;
            atomicExch(changed,1);
        }
    }

    __global__ void residual_flag_kernel(int n,const int* distance,int* residual)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex < n)
        {
            residual[vertex] = distance[vertex] < 0 ? 1 : 0;
        }
    }

    __global__ void residual_degree_kernel(int n,const int* offsets,const int* neighbors,const int* residual,int* degree)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= n)
        {
            return;
        }

        int count = 0;
        if(residual[vertex])
        {
            for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
            {
                count += residual[neighbors[edge]];
            }
        }
        degree[vertex] = count;
    }

    __global__ void residual_build_kernel(int n,const int* offsets,const int* neighbors,const int* residual,const int* new_id,const int* position,int* residual_offsets,int* residual_neighbors,int* old_vertex)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= n || !residual[vertex])
        {
            return;
        }

        const int residual_vertex = new_id[vertex];
        int output = position[vertex];
        residual_offsets[residual_vertex] = output;
        old_vertex[residual_vertex] = vertex;

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];
            if(residual[neighbor])
            {
                residual_neighbors[output++] = new_id[neighbor];
            }
        }
    }

    __global__ void append_tree_edges_kernel(int n,const int* distance,const int* parent,const int* old_vertex,bool residual_tree,std::uint64_t* packed_edges,int* edge_count,int capacity)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= n)
        {
            return;
        }

        const int p = parent[vertex];
        if(distance[vertex] <= 0 || p < 0 || p == vertex)
        {
            return;
        }

        const int source = residual_tree ? old_vertex[vertex] : vertex;
        const int destination = residual_tree ? old_vertex[p] : p;
        const int low = source < destination ? source : destination;
        const int high = source < destination ? destination : source;
        const int position = atomicAdd(edge_count,1);
        if(position < capacity)
        {
            packed_edges[position] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) << 32) |
                                     static_cast<std::uint32_t>(high);
        }
    }

    __global__ void assemble_clusters_kernel(int n,const int* high_distance,const int* high_center,const int* new_id,const int* residual_center,const int* old_vertex,int* cluster)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= n)
        {
            return;
        }

        if(high_distance[vertex] >= 0)
        {
            cluster[vertex] = high_center[vertex];
        }
        else
        {
            cluster[vertex] = old_vertex[residual_center[new_id[vertex]]];
        }
    }

    __global__ void boundary_candidates_kernel(int n,const int* offsets,const int* neighbors,const int* cluster,std::uint64_t* keys,std::uint64_t* values,int* count)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
        if(vertex >= n)
        {
            return;
        }

        const int source_cluster = cluster[vertex];
        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];
            const int destination_cluster = cluster[neighbor];

            if(source_cluster < destination_cluster)
            {
                const int position = atomicAdd(count,1);
                keys[position] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertex)) << 32) |
                                 static_cast<std::uint32_t>(destination_cluster);
                values[position] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(vertex)) << 32) |
                                   static_cast<std::uint32_t>(neighbor);
            }
        }
    }

    __global__ void append_boundary_edges_kernel(int count,const std::uint64_t* values,std::uint64_t* packed_edges,int* edge_count,int capacity)
    {
        const int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count)
        {
            return;
        }

        const std::uint64_t value = values[index];
        const int source = static_cast<int>(value >> 32);
        const int destination = static_cast<int>(value & 0xffffffffULL);
        const int low = source < destination ? source : destination;
        const int high = source < destination ? destination : source;
        const int position = atomicAdd(edge_count,1);
        if(position < capacity)
        {
            packed_edges[position] = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) << 32) |
                                     static_cast<std::uint32_t>(high);
        }
    }

    void launch_check(const char* message)
    {
        check_cuda(cudaGetLastError(),message);
    }
}

namespace spanner
{
    void run_mpvx_cs(const int* offsets,const int* neighbors,int num_vertices,int num_adjacency_entries,int k,std::vector<Edge>& edges,int& num_centers,int& num_residual)
    {
        edges.clear();
        num_centers = 0;
        num_residual = 0;

        if(num_vertices <= 0)
        {
            return;
        }
        if(k < 1 || k > 6)
        {
            throw std::invalid_argument("MPVXCS requires 1 <= k <= 6 (HCIS radius is limited to 12)");
        }

        const int n = num_vertices;
        const int radius = 2 * k;
        const int blocks = grid_size(n);
        DeviceBuffer<int> is_center(n),high_distance(n),high_center(n),high_parent(n),residual(n),new_id(n),cluster(n);

        const double beta_cs_value = std::pow(static_cast<double>(n),1.0 + 1.0 / static_cast<double>(k));
        const long long beta_cs = beta_cs_value >= static_cast<double>(LLONG_MAX)
            ? LLONG_MAX
            : static_cast<long long>(beta_cs_value);
        run_hcis_r(n,radius,offsets,neighbors,is_center.get(),10,beta_cs);
        num_centers = static_cast<int>(thrust::reduce(device_ptr(is_center.get()),device_ptr(is_center.get()) + n,0));

        init_growth_kernel<<<blocks,kBlockSize>>>(n,is_center.get(),high_distance.get(),high_center.get(),high_parent.get());
        launch_check("MPVX-CS high-coverage BFS initialization failed");

        DeviceBuffer<int> changed(1);
        for(int level = 1;level <= radius;++level)
        {
            check_cuda(cudaMemset(changed.get(),0,sizeof(int)),"Failed to reset MPVX-CS BFS flag");
            grow_level_kernel<<<blocks,kBlockSize>>>(n,level,offsets,neighbors,high_distance.get(),high_center.get(),high_parent.get(),changed.get());
            launch_check("MPVX-CS high-coverage BFS level failed");

            int host_changed = 0;
            check_cuda(cudaMemcpy(&host_changed,changed.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read MPVX-CS BFS flag");
            if(host_changed == 0)
            {
                break;
            }
        }

        residual_flag_kernel<<<blocks,kBlockSize>>>(n,high_distance.get(),residual.get());
        launch_check("MPVX-CS residual marking failed");
        thrust::exclusive_scan(device_ptr(residual.get()),device_ptr(residual.get()) + n,device_ptr(new_id.get()));

        int last_residual = 0;
        int last_id = 0;
        check_cuda(cudaMemcpy(&last_residual,residual.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read MPVX-CS residual count");
        check_cuda(cudaMemcpy(&last_id,new_id.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read MPVX-CS residual id");
        const int residual_vertices = last_id + last_residual;
        num_residual = residual_vertices;

        const int capacity = 2 * n + num_adjacency_entries + 1;
        DeviceBuffer<std::uint64_t> packed_edges(capacity);
        DeviceBuffer<int> edge_count(1);
        check_cuda(cudaMemset(edge_count.get(),0,sizeof(int)),"Failed to reset MPVX-CS edge counter");

        append_tree_edges_kernel<<<blocks,kBlockSize>>>(n,high_distance.get(),high_parent.get(),nullptr,false,packed_edges.get(),edge_count.get(),capacity);
        launch_check("MPVX-CS high-coverage tree extraction failed");

        if(residual_vertices > 0)
        {
            DeviceBuffer<int> degree(n),position(n);
            residual_degree_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,residual.get(),degree.get());
            launch_check("MPVX-CS residual degree construction failed");
            thrust::exclusive_scan(device_ptr(degree.get()),device_ptr(degree.get()) + n,device_ptr(position.get()));

            int last_degree = 0;
            int last_position = 0;
            check_cuda(cudaMemcpy(&last_degree,degree.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual degree total");
            check_cuda(cudaMemcpy(&last_position,position.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual adjacency position");
            const int residual_adjacency = last_degree + last_position;

            DeviceBuffer<int> residual_offsets(static_cast<std::size_t>(residual_vertices) + 1);
            DeviceBuffer<int> residual_neighbors(std::max(1,residual_adjacency));
            DeviceBuffer<int> old_vertex(residual_vertices);
            residual_build_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,residual.get(),new_id.get(),position.get(),residual_offsets.get(),residual_neighbors.get(),old_vertex.get());
            launch_check("MPVX-CS residual CSR construction failed");
            check_cuda(cudaMemcpy(residual_offsets.get() + residual_vertices,&residual_adjacency,sizeof(int),cudaMemcpyHostToDevice),"Failed to finish residual CSR offsets");

            DeviceBuffer<float> shifts(residual_vertices);
            DeviceBuffer<int> residual_distance(residual_vertices),residual_center(residual_vertices),residual_parent(residual_vertices);
            const float beta_est = n == 1 ? 1.0f : static_cast<float>(std::log(static_cast<double>(n)) / (2.0 * static_cast<double>(k)));
            initialize_miller(residual_vertices,beta_est,shifts.get(),residual_distance.get(),residual_center.get(),residual_parent.get());
            build_miller_clusters(residual_vertices,0,residual_offsets.get(),residual_neighbors.get(),shifts.get(),residual_distance.get(),residual_center.get(),residual_parent.get());

            const int residual_blocks = grid_size(residual_vertices);
            append_tree_edges_kernel<<<residual_blocks,kBlockSize>>>(residual_vertices,residual_distance.get(),residual_parent.get(),old_vertex.get(),true,packed_edges.get(),edge_count.get(),capacity);
            launch_check("MPVX-CS Miller forest extraction failed");
            assemble_clusters_kernel<<<blocks,kBlockSize>>>(n,high_distance.get(),high_center.get(),new_id.get(),residual_center.get(),old_vertex.get(),cluster.get());
            launch_check("MPVX-CS global cluster assembly failed");
        }
        else
        {
            check_cuda(cudaMemcpy(cluster.get(),high_center.get(),static_cast<std::size_t>(n) * sizeof(int),cudaMemcpyDeviceToDevice),"Failed to copy high-coverage clusters");
        }

        if(num_adjacency_entries > 0)
        {
            DeviceBuffer<std::uint64_t> candidate_keys(num_adjacency_entries),candidate_values(num_adjacency_entries);
            DeviceBuffer<int> candidate_count(1);
            check_cuda(cudaMemset(candidate_count.get(),0,sizeof(int)),"Failed to reset MPVX-CS boundary counter");

            boundary_candidates_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,cluster.get(),candidate_keys.get(),candidate_values.get(),candidate_count.get());
            launch_check("MPVX-CS boundary candidate generation failed");

            int candidates = 0;
            check_cuda(cudaMemcpy(&candidates,candidate_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read MPVX-CS boundary candidate count");
            if(candidates > num_adjacency_entries)
            {
                throw std::runtime_error("MPVX-CS boundary candidates exceeded adjacency capacity");
            }

            if(candidates > 0)
            {
                thrust::sort_by_key(device_ptr(candidate_keys.get()),device_ptr(candidate_keys.get()) + candidates,device_ptr(candidate_values.get()));
                const auto unique_end = thrust::unique_by_key(device_ptr(candidate_keys.get()),device_ptr(candidate_keys.get()) + candidates,device_ptr(candidate_values.get()));
                const int unique_count = static_cast<int>(unique_end.first - device_ptr(candidate_keys.get()));

                append_boundary_edges_kernel<<<grid_size(unique_count),kBlockSize>>>(unique_count,candidate_values.get(),packed_edges.get(),edge_count.get(),capacity);
                launch_check("MPVX-CS boundary edge extraction failed");
            }
        }

        check_cuda(cudaDeviceSynchronize(),"MPVX-CS pipeline failed");
        int total_edges = 0;
        check_cuda(cudaMemcpy(&total_edges,edge_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read MPVX-CS edge count");
        if(total_edges > capacity)
        {
            throw std::runtime_error("MPVX-CS spanner exceeded allocated edge capacity");
        }
        edges = finalize_packed_edges(packed_edges.get(),total_edges);
    }
}