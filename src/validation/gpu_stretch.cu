#include "validation/gpu_stretch.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

struct DeviceValidationResult
{
    int violation_flag;
    int violation_source;
    int violation_destination;
    int maximum_stretch;
    int checked_edge_incidences;
};

__global__ void initialize_source_kernel(int source,int* distance,int* frontier)
{
    distance[source] = 0;
    frontier[0] = source;
}

__global__ void expand_frontier_kernel(const int* offsets,const int* neighbors,const int* frontier,int frontier_count,int depth,int* distance,int* next_frontier,int* next_count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index >= frontier_count)
    {
        return;
    }

    const int vertex = frontier[index];
    const int begin = offsets[vertex];
    const int end = offsets[vertex + 1];

    for(int edge_index = begin;edge_index < end;++edge_index)
    {
        const int neighbor = neighbors[edge_index];

        if(atomicCAS(&distance[neighbor],-1,depth + 1) == -1)
        {
            const int position = atomicAdd(next_count,1);
            next_frontier[position] = neighbor;
        }
    }
}

__global__ void check_source_edges_kernel(int source,const int* offsets,const int* neighbors,const int* distance,int stretch_bound,DeviceValidationResult* result)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int begin = offsets[source];
    const int end = offsets[source + 1];
    const int edge_index = begin + index;

    if(edge_index >= end)
    {
        return;
    }

    const int neighbor = neighbors[edge_index];

    if(neighbor == source)
    {
        return;
    }

    atomicAdd(&result->checked_edge_incidences,1);

    const int neighbor_distance = distance[neighbor];

    if(neighbor_distance < 0 || neighbor_distance > stretch_bound)
    {
        if(atomicCAS(&result->violation_flag,0,1) == 0)
        {
            result->violation_source = source;
            result->violation_destination = neighbor;
        }

        return;
    }

    atomicMax(&result->maximum_stretch,neighbor_distance);
}

void check_cuda(cudaError_t error,const char* message)
{
    if(error != cudaSuccess)
    {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

void build_device_csr(const spanner::CSRGraph& graph,int*& d_offsets,int*& d_neighbors)
{
    const int num_vertices = static_cast<int>(graph.num_vertices());
    const int adjacency_entries = static_cast<int>(graph.num_adjacency_entries());

    std::vector<int> offsets(static_cast<std::size_t>(num_vertices + 1));
    std::vector<int> neighbors(static_cast<std::size_t>(adjacency_entries));

    for(int vertex = 0;vertex <= num_vertices;++vertex)
    {
        offsets[vertex] = static_cast<int>(graph.offsets()[vertex]);
    }

    for(int index = 0;index < adjacency_entries;++index)
    {
        neighbors[index] = graph.neighbors_data()[index];
    }

    check_cuda(cudaMalloc(&d_offsets,static_cast<std::size_t>(num_vertices + 1) * sizeof(int)),"Failed to allocate input CSR offsets");

    check_cuda(cudaMalloc(&d_neighbors,static_cast<std::size_t>(adjacency_entries) * sizeof(int)),"Failed to allocate input CSR neighbors");

    check_cuda(cudaMemcpy(d_offsets,offsets.data(),static_cast<std::size_t>(num_vertices + 1) * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy input CSR offsets");

    check_cuda(cudaMemcpy(d_neighbors,neighbors.data(),static_cast<std::size_t>(adjacency_entries) * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy input CSR neighbors");
}

void build_spanner_csr(int num_vertices,const std::vector<spanner::Edge>& edges,int*& d_offsets,int*& d_neighbors)
{
    std::vector<int> degree(static_cast<std::size_t>(num_vertices),0);

    for(const spanner::Edge& edge : edges)
    {
        if(edge.source < 0 || edge.destination < 0 || edge.source >= num_vertices || edge.destination >= num_vertices)
        {
            throw std::runtime_error("Invalid spanner edge while building validation CSR");
        }

        if(edge.source == edge.destination)
        {
            continue;
        }

        ++degree[edge.source];
        ++degree[edge.destination];
    }

    std::vector<int> offsets(static_cast<std::size_t>(num_vertices + 1),0);

    for(int vertex = 0;vertex < num_vertices;++vertex)
    {
        offsets[vertex + 1] = offsets[vertex] + degree[vertex];
    }

    const int adjacency_entries = offsets[num_vertices];

    std::vector<int> neighbors(static_cast<std::size_t>(adjacency_entries));
    std::vector<int> cursor = offsets;

    for(const spanner::Edge& edge : edges)
    {
        if(edge.source == edge.destination)
        {
            continue;
        }

        neighbors[cursor[edge.source]++] = edge.destination;
        neighbors[cursor[edge.destination]++] = edge.source;
    }

    check_cuda(cudaMalloc(&d_offsets,static_cast<std::size_t>(num_vertices + 1) * sizeof(int)),"Failed to allocate spanner CSR offsets");

    check_cuda(cudaMalloc(&d_neighbors,static_cast<std::size_t>(adjacency_entries) * sizeof(int)),"Failed to allocate spanner CSR neighbors");

    check_cuda(cudaMemcpy(d_offsets,offsets.data(),static_cast<std::size_t>(num_vertices + 1) * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy spanner CSR offsets");

    if(adjacency_entries > 0)
    {
        check_cuda(cudaMemcpy(d_neighbors,neighbors.data(),static_cast<std::size_t>(adjacency_entries) * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy spanner CSR neighbors");
    }
}

}

namespace spanner
{

GPUStretchResult validate_gpu_stretch(const CSRGraph& graph,const std::vector<Edge>& spanner_edges,int stretch_bound,int max_sources)
{
    if(stretch_bound <= 0)
    {
        throw std::runtime_error("GPU stretch bound must be positive");
    }

    const int num_vertices = static_cast<int>(graph.num_vertices());

    if(num_vertices == 0)
    {
        throw std::runtime_error("Cannot validate an empty graph");
    }

    int* d_input_offsets = nullptr;
    int* d_input_neighbors = nullptr;
    int* d_spanner_offsets = nullptr;
    int* d_spanner_neighbors = nullptr;
    int* d_distance = nullptr;
    int* d_frontier = nullptr;
    int* d_next_frontier = nullptr;
    int* d_next_count = nullptr;
    DeviceValidationResult* d_result = nullptr;

    try
    {
        build_device_csr(graph,d_input_offsets,d_input_neighbors);
        build_spanner_csr(num_vertices,spanner_edges,d_spanner_offsets,d_spanner_neighbors);

        check_cuda(cudaMalloc(&d_distance,static_cast<std::size_t>(num_vertices) * sizeof(int)),"Failed to allocate GPU distance array");

        check_cuda(cudaMalloc(&d_frontier,static_cast<std::size_t>(num_vertices) * sizeof(int)),"Failed to allocate GPU frontier");

        check_cuda(cudaMalloc(&d_next_frontier,static_cast<std::size_t>(num_vertices) * sizeof(int)),"Failed to allocate GPU next frontier");

        check_cuda(cudaMalloc(&d_next_count,sizeof(int)),"Failed to allocate GPU frontier count");

        check_cuda(cudaMalloc(&d_result,sizeof(DeviceValidationResult)),"Failed to allocate GPU validation result");

        DeviceValidationResult initial_result{};
        initial_result.violation_flag = 0;
        initial_result.violation_source = -1;
        initial_result.violation_destination = -1;
        initial_result.maximum_stretch = 0;
        initial_result.checked_edge_incidences = 0;

        check_cuda(cudaMemcpy(d_result,&initial_result,sizeof(DeviceValidationResult),cudaMemcpyHostToDevice),"Failed to initialize GPU validation result");

        const int source_count = max_sources <= 0 ? num_vertices : std::min(max_sources,num_vertices);

        GPUStretchResult result;
        result.total_sources = num_vertices;
        result.full_validation = source_count == num_vertices;

        constexpr int block_size = 256;

        for(int source_index = 0;source_index < source_count;++source_index)
        {
            const int source = static_cast<int>((static_cast<long long>(source_index) * num_vertices) / source_count);

            check_cuda(cudaMemset(d_distance,0xFF,static_cast<std::size_t>(num_vertices) * sizeof(int)),"Failed to reset GPU distances");

            check_cuda(cudaMemset(d_next_count,0,sizeof(int)),"Failed to reset GPU frontier count");

            initialize_source_kernel<<<1,1>>>(source,d_distance,d_frontier);

            check_cuda(cudaGetLastError(),"Failed to launch GPU source initialization kernel");

            check_cuda(cudaDeviceSynchronize(),"GPU source initialization failed");

            int frontier_count = 1;

            for(int depth = 0;depth < stretch_bound && frontier_count > 0;++depth)
            {
                check_cuda(cudaMemset(d_next_count,0,sizeof(int)),"Failed to reset GPU next frontier count");

                const int grid_size = (frontier_count + block_size - 1) / block_size;

                expand_frontier_kernel<<<grid_size,block_size>>>(d_spanner_offsets,d_spanner_neighbors,d_frontier,frontier_count,depth,d_distance,d_next_frontier,d_next_count);

                check_cuda(cudaGetLastError(),"Failed to launch GPU frontier expansion kernel");

                int next_count = 0;

                check_cuda(cudaMemcpy(&next_count,d_next_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read GPU frontier count");

                std::swap(d_frontier,d_next_frontier);

                frontier_count = next_count;
            }

            const int begin = static_cast<int>(graph.offsets()[source]);
            const int end = static_cast<int>(graph.offsets()[source + 1]);
            const int source_degree = end - begin;

            if(source_degree > 0)
            {
                const int grid_size = (source_degree + block_size - 1) / block_size;

                check_source_edges_kernel<<<grid_size,block_size>>>(source,d_input_offsets,d_input_neighbors,d_distance,stretch_bound,d_result);

                check_cuda(cudaGetLastError(),"Failed to launch GPU stretch validation kernel");

                check_cuda(cudaDeviceSynchronize(),"GPU stretch validation kernel failed");
            }

            DeviceValidationResult host_result{};

            check_cuda(cudaMemcpy(&host_result,d_result,sizeof(DeviceValidationResult),cudaMemcpyDeviceToHost),"Failed to read GPU validation result");

            result.checked_sources = source_index + 1;

            if(host_result.violation_flag != 0)
            {
                result.passed = false;
                result.checked_edge_incidences = host_result.checked_edge_incidences;
                result.maximum_stretch = host_result.maximum_stretch;
                result.violating_source = host_result.violation_source;
                result.violating_destination = host_result.violation_destination;

                cudaFree(d_input_offsets);
                cudaFree(d_input_neighbors);
                cudaFree(d_spanner_offsets);
                cudaFree(d_spanner_neighbors);
                cudaFree(d_distance);
                cudaFree(d_frontier);
                cudaFree(d_next_frontier);
                cudaFree(d_next_count);
                cudaFree(d_result);

                return result;
            }
        }

        DeviceValidationResult host_result{};

        check_cuda(cudaMemcpy(&host_result,d_result,sizeof(DeviceValidationResult),cudaMemcpyDeviceToHost),"Failed to read final GPU validation result");

        result.checked_sources = source_count;
        result.checked_edge_incidences = host_result.checked_edge_incidences;
        result.maximum_stretch = host_result.maximum_stretch;
        result.violating_source = host_result.violation_source;
        result.violating_destination = host_result.violation_destination;
        result.passed = host_result.violation_flag == 0;

        cudaFree(d_input_offsets);
        cudaFree(d_input_neighbors);
        cudaFree(d_spanner_offsets);
        cudaFree(d_spanner_neighbors);
        cudaFree(d_distance);
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_next_count);
        cudaFree(d_result);

        return result;
    }
    catch(...)
    {
        cudaFree(d_input_offsets);
        cudaFree(d_input_neighbors);
        cudaFree(d_spanner_offsets);
        cudaFree(d_spanner_neighbors);
        cudaFree(d_distance);
        cudaFree(d_frontier);
        cudaFree(d_next_frontier);
        cudaFree(d_next_count);
        cudaFree(d_result);

        throw;
    }
}

}