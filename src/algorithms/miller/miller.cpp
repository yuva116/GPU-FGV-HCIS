#include "algorithms/miller/miller.hpp"

#include "gpu/edge_finalize.hpp"
#include "gpu/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void check_cuda(cudaError_t status, const char* message)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
        }
    }
}

namespace spanner
{
    Miller::Miller(const GPUGraph& graph, int k)
        : graph_(graph),
          k_(k),
          beta_(0.0f),
          edges_()
    {
        if(k_ < 1)
        {
            throw std::invalid_argument("Miller requires k >= 1");
        }

        const double n = static_cast<double>(graph_.num_vertices());
        if(n == 0.0)
        {
            throw std::invalid_argument("Miller requires at least one vertex");
        }

        beta_ = n == 1.0
            ? 1.0f
            : static_cast<float>(std::log(n) / (2.0 * static_cast<double>(k_)));
    }

    void Miller::run()
    {
        const int num_vertices = static_cast<int>(graph_.num_vertices());
        const int num_adjacency_entries = static_cast<int>(graph_.num_adjacency_entries());

        float* shifts = nullptr;
        int* distances = nullptr;
        int* centers = nullptr;
        int* parents = nullptr;
        int* edge_count = nullptr;
        int* spanner_sources = nullptr;
        int* spanner_destinations = nullptr;

        const std::size_t vertex_bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);
        const std::size_t shift_bytes = static_cast<std::size_t>(num_vertices) * sizeof(float);
        const int max_edges = std::max(1, num_vertices + num_adjacency_entries * 2);

        check_cuda(cudaMalloc(&shifts,shift_bytes),"Failed to allocate Miller shifts");
        check_cuda(cudaMalloc(&distances,vertex_bytes),"Failed to allocate Miller distances");
        check_cuda(cudaMalloc(&centers,vertex_bytes),"Failed to allocate Miller centers");
        check_cuda(cudaMalloc(&parents,vertex_bytes),"Failed to allocate Miller parents");
        check_cuda(cudaMalloc(&edge_count,sizeof(int)),"Failed to allocate Miller edge count");
        check_cuda(cudaMalloc(&spanner_sources,static_cast<std::size_t>(max_edges) * sizeof(int)),"Failed to allocate Miller spanner sources");
        check_cuda(cudaMalloc(&spanner_destinations,static_cast<std::size_t>(max_edges) * sizeof(int)),"Failed to allocate Miller spanner destinations");

        initialize_miller(num_vertices,beta_,shifts,distances,centers,parents);
        build_miller_clusters(num_vertices,0,graph_.offsets(),graph_.neighbors(),shifts,distances,centers,parents);
        build_miller_spanner(num_vertices,graph_.offsets(),graph_.neighbors(),distances,centers,parents,edge_count,spanner_sources,spanner_destinations,max_edges);

        int host_edge_count = 0;
        check_cuda(cudaMemcpy(&host_edge_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy Miller edge count");

        if(host_edge_count > max_edges)
        {
            cudaFree(shifts);
            cudaFree(distances);
            cudaFree(centers);
            cudaFree(parents);
            cudaFree(edge_count);
            cudaFree(spanner_sources);
            cudaFree(spanner_destinations);
            throw std::runtime_error("Miller spanner exceeded allocated edge capacity");
        }

        try
        {
            edges_ = copy_canonical_unique_edges(spanner_sources,spanner_destinations,host_edge_count);
        }
        catch(...)
        {
            cudaFree(shifts);
            cudaFree(distances);
            cudaFree(centers);
            cudaFree(parents);
            cudaFree(edge_count);
            cudaFree(spanner_sources);
            cudaFree(spanner_destinations);
            throw;
        }

        cudaFree(shifts);
        cudaFree(distances);
        cudaFree(centers);
        cudaFree(parents);
        cudaFree(edge_count);
        cudaFree(spanner_sources);
        cudaFree(spanner_destinations);
    }

    const std::vector<Edge>& Miller::edges() const noexcept
    {
        return edges_;
    }
}
