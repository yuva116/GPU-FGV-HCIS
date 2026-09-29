#include "algorithms/fgv/fgv.hpp"
#include "gpu/kernels.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void check_cuda(cudaError_t status,const char* message)
    {
        if(status != cudaSuccess)
        {
            throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
        }
    }
}

namespace spanner
{
    FGV::FGV(const GPUGraph& graph,int k)
        : graph_(graph),
          k_(k),
          radius_(k - 1),
          probability_(0.0f)
    {
        if(k_ < 2)
        {
            throw std::invalid_argument("FGV requires k >= 2");
        }

        const double n = static_cast<double>(graph_.num_vertices());

        probability_ = static_cast<float>(1.0 - std::pow(n,-1.0 / static_cast<double>(k_)));
    }

    void FGV::run()
    {
        const int num_vertices = static_cast<int>(graph_.num_vertices());
        const int num_adjacency_entries = static_cast<int>(graph_.num_adjacency_entries());

        int* shifts = nullptr;
        int* distances = nullptr;
        int* centers = nullptr;
        int* parents = nullptr;
        int* edge_count = nullptr;
        int* spanner_sources = nullptr;
        int* spanner_destinations = nullptr;

        const std::size_t vertex_bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);
        const int max_edges = num_vertices + num_adjacency_entries;

        check_cuda(cudaMalloc(&shifts,vertex_bytes),"Failed to allocate FGV shifts");
        check_cuda(cudaMalloc(&distances,vertex_bytes),"Failed to allocate FGV distances");
        check_cuda(cudaMalloc(&centers,vertex_bytes),"Failed to allocate FGV centers");
        check_cuda(cudaMalloc(&parents,vertex_bytes),"Failed to allocate FGV parents");
        check_cuda(cudaMalloc(&edge_count,sizeof(int)),"Failed to allocate FGV edge count");
        check_cuda(cudaMalloc(&spanner_sources,static_cast<std::size_t>(max_edges) * sizeof(int)),"Failed to allocate FGV spanner sources");
        check_cuda(cudaMalloc(&spanner_destinations,static_cast<std::size_t>(max_edges) * sizeof(int)),"Failed to allocate FGV spanner destinations");

        initialize_fgv(num_vertices,radius_,probability_,shifts,distances,centers,parents);

        build_fgv_clusters(num_vertices,radius_,graph_.offsets(),graph_.neighbors(),shifts,distances,centers,parents);

        build_fgv_spanner(num_vertices,graph_.offsets(),graph_.neighbors(),distances,centers,parents,edge_count,spanner_sources,spanner_destinations,max_edges);

        int host_edge_count = 0;

        check_cuda(cudaMemcpy(&host_edge_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy FGV edge count");

        if(host_edge_count > max_edges)
        {
            cudaFree(shifts);
            cudaFree(distances);
            cudaFree(centers);
            cudaFree(parents);
            cudaFree(edge_count);
            cudaFree(spanner_sources);
            cudaFree(spanner_destinations);

            throw std::runtime_error("FGV spanner exceeded allocated edge capacity");
        }

        std::vector<int> host_sources(static_cast<std::size_t>(host_edge_count));
        std::vector<int> host_destinations(static_cast<std::size_t>(host_edge_count));

        if(host_edge_count > 0)
        {
            check_cuda(cudaMemcpy(host_sources.data(),spanner_sources,static_cast<std::size_t>(host_edge_count) * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy FGV spanner sources");

            check_cuda(cudaMemcpy(host_destinations.data(),spanner_destinations,static_cast<std::size_t>(host_edge_count) * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy FGV spanner destinations");
        }

        edges_.clear();
        edges_.reserve(static_cast<std::size_t>(host_edge_count));

        for(int i = 0;i < host_edge_count;++i)
        {
            const int source = host_sources[i];
            const int destination = host_destinations[i];

            if(source == destination)
            {
                continue;
            }

            if(source < destination)
            {
                edges_.push_back({source,destination});
            }
            else
            {
                edges_.push_back({destination,source});
            }
        }

        std::sort(edges_.begin(),edges_.end(),[](const Edge& lhs,const Edge& rhs)
        {
            if(lhs.source != rhs.source)
            {
                return lhs.source < rhs.source;
            }

            return lhs.destination < rhs.destination;
        });

        edges_.erase(std::unique(edges_.begin(),edges_.end(),[](const Edge& lhs,const Edge& rhs)
        {
            return lhs.source == rhs.source && lhs.destination == rhs.destination;
        }),edges_.end());

        cudaFree(shifts);
        cudaFree(distances);
        cudaFree(centers);
        cudaFree(parents);
        cudaFree(edge_count);
        cudaFree(spanner_sources);
        cudaFree(spanner_destinations);
    }

    const std::vector<Edge>& FGV::edges() const noexcept
    {
        return edges_;
    }
}