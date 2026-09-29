#include "algorithms/compact/compact.hpp"
#include "gpu/compact_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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
    Compact::Compact(const GPUGraph& graph,int radius)
        : graph_(graph), radius_(radius)
    {
        if(radius_ < 0)
        {
            throw std::invalid_argument("Compact requires a non-negative radius");
        }
    }

    void Compact::run()
    {
        const int num_vertices = static_cast<int>(graph_.num_vertices());
        const int max_edges = static_cast<int>(graph_.num_vertices() + graph_.num_adjacency_entries());

        const std::size_t vertex_bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);
        const std::size_t edge_bytes = static_cast<std::size_t>(max_edges) * sizeof(int);

        int* is_center = nullptr;
        int* distances = nullptr;
        int* centers = nullptr;
        int* parents = nullptr;
        int* edge_count = nullptr;
        int* spanner_sources = nullptr;
        int* spanner_destinations = nullptr;

        auto free_all = [&]()
        {
            cudaFree(is_center);
            cudaFree(distances);
            cudaFree(centers);
            cudaFree(parents);
            cudaFree(edge_count);
            cudaFree(spanner_sources);
            cudaFree(spanner_destinations);
        };

        try
        {
            check_cuda(cudaMalloc(&is_center,vertex_bytes),"Failed to allocate is_center");
            check_cuda(cudaMalloc(&distances,vertex_bytes),"Failed to allocate distances");
            check_cuda(cudaMalloc(&centers,vertex_bytes),"Failed to allocate centers");
            check_cuda(cudaMalloc(&parents,vertex_bytes),"Failed to allocate parents");
            check_cuda(cudaMalloc(&edge_count,sizeof(int)),"Failed to allocate edge count");
            check_cuda(cudaMalloc(&spanner_sources,edge_bytes),"Failed to allocate spanner sources");
            check_cuda(cudaMalloc(&spanner_destinations,edge_bytes),"Failed to allocate spanner destinations");

            // Stage 1: select a maximal radius_-hop high-coverage
            // independent set of cluster centers (Algorithms 1 & 2).
            run_hcis_r(num_vertices,radius_,graph_.offsets(),graph_.neighbors(),is_center);

            // Stage 2: grow radius_-bounded clusters around those
            // centers (bounded multi-source BFS).
            grow_compact_clusters(num_vertices,radius_,graph_.offsets(),graph_.neighbors(),is_center,distances,centers,parents);

            // Stage 3: union the cluster trees with MPVX-Rule
            // inter-cluster bridge edges.
            build_compact_spanner(num_vertices,graph_.offsets(),graph_.neighbors(),distances,centers,parents,edge_count,spanner_sources,spanner_destinations,max_edges);

            int host_edge_count = 0;

            check_cuda(cudaMemcpy(&host_edge_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read compact edge count");

            if(host_edge_count > max_edges)
            {
                throw std::runtime_error("Compact spanner reported more edges than the allocated capacity");
            }

            std::vector<int> host_sources(static_cast<std::size_t>(host_edge_count));
            std::vector<int> host_destinations(static_cast<std::size_t>(host_edge_count));

            if(host_edge_count > 0)
            {
                check_cuda(cudaMemcpy(host_sources.data(),spanner_sources,static_cast<std::size_t>(host_edge_count) * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy compact spanner sources");
                check_cuda(cudaMemcpy(host_destinations.data(),spanner_destinations,static_cast<std::size_t>(host_edge_count) * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy compact spanner destinations");
            }

            std::vector<int> host_is_center(static_cast<std::size_t>(num_vertices));

            check_cuda(cudaMemcpy(host_is_center.data(),is_center,vertex_bytes,cudaMemcpyDeviceToHost),"Failed to copy compact center flags");

            num_centers_ = 0;

            for(int flag : host_is_center)
            {
                if(flag != 0)
                {
                    ++num_centers_;
                }
            }

            edges_.clear();
            edges_.reserve(static_cast<std::size_t>(host_edge_count));

            for(int i = 0;i < host_edge_count;++i)
            {
                int source = host_sources[static_cast<std::size_t>(i)];
                int destination = host_destinations[static_cast<std::size_t>(i)];

                if(source == destination)
                {
                    continue;
                }

                if(source > destination)
                {
                    std::swap(source,destination);
                }

                edges_.push_back(Edge{source,destination});
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
        }
        catch(...)
        {
            free_all();
            throw;
        }

        free_all();
    }

    const std::vector<Edge>& Compact::edges() const noexcept
    {
        return edges_;
    }

    int Compact::num_centers() const noexcept
    {
        return num_centers_;
    }
}