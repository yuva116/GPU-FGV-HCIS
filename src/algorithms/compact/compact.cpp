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
        : graph_(graph),radius_(radius)
    {
        if(radius_ < 1)
        {
            throw std::invalid_argument("Compact requires a positive radius");
        }
    }

    void Compact::run()
    {
        const int num_vertices = static_cast<int>(graph_.num_vertices());
        const int max_edges = static_cast<int>(graph_.num_adjacency_entries());

        if(num_vertices <= 0)
        {
            edges_.clear();
            num_centers_ = 0;
            return;
        }

        const std::size_t vertex_bytes = static_cast<std::size_t>(num_vertices) * sizeof(int);
        const std::size_t edge_bytes = static_cast<std::size_t>(std::max(1,max_edges)) * sizeof(int);

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
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&is_center),vertex_bytes),"Failed to allocate compact center flags");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&distances),vertex_bytes),"Failed to allocate compact distances");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&centers),vertex_bytes),"Failed to allocate compact centers");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&parents),vertex_bytes),"Failed to allocate compact parents");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&edge_count),sizeof(int)),"Failed to allocate compact edge count");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&spanner_sources),edge_bytes),"Failed to allocate compact spanner sources");
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&spanner_destinations),edge_bytes),"Failed to allocate compact spanner destinations");

            check_cuda(cudaMemset(is_center,0,vertex_bytes),"Failed to initialize compact center flags");
            check_cuda(cudaMemset(distances,0,vertex_bytes),"Failed to initialize compact distances");
            check_cuda(cudaMemset(centers,0,vertex_bytes),"Failed to initialize compact centers");
            check_cuda(cudaMemset(parents,0,vertex_bytes),"Failed to initialize compact parents");
            check_cuda(cudaMemset(edge_count,0,sizeof(int)),"Failed to initialize compact edge count");

            run_hcis_r(num_vertices,radius_,graph_.offsets(),graph_.neighbors(),is_center);

            check_cuda(cudaGetLastError(),"HCIS-r kernel launch failed");
            check_cuda(cudaDeviceSynchronize(),"HCIS-r execution failed");

            grow_compact_clusters(num_vertices,radius_,graph_.offsets(),graph_.neighbors(),is_center,distances,centers,parents);

            check_cuda(cudaGetLastError(),"Compact cluster kernel launch failed");
            check_cuda(cudaDeviceSynchronize(),"Compact cluster execution failed");

            build_compact_spanner(num_vertices,graph_.offsets(),graph_.neighbors(),distances,centers,parents,edge_count,spanner_sources,spanner_destinations,max_edges);

            check_cuda(cudaGetLastError(),"Compact spanner kernel launch failed");
            check_cuda(cudaDeviceSynchronize(),"Compact spanner execution failed");

            int host_edge_count = 0;

            check_cuda(cudaMemcpy(&host_edge_count,edge_count,sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy compact edge count");

            if(host_edge_count < 0)
            {
                throw std::runtime_error("Compact spanner produced an invalid edge count");
            }

            if(host_edge_count > max_edges)
            {
                host_edge_count = max_edges;
            }

            std::vector<int> host_sources(static_cast<std::size_t>(host_edge_count));
            std::vector<int> host_destinations(static_cast<std::size_t>(host_edge_count));

            if(host_edge_count > 0)
            {
                const std::size_t result_bytes = static_cast<std::size_t>(host_edge_count) * sizeof(int);

                check_cuda(cudaMemcpy(host_sources.data(),spanner_sources,result_bytes,cudaMemcpyDeviceToHost),"Failed to copy compact spanner sources");
                check_cuda(cudaMemcpy(host_destinations.data(),spanner_destinations,result_bytes,cudaMemcpyDeviceToHost),"Failed to copy compact spanner destinations");
            }

            std::vector<int> host_is_center(static_cast<std::size_t>(num_vertices));

            check_cuda(cudaMemcpy(host_is_center.data(),is_center,vertex_bytes,cudaMemcpyDeviceToHost),"Failed to copy compact center flags");

            num_centers_ = 0;

            for(const int flag : host_is_center)
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

                if(source < 0 || destination < 0 || source >= num_vertices || destination >= num_vertices)
                {
                    continue;
                }

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