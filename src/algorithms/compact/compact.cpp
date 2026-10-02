#include "algorithms/compact/compact.hpp"
#include "gpu/compact_kernels.hpp"

#include <stdexcept>

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
        run_compact_pipeline(graph_.offsets(),graph_.neighbors(),static_cast<int>(graph_.num_vertices()),static_cast<int>(graph_.num_adjacency_entries()),radius_,edges_,num_centers_);
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
