#pragma once

#include <cstddef>
#include <vector>

#include "graph/edge.hpp"
#include "gpu/gpu_graph.hpp"

namespace spanner
{
    class Miller
    {
    public:
        Miller(const GPUGraph& graph, int k);
        void run();
        const std::vector<Edge>& edges() const noexcept;

    private:
        const GPUGraph& graph_;
        int k_;
        float beta_;
        std::vector<Edge> edges_;
    };
}
