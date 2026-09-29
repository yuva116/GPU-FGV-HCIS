#pragma once

#include <cstddef>
#include <vector>

#include "graph/edge.hpp"
#include "gpu/gpu_graph.hpp"

namespace spanner
{
    class Compact
    {
        public:
            Compact(const GPUGraph& graph,int radius);
            void run();
            const std::vector<Edge>& edges() const noexcept;
            int num_centers() const noexcept;

        private:
            const GPUGraph& graph_;
            int radius_;
            std::vector<Edge> edges_;
            int num_centers_ = 0;
    };
}