#pragma once

#include <vector>

#include "graph/edge.hpp"
#include "gpu/gpu_graph.hpp"

namespace spanner
{
    class MPVXCS
    {
        public:
            explicit MPVXCS(const GPUGraph& graph,int k);
            void run();
            const std::vector<Edge>& edges() const noexcept;
            int num_centers() const noexcept;
            int num_residual() const noexcept;

        private:
            const GPUGraph& graph_;
            int k_;
            std::vector<Edge> edges_;
            int num_centers_ = 0;
            int num_residual_ = 0;
    };
}