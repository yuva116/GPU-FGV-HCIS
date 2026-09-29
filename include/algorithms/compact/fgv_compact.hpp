#pragma once

#include <cstddef>
#include <vector>

#include "graph/edge.hpp"
#include "gpu/gpu_graph.hpp"

namespace spanner
{
    // Hybrid FGV-CS spanner (paper Sec. 4.3), stretch 2k-1.
    //   1. modified HCIS-r (r = k-1, alpha iterations, beta = n^(1+1/k)) picks high-coverage centers
    //   2. clusters are grown around them up to radius r          (set D)
    //   3. FGV-Spanner runs on the remaining vertices              (set B)
    //   4. inter-cluster edges: FGV-Rule inside B (Case 1); every boundary vertex of a
    //      D cluster adds one edge to each adjacent cluster (Case 2)
    class FGVCompact
    {
        public:
            FGVCompact(const GPUGraph& graph,int k,int alpha = 10);
            void run();
            const std::vector<Edge>& edges() const noexcept;
            int num_centers() const noexcept;   // high-coverage centers accepted by HCIS-r
            int num_residual() const noexcept;  // vertices handed over to FGV

        private:
            const GPUGraph& graph_;
            int k_;
            int alpha_;
            std::vector<Edge> edges_;
            int num_centers_ = 0;
            int num_residual_ = 0;
    };
}
