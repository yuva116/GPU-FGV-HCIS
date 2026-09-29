#pragma once
#include <cstddef>
#include "graph/csr.hpp"

namespace spanner
{
    class GPUGraph
    {
        public:
            explicit GPUGraph(const CSRGraph& graph);
            ~GPUGraph();
            GPUGraph(const GPUGraph&) = delete;
            GPUGraph& operator=(const GPUGraph&) = delete;

            std::size_t num_vertices() const noexcept;
            std::size_t num_edges()  const noexcept;
            std::size_t num_adjacency_entries() const noexcept;
            const int* offsets() const noexcept;
            const int* neighbors() const noexcept;

        private:
            std::size_t num_vertices_ = 0;
            std::size_t num_edges_ = 0;
            std::size_t num_adjacency_entries_ = 0;

            int* d_offsets_ = nullptr;
            int* d_neighbors_ = nullptr;

    };
}