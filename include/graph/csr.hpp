#pragma once

#include <cstddef>
#include <vector>

#include "graph/edge.hpp"

namespace spanner {

class CSRGraph {
public:
    CSRGraph() = default;

    CSRGraph(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed= false);

    std::size_t num_vertices() const noexcept;
    std::size_t num_edges() const noexcept;
    std::size_t num_adjacency_entries() const noexcept;
    std::size_t degree(std::size_t vertex) const;
    std::size_t neighbor_count(std::size_t vertex) const;
    const int* neighbors(std::size_t vertex) const;
    const std::vector<std::size_t>& offsets() const noexcept;
    const std::vector<int>& neighbors_data() const noexcept;

    bool directed() const noexcept;
    bool valid_vertex(std::size_t vertex) const noexcept;

private:
    std::size_t num_vertices_ = 0;
    std::size_t num_edges_ = 0;
    std::size_t num_adjacency_entries_ = 0;

    bool directed_ = false;

    std::vector<std::size_t> offsets_;
    std::vector<int> neighbors_;

    void build(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed);
};

}