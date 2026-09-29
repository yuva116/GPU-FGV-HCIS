#pragma once
#include <cstddef>
#include <vector>
#include "graph/edge.hpp"

namespace spanner {

class Graph {
public:
    Graph() = default;
    Graph(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed = false);

    std::size_t num_vertices() const noexcept;
    std::size_t num_edges() const noexcept;
    bool directed() const noexcept;
    const std::vector<Edge>& edges() const noexcept;
    bool valid_vertex(std::size_t vertex) const noexcept;

private:
    std::size_t num_vertices_ = 0;
    std::size_t num_edges_ = 0;

    bool directed_ = false;

    std::vector<Edge> edges_;
};

} 