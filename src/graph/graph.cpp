#include "graph/graph.hpp"
#include <stdexcept>

namespace spanner 
{
Graph::Graph(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed): num_vertices_(num_vertices),num_edges_(edges.size()),directed_(directed),edges_(edges) 
{

    for (const Edge& edge : edges_) {
        if (edge.source < 0 || edge.destination < 0) {
            throw std::out_of_range("Vertex ID cannot be negative");
        }

        if (!valid_vertex(edge.source) ||!valid_vertex(edge.destination)) {
            throw std::out_of_range("Edge contains a vertex outside the graph");
        }
    }
}

std::size_t Graph::num_vertices() const noexcept {
    return num_vertices_;
}

std::size_t Graph::num_edges() const noexcept {
    return num_edges_;
}

bool Graph::directed() const noexcept {
    return directed_;
}

const std::vector<Edge>& Graph::edges() const noexcept {
    return edges_;
}

bool Graph::valid_vertex(std::size_t vertex) const noexcept {
    return vertex < num_vertices_;
}

} 