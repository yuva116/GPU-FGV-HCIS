#include "graph/csr.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace spanner {

CSRGraph::CSRGraph(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed) {
    build(num_vertices, edges, directed);
}

void CSRGraph::build(std::size_t num_vertices,const std::vector<Edge>& edges,bool directed) {
    num_vertices_ = num_vertices;
    num_edges_ = edges.size();
    directed_ = directed;
    offsets_.assign(num_vertices_ + 1, 0);

    // Count the number of neighbors for every vertex.
    for (const Edge& edge : edges) {
        if (edge.source < 0 || edge.destination < 0) {
            throw std::out_of_range(
                "Vertex ID cannot be negative"
            );
        }

        if (!valid_vertex(edge.source) ||
            !valid_vertex(edge.destination)) {
            throw std::out_of_range(
                "Edge contains a vertex outside the graph"
            );
        }

        ++offsets_[static_cast<std::size_t>(edge.source) + 1];

        if (!directed_) {
            ++offsets_[
                static_cast<std::size_t>(edge.destination) + 1
            ];
        }
    }

    // Convert neighbor counts into offsets.
    for (std::size_t vertex = 1;
         vertex <= num_vertices_;
         ++vertex) {
        offsets_[vertex] += offsets_[vertex - 1];
    }

    num_adjacency_entries_ = offsets_[num_vertices_];

    neighbors_.resize(num_adjacency_entries_);

    // Store the neighbors.
    std::vector<std::size_t> next_position = offsets_;

    for (const Edge& edge : edges) {
        const std::size_t source =
            static_cast<std::size_t>(edge.source);

        const std::size_t destination =
            static_cast<std::size_t>(edge.destination);

        neighbors_[next_position[source]] = edge.destination;
        ++next_position[source];

        if (!directed_) {
            neighbors_[next_position[destination]] = edge.source;
            ++next_position[destination];
        }
    }

    // Keep each adjacency list sorted.
    for (std::size_t vertex = 0;vertex < num_vertices_;++vertex) {

        auto begin = neighbors_.begin() + offsets_[vertex];
        auto end = neighbors_.begin() + offsets_[vertex + 1];

        std::sort(begin, end);
    }
}

std::size_t CSRGraph::num_vertices() const noexcept {
    return num_vertices_;
}

std::size_t CSRGraph::num_edges() const noexcept {
    return num_edges_;
}

std::size_t CSRGraph::num_adjacency_entries() const noexcept {
    return num_adjacency_entries_;
}

std::size_t CSRGraph::degree(std::size_t vertex) const {
    if (!valid_vertex(vertex)) {
        throw std::out_of_range("Invalid vertex ID");
    }

    return offsets_[vertex + 1] - offsets_[vertex];
}

std::size_t CSRGraph::neighbor_count(std::size_t vertex) const {
    return degree(vertex);
}

const int* CSRGraph::neighbors(std::size_t vertex) const {
    if (!valid_vertex(vertex)) {
        throw std::out_of_range("Invalid vertex ID");
    }
    return neighbors_.data() + offsets_[vertex];
}

const std::vector<std::size_t>&
CSRGraph::offsets() const noexcept {
    return offsets_;
}

const std::vector<int>&
CSRGraph::neighbors_data() const noexcept {
    return neighbors_;
}

bool CSRGraph::directed() const noexcept {
    return directed_;
}

bool CSRGraph::valid_vertex(std::size_t vertex) const noexcept {
    return vertex < num_vertices_;
}

}