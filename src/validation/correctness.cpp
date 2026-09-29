#include "validation/correctness.hpp"

namespace spanner::validation
{

bool validate_csr(const CSRGraph& graph) {
    const auto& offsets = graph.offsets();
    const auto& neighbors = graph.neighbors_data();

    if (offsets.size() != graph.num_vertices() + 1) {
        return false;
    }

    if (offsets.empty() || offsets.front() != 0) {
        return false;
    }

    for (std::size_t vertex = 0; vertex < graph.num_vertices(); ++vertex) {

        if (offsets[vertex] > offsets[vertex + 1]) {
            return false;
        }
    }

    if (offsets.back() != neighbors.size()) {
        return false;
    }

    for (int neighbor : neighbors) {
        if (neighbor < 0) {
            return false;
        }

        if (static_cast<std::size_t>(neighbor) >= graph.num_vertices()) {
            return false;
        }
    }

    return true;
}

} 