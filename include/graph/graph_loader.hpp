#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "graph/edge.hpp"

namespace spanner {

struct GraphData {
    std::size_t num_vertices = 0;
    std::vector<Edge> edges;
};

class GraphLoader {
public:
    static GraphData load_edge_list(const std::string& filename);
    static GraphData load_snap(const std::string& filename);
};

}