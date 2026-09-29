#pragma once
#include <vector>
#include "graph/csr.hpp"
#include "graph/edge.hpp"

namespace spanner
{

struct GPUStretchResult
{
    bool passed = false;
    bool full_validation = false;
    int checked_sources = 0;
    int total_sources = 0;
    int checked_edge_incidences = 0;
    int maximum_stretch = 0;
    int violating_source = -1;
    int violating_destination = -1;
};

GPUStretchResult validate_gpu_stretch(const CSRGraph& graph,const std::vector<Edge>& spanner_edges,int stretch_bound,int max_sources);

}