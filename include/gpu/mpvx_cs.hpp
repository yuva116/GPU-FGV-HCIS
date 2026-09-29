#pragma once

#include <vector>

#include "graph/edge.hpp"

namespace spanner
{
    void run_mpvx_cs(const int* offsets,const int* neighbors,int num_vertices,int num_adjacency_entries,int k,std::vector<Edge>& edges,int& num_centers,int& num_residual);
}