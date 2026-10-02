#pragma once

#include <cstdint>
#include <vector>

#include "graph/edge.hpp"

namespace spanner
{
    // Canonicalize (source < destination), drop self-loops, sort and unique -- all on the GPU.
    // src/dst are device arrays holding `count` edges in any orientation.
    std::vector<Edge> finalize_edges(const int* src,const int* dst,int count);

    // Canonicalize already-unique device edges and copy them to the host without sorting.
    std::vector<Edge> copy_canonical_unique_edges(const int* src,const int* dst,int count);

    // Same, for edges already packed on the device as (min << 32) | max. `keys` is sorted in place.
    std::vector<Edge> finalize_packed_edges(std::uint64_t* keys,int count);
}
