#include "gpu/edge_finalize.hpp"
#include "compact/compact_internal.hpp"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace
{
    __global__ void pack_edges_kernel(int count,const int* src,const int* dst,std::uint64_t* keys)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int a = src[i];
        const int b = dst[i];

        if(a == b || a < 0 || b < 0)
        {
            keys[i] = UINT64_MAX;
            return;
        }

        const int lo = a < b ? a : b;
        const int hi = a < b ? b : a;

        keys[i] = (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
    }
}

namespace spanner
{
    std::vector<Edge> finalize_packed_edges(std::uint64_t* keys,int count)
    {
        std::vector<Edge> edges;

        if(count <= 0)
        {
            return edges;
        }

        auto first = thrust::device_pointer_cast(keys);

        thrust::sort(first,first + count);

        auto last = thrust::unique(first,first + count);

        int unique_count = static_cast<int>(last - first);

        if(unique_count > 0)
        {
            std::uint64_t tail = 0;

            detail::check_cuda(cudaMemcpy(&tail,keys + unique_count - 1,sizeof(tail),cudaMemcpyDeviceToHost),"Failed to read last edge key");

            if(tail == UINT64_MAX)
            {
                --unique_count;   // the self-loop sentinel sorts last
            }
        }

        std::vector<std::uint64_t> host(static_cast<std::size_t>(unique_count));

        if(unique_count > 0)
        {
            detail::check_cuda(cudaMemcpy(host.data(),keys,static_cast<std::size_t>(unique_count) * sizeof(std::uint64_t),cudaMemcpyDeviceToHost),"Failed to copy edge keys");
        }

        edges.resize(host.size());

        for(std::size_t i = 0;i < host.size();++i)
        {
            edges[i] = Edge{static_cast<int>(host[i] >> 32),static_cast<int>(host[i] & 0xFFFFFFFFULL)};
        }

        return edges;
    }

    std::vector<Edge> finalize_edges(const int* src,const int* dst,int count)
    {
        if(count <= 0)
        {
            return {};
        }

        detail::DeviceBuffer<std::uint64_t> keys(static_cast<std::size_t>(count));

        pack_edges_kernel<<<detail::grid_size(count),detail::kBlockSize>>>(count,src,dst,keys.get());

        detail::check_cuda(cudaGetLastError(),"Edge packing launch failed");

        return finalize_packed_edges(keys.get(),count);
    }
}
