// HCIS-r (Algorithms 1 and 2 of the paper) with the alpha / beta modification of Sec. 4.1.
//
// Differences from a straightforward port (all semantics-preserving):
//   * coverage is 64-bit (32-bit overflows for r >= 3 on graphs with hubs);
//   * kernels run over a compacted list of the still-active vertices, so late phases are cheap;
//   * ActiveNbrs is decremented incrementally when vertices are removed (Alg. 1, step 13),
//     instead of being recomputed from scratch every phase;
//   * all buffers are allocated once; the r-hop "out" marking uses an epoch stamp instead of
//     per-phase allocation + memset; no per-kernel device synchronization.
#include "gpu/compact_kernels.hpp"
#include "compact_internal.hpp"

#include <cuda_runtime.h>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace
{
    using spanner::detail::DeviceBuffer;
    using spanner::detail::check_cuda;
    using spanner::detail::grid_size;
    using spanner::detail::kBlockSize;

    // mark[v] == phase * 16 + level  <=>  v is within `level` hops of a center chosen in this phase.
    constexpr int kLevelRadix = 16;

    struct IsActive
    {
        const int* active;

        __host__ __device__ bool operator()(int v) const
        {
            return active[v] != 0;
        }
    };

    __global__ void init_kernel(int n,const int* offsets,int* active,int* active_nbrs,int* mark)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n)
        {
            return;
        }

        active[v] = 1;
        active_nbrs[v] = offsets[v + 1] - offsets[v];   // Alg. 1, step 3: ActiveNbrs(v) = deg(v)
        mark[v] = 0;
    }

    __global__ void cov1_kernel(int count,const int* list,const int* active_nbrs,long long* cov)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < count)
        {
            const int v = list[i];
            cov[v] = active_nbrs[v];
        }
    }

    // Alg. 2, step 5: Cov_i[v] = sum over active neighbors u of Cov_{i-1}[u]
    __global__ void cov_next_kernel(int count,const int* list,const int* offsets,const int* neighbors,const int* active,const long long* previous,long long* next)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int v = list[i];
        long long sum = 0;

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            const int u = neighbors[e];

            if(active[u] != 0)
            {
                sum += previous[u];
            }
        }

        next[v] = sum;
    }

    __global__ void max_init_kernel(int count,const int* list,const long long* cov,long long* maximum,int* owner)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < count)
        {
            const int v = list[i];
            maximum[v] = cov[v];
            owner[v] = v;
        }
    }

    // Alg. 2, steps 11-19: propagate the (coverage, smaller-id-wins) champion one more hop.
    __global__ void max_step_kernel(int count,const int* list,const int* offsets,const int* neighbors,const int* active,const long long* previous_max,const int* previous_owner,long long* next_max,int* next_owner)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int v = list[i];
        long long best = previous_max[v];
        int best_owner = previous_owner[v];

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            const int u = neighbors[e];

            if(active[u] == 0)
            {
                continue;
            }

            const long long value = previous_max[u];
            const int owner = previous_owner[u];

            if(value > best || (value == best && owner < best_owner))
            {
                best = value;
                best_owner = owner;
            }
        }

        next_max[v] = best;
        next_owner[v] = best_owner;
    }

    __global__ void select_kernel(int count,const int* list,int phase,const long long* cov,const long long* maximum,const int* owner,int* is_center,int* mark,int* selected_count)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int v = list[i];

        if(owner[v] == v && maximum[v] == cov[v])
        {
            is_center[v] = 1;
            mark[v] = phase * kLevelRadix;   // level 0
            atomicAdd(selected_count,1);
        }
    }

    // One BFS level (over the whole graph G, as in Alg. 1 step 10).
    __global__ void bfs_expand_kernel(int n,int level,int phase,const int* offsets,const int* neighbors,int* mark)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || mark[v] != phase * kLevelRadix + (level - 1))
        {
            return;
        }

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            const int u = neighbors[e];

            if(mark[u] / kLevelRadix != phase)
            {
                mark[u] = phase * kLevelRadix + level;   // identical value from every writer
            }
        }
    }

    // Sec. 4.1: boundary edges = edges between BFS level r and r+1 around the selected centers.
    __global__ void count_boundary_kernel(int n,int radius,int phase,const int* offsets,const int* neighbors,const int* mark,unsigned long long* total)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || mark[v] != phase * kLevelRadix + radius)
        {
            return;
        }

        unsigned long long local = 0;

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            if(mark[neighbors[e]] == phase * kLevelRadix + radius + 1)
            {
                ++local;
            }
        }

        if(local > 0)
        {
            atomicAdd(total,local);
        }
    }

    // Alg. 1, steps 9-13: everything within r hops becomes inactive; neighbors' ActiveNbrs are decremented.
    __global__ void deactivate_kernel(int count,const int* list,int phase,int radius,const int* offsets,const int* neighbors,const int* mark,int* active,int* active_nbrs)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int v = list[i];
        const int m = mark[v];

        if(m / kLevelRadix == phase && m % kLevelRadix <= radius)
        {
            active[v] = 0;

            for(int e = offsets[v];e < offsets[v + 1];++e)
            {
                atomicSub(&active_nbrs[neighbors[e]],1);
            }
        }
    }

    // Sec. 4.1: a rejected batch is dropped (the centers only, not their r-hop neighborhood).
    __global__ void reject_kernel(int count,const int* list,int phase,const int* offsets,const int* neighbors,const int* mark,int* is_center,int* active,int* active_nbrs)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i >= count)
        {
            return;
        }

        const int v = list[i];

        if(mark[v] == phase * kLevelRadix)
        {
            is_center[v] = 0;
            active[v] = 0;

            for(int e = offsets[v];e < offsets[v + 1];++e)
            {
                atomicSub(&active_nbrs[neighbors[e]],1);
            }
        }
    }

    void launch_check(const char* message)
    {
        check_cuda(cudaGetLastError(),message);
    }
}

namespace spanner
{
    void run_hcis_r(int num_vertices,int radius,const int* offsets,const int* neighbors,int* is_center,int max_iterations,long long max_boundary_edges)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        if(radius < 1 || radius > 12)
        {
            throw std::invalid_argument("HCIS-r requires 1 <= radius <= 12");
        }

        static const bool verbose = std::getenv("SPANNER_VERBOSE") != nullptr;

        const int n = num_vertices;
        const int blocks = grid_size(n);
        const bool check_beta = max_boundary_edges >= 0;
        const int bfs_depth = check_beta ? radius + 1 : radius;

        DeviceBuffer<int> active(n),active_nbrs(n),mark(n),list_a(n),list_b(n),selected_count(1);
        DeviceBuffer<long long> cov_a(n),cov_b(n),max_a(n),max_b(n);
        DeviceBuffer<int> own_a(n),own_b(n);
        DeviceBuffer<unsigned long long> boundary(1);

        check_cuda(cudaMemset(is_center,0,static_cast<std::size_t>(n) * sizeof(int)),"Failed to clear HCIS centers");

        init_kernel<<<blocks,kBlockSize>>>(n,offsets,active.get(),active_nbrs.get(),mark.get());
        launch_check("HCIS initialization failed");

        thrust::sequence(thrust::device_pointer_cast(list_a.get()),thrust::device_pointer_cast(list_a.get()) + n);

        int* list = list_a.get();
        int* next_list = list_b.get();
        int active_count = n;

        for(int phase = 1;;++phase)
        {
            if(active_count == 0)
            {
                if(verbose)
                {
                    std::cout << "HCIS round " << phase << ": active=0 selected=0\n";
                }

                break;
            }

            if(max_iterations > 0 && phase > max_iterations)   // at most alpha phases
            {
                break;
            }

            const int wb = grid_size(active_count);

            // ---- CalcCoverage (Alg. 2)
            long long* cov = cov_a.get();
            long long* cov_spare = cov_b.get();

            cov1_kernel<<<wb,kBlockSize>>>(active_count,list,active_nbrs.get(),cov);

            for(int level = 2;level <= radius;++level)
            {
                cov_next_kernel<<<wb,kBlockSize>>>(active_count,list,offsets,neighbors,active.get(),cov,cov_spare);
                std::swap(cov,cov_spare);
            }

            long long* mx = max_a.get();
            long long* mx_spare = max_b.get();
            int* ow = own_a.get();
            int* ow_spare = own_b.get();

            max_init_kernel<<<wb,kBlockSize>>>(active_count,list,cov,mx,ow);

            for(int level = 1;level <= radius;++level)
            {
                max_step_kernel<<<wb,kBlockSize>>>(active_count,list,offsets,neighbors,active.get(),mx,ow,mx_spare,ow_spare);
                std::swap(mx,mx_spare);
                std::swap(ow,ow_spare);
            }

            // ---- select the r-hop coverage maxima
            check_cuda(cudaMemset(selected_count.get(),0,sizeof(int)),"Failed to reset HCIS selected count");

            select_kernel<<<wb,kBlockSize>>>(active_count,list,phase,cov,mx,ow,is_center,mark.get(),selected_count.get());
            launch_check("HCIS coverage/selection failed");

            int selected = 0;

            check_cuda(cudaMemcpy(&selected,selected_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read HCIS selected count");

            if(selected == 0)
            {
                throw std::runtime_error("HCIS selected zero centers while active vertices remain");
            }

            // ---- mark everything within r (and r+1, if beta is checked) hops of the new centers
            for(int level = 1;level <= bfs_depth;++level)
            {
                bfs_expand_kernel<<<blocks,kBlockSize>>>(n,level,phase,offsets,neighbors,mark.get());
            }

            launch_check("HCIS BFS failed");

            bool accept = true;
            unsigned long long boundary_edges = 0;

            if(check_beta)
            {
                check_cuda(cudaMemset(boundary.get(),0,sizeof(unsigned long long)),"Failed to reset boundary counter");

                count_boundary_kernel<<<blocks,kBlockSize>>>(n,radius,phase,offsets,neighbors,mark.get(),boundary.get());
                launch_check("HCIS boundary count failed");

                check_cuda(cudaMemcpy(&boundary_edges,boundary.get(),sizeof(boundary_edges),cudaMemcpyDeviceToHost),"Failed to read boundary count");

                accept = boundary_edges <= static_cast<unsigned long long>(max_boundary_edges);
            }

            if(accept)
            {
                deactivate_kernel<<<wb,kBlockSize>>>(active_count,list,phase,radius,offsets,neighbors,mark.get(),active.get(),active_nbrs.get());
            }
            else
            {
                reject_kernel<<<wb,kBlockSize>>>(active_count,list,phase,offsets,neighbors,mark.get(),is_center,active.get(),active_nbrs.get());
            }

            launch_check("HCIS deactivation failed");

            if(verbose)
            {
                std::cout << "HCIS round " << phase << ": active=" << active_count << " selected=" << selected;

                if(!accept)
                {
                    std::cout << " REJECTED (boundary edges=" << boundary_edges << " > beta=" << max_boundary_edges << ")";
                }

                std::cout << '\n';
            }

            // ---- keep only the still-active vertices for the next phase
            auto end = thrust::copy_if(thrust::device_pointer_cast(list),thrust::device_pointer_cast(list) + active_count,thrust::device_pointer_cast(next_list),IsActive{active.get()});

            active_count = static_cast<int>(end - thrust::device_pointer_cast(next_list));
            std::swap(list,next_list);
        }

        check_cuda(cudaDeviceSynchronize(),"HCIS-r execution failed");
    }
}
