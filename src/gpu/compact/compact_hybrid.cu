// Hybrid FGV-CS spanner (Sec. 4.3 of the paper) + helpers for the modified HCIS-r (Sec. 4.1).
#include "algorithms/compact/fgv_compact.hpp"
#include "gpu/compact_kernels.hpp"
#include "gpu/kernels.hpp"
#include "compact_internal.hpp"

#include <cuda_runtime.h>

#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace
{
    using spanner::detail::kBlockSize;

    // ---------------------------------------------------------------- BFS growth
    __global__ void init_bfs_kernel(int n,const int* is_center,int* dist,int* cen,int* par)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n)
        {
            return;
        }

        const bool c = is_center[v] != 0;

        dist[v] = c ? 0 : -1;
        cen[v] = c ? v : -1;
        par[v] = c ? v : -1;
    }

    // Level-synchronous: only vertices at level-1 are read, and they are never written in this launch.
    __global__ void bfs_level_kernel(int n,int level,const int* offsets,const int* neighbors,int* dist,int* cen,int* par,int* changed)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || dist[v] >= 0)
        {
            return;
        }

        int best = -1;
        int best_center = INT_MAX;

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            const int u = neighbors[e];

            if(dist[u] == level - 1 && cen[u] < best_center)
            {
                best_center = cen[u];
                best = u;
            }
        }

        if(best >= 0)
        {
            dist[v] = level;
            cen[v] = best_center;
            par[v] = best;
            *changed = 1;
        }
    }

    __global__ void count_boundary_kernel(int n,int radius,const int* offsets,const int* neighbors,const int* dist,unsigned long long* total)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || dist[v] != radius)
        {
            return;
        }

        unsigned long long local = 0;

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            if(dist[neighbors[e]] == radius + 1)
            {
                ++local;
            }
        }

        if(local > 0)
        {
            atomicAdd(total,local);
        }
    }

    // ---------------------------------------------------------------- residual graph (vertices not in a D cluster)
    __global__ void flag_residual_kernel(int n,const int* dist,int* flag)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v < n)
        {
            flag[v] = dist[v] < 0 ? 1 : 0;
        }
    }

    __global__ void residual_degree_kernel(int n,const int* offsets,const int* neighbors,const int* flag,int* deg)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n)
        {
            return;
        }

        int d = 0;

        if(flag[v])
        {
            for(int e = offsets[v];e < offsets[v + 1];++e)
            {
                d += flag[neighbors[e]];
            }
        }

        deg[v] = d;
    }

    __global__ void residual_build_kernel(int n,const int* offsets,const int* neighbors,const int* flag,const int* newid,const int* pos,int* res_off,int* res_nbrs,int* old_of_new)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || !flag[v])
        {
            return;
        }

        const int i = newid[v];
        int w = pos[v];

        res_off[i] = pos[v];
        old_of_new[i] = v;

        for(int e = offsets[v];e < offsets[v + 1];++e)
        {
            const int u = neighbors[e];

            if(flag[u])
            {
                res_nbrs[w++] = newid[u];
            }
        }
    }

    // ---------------------------------------------------------------- final cluster ids + edges
    __global__ void assemble_cluster_kernel(int n,const int* dist,const int* cen,const int* newid,const int* res_cen,const int* old_of_new,int* cluster)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n)
        {
            return;
        }

        cluster[v] = dist[v] >= 0 ? cen[v] : old_of_new[res_cen[newid[v]]];
    }

    __global__ void tree_edges_kernel(int n,const int* dist,const int* par,int* count,int* src,int* dst)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n || dist[v] <= 0 || par[v] < 0 || par[v] == v)
        {
            return;
        }

        const int p = atomicAdd(count,1);

        src[p] = v;
        dst[p] = par[v];
    }

    // Case 2: boundary vertex u of a D cluster -> candidate edge into each adjacent cluster.
    __global__ void case2_candidates_kernel(int n,const int* offsets,const int* neighbors,const int* dist,const int* cluster,std::uint64_t* keys,int* dsts)
    {
        const int u = blockIdx.x * blockDim.x + threadIdx.x;

        if(u >= n || dist[u] < 0)
        {
            return;
        }

        for(int e = offsets[u];e < offsets[u + 1];++e)
        {
            const int w = neighbors[e];

            if(cluster[w] != cluster[u])
            {
                keys[e] = static_cast<std::uint64_t>(u) * static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(cluster[w]);
                dsts[e] = w;
            }
        }
    }

    void sync(const char* what)
    {
        spanner::detail::check_cuda(cudaGetLastError(),what);
        spanner::detail::check_cuda(cudaDeviceSynchronize(),what);
    }

    template<typename T>
    thrust::device_ptr<T> dp(T* ptr)
    {
        return thrust::device_pointer_cast(ptr);
    }
}

namespace spanner
{
    using detail::DeviceBuffer;
    using detail::check_cuda;
    using detail::grid_size;

    void grow_compact_bfs(int num_vertices,int depth,const int* offsets,const int* neighbors,const int* is_center,int* distances,int* centers,int* parents)
    {
        if(num_vertices <= 0)
        {
            return;
        }

        const int blocks = grid_size(num_vertices);
        DeviceBuffer<int> changed(1);

        init_bfs_kernel<<<blocks,kBlockSize>>>(num_vertices,is_center,distances,centers,parents);
        sync("BFS initialization failed");

        for(int level = 1;level <= depth;++level)
        {
            check_cuda(cudaMemset(changed.get(),0,sizeof(int)),"Failed to reset BFS flag");

            bfs_level_kernel<<<blocks,kBlockSize>>>(num_vertices,level,offsets,neighbors,distances,centers,parents,changed.get());
            sync("BFS level failed");

            int host_changed = 0;

            check_cuda(cudaMemcpy(&host_changed,changed.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read BFS flag");

            if(host_changed == 0)
            {
                break;
            }
        }
    }

    long long count_compact_boundary_edges(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* seeds)
    {
        DeviceBuffer<int> dist(num_vertices),cen(num_vertices),par(num_vertices);
        DeviceBuffer<unsigned long long> total(1);

        grow_compact_bfs(num_vertices,radius + 1,offsets,neighbors,seeds,dist.get(),cen.get(),par.get());

        check_cuda(cudaMemset(total.get(),0,sizeof(unsigned long long)),"Failed to reset boundary counter");

        count_boundary_kernel<<<grid_size(num_vertices),kBlockSize>>>(num_vertices,radius,offsets,neighbors,dist.get(),total.get());
        sync("Boundary edge count failed");

        unsigned long long host_total = 0;

        check_cuda(cudaMemcpy(&host_total,total.get(),sizeof(host_total),cudaMemcpyDeviceToHost),"Failed to read boundary count");

        return static_cast<long long>(host_total);
    }

    FGVCompact::FGVCompact(const GPUGraph& graph,int k,int alpha)
        : graph_(graph),k_(k),alpha_(alpha)
    {
        if(k_ < 2)
        {
            throw std::invalid_argument("FGVCompact requires k >= 2");
        }

        if(alpha_ < 1)
        {
            throw std::invalid_argument("FGVCompact requires alpha >= 1");
        }
    }

    void FGVCompact::run()
    {
        edges_.clear();
        num_centers_ = 0;
        num_residual_ = 0;

        const int n = static_cast<int>(graph_.num_vertices());
        const int adj = static_cast<int>(graph_.num_adjacency_entries());

        if(n <= 0)
        {
            return;
        }

        const int r = k_ - 1;
        const int* offsets = graph_.offsets();
        const int* neighbors = graph_.neighbors();
        const int blocks = grid_size(n);

        std::vector<Edge> raw;

        // ---- Stage 1 + 2: high-coverage centers and their radius-r clusters (set D)
        DeviceBuffer<int> is_center(n),dist(n),cen(n),par(n);

        const double beta_double = std::pow(static_cast<double>(n),1.0 + 1.0 / static_cast<double>(k_));
        const long long beta = static_cast<long long>(std::min(beta_double,9.0e18));

        run_hcis_r(n,r,offsets,neighbors,is_center.get(),alpha_,beta);
        sync("Modified HCIS-r failed");

        num_centers_ = thrust::reduce(dp(is_center.get()),dp(is_center.get()) + n,0);

        grow_compact_bfs(n,r,offsets,neighbors,is_center.get(),dist.get(),cen.get(),par.get());

        // D tree edges
        {
            DeviceBuffer<int> count(1),src(n),dst(n);

            check_cuda(cudaMemset(count.get(),0,sizeof(int)),"Failed to reset tree count");
            tree_edges_kernel<<<blocks,kBlockSize>>>(n,dist.get(),par.get(),count.get(),src.get(),dst.get());
            sync("D tree edges failed");

            int host_count = 0;
            check_cuda(cudaMemcpy(&host_count,count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read tree count");

            std::vector<int> hs(host_count),hd(host_count);

            if(host_count > 0)
            {
                check_cuda(cudaMemcpy(hs.data(),src.get(),host_count * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy tree sources");
                check_cuda(cudaMemcpy(hd.data(),dst.get(),host_count * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy tree destinations");
            }

            for(int i = 0;i < host_count;++i)
            {
                raw.push_back(Edge{hs[i],hd[i]});
            }
        }

        // ---- Stage 3: FGV-Spanner on the vertices not covered by D (set B)
        DeviceBuffer<int> cluster(n);
        DeviceBuffer<int> flag(n),newid(n);

        flag_residual_kernel<<<blocks,kBlockSize>>>(n,dist.get(),flag.get());
        sync("Residual flag failed");

        thrust::exclusive_scan(dp(flag.get()),dp(flag.get()) + n,dp(newid.get()));

        int last_flag = 0,last_id = 0;
        check_cuda(cudaMemcpy(&last_flag,flag.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual flag");
        check_cuda(cudaMemcpy(&last_id,newid.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual id");

        const int n_res = last_id + last_flag;
        num_residual_ = n_res;

        if(n_res == 0)
        {
            check_cuda(cudaMemcpy(cluster.get(),cen.get(),static_cast<std::size_t>(n) * sizeof(int),cudaMemcpyDeviceToDevice),"Failed to copy clusters");
        }
        else
        {
            DeviceBuffer<int> deg(n),pos(n);

            residual_degree_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,flag.get(),deg.get());
            sync("Residual degree failed");

            thrust::exclusive_scan(dp(deg.get()),dp(deg.get()) + n,dp(pos.get()));

            int last_deg = 0,last_pos = 0;
            check_cuda(cudaMemcpy(&last_deg,deg.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual degree");
            check_cuda(cudaMemcpy(&last_pos,pos.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual position");

            const int res_adj = last_pos + last_deg;

            DeviceBuffer<int> res_off(n_res + 1),res_nbrs(std::max(1,res_adj)),old_of_new(n_res);

            residual_build_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,flag.get(),newid.get(),pos.get(),res_off.get(),res_nbrs.get(),old_of_new.get());
            sync("Residual graph build failed");

            check_cuda(cudaMemcpy(res_off.get() + n_res,&res_adj,sizeof(int),cudaMemcpyHostToDevice),"Failed to finish residual offsets");

            // Same steps as FGV::run(), on the residual graph.
            const float probability = static_cast<float>(1.0 - std::pow(static_cast<double>(n_res),-1.0 / static_cast<double>(k_)));
            const int res_max_edges = n_res + res_adj;

            DeviceBuffer<int> shifts(n_res),r_dist(n_res),r_cen(n_res),r_par(n_res);
            DeviceBuffer<int> f_count(1),f_src(res_max_edges),f_dst(res_max_edges);

            initialize_fgv(n_res,r,probability,shifts.get(),r_dist.get(),r_cen.get(),r_par.get());
            build_fgv_clusters(n_res,r,res_off.get(),res_nbrs.get(),shifts.get(),r_dist.get(),r_cen.get(),r_par.get());
            build_fgv_spanner(n_res,res_off.get(),res_nbrs.get(),r_dist.get(),r_cen.get(),r_par.get(),f_count.get(),f_src.get(),f_dst.get(),res_max_edges);

            int host_count = 0;
            check_cuda(cudaMemcpy(&host_count,f_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual edge count");

            if(host_count < 0 || host_count > res_max_edges)
            {
                throw std::runtime_error("Residual FGV produced an invalid edge count");
            }

            std::vector<int> hs(host_count),hd(host_count),map(n_res);

            if(host_count > 0)
            {
                check_cuda(cudaMemcpy(hs.data(),f_src.get(),host_count * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy residual sources");
                check_cuda(cudaMemcpy(hd.data(),f_dst.get(),host_count * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy residual destinations");
            }

            check_cuda(cudaMemcpy(map.data(),old_of_new.get(),n_res * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy residual id map");

            for(int i = 0;i < host_count;++i)
            {
                raw.push_back(Edge{map[hs[i]],map[hd[i]]});
            }

            assemble_cluster_kernel<<<blocks,kBlockSize>>>(n,dist.get(),cen.get(),newid.get(),r_cen.get(),old_of_new.get(),cluster.get());
            sync("Cluster assembly failed");
        }

        // ---- Stage 4 (Case 2): one edge from each D boundary vertex into every adjacent cluster
        if(adj > 0)
        {
            DeviceBuffer<std::uint64_t> keys(adj);
            DeviceBuffer<int> dsts(adj);

            check_cuda(cudaMemset(keys.get(),0xFF,static_cast<std::size_t>(adj) * sizeof(std::uint64_t)),"Failed to init case-2 keys");

            case2_candidates_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,dist.get(),cluster.get(),keys.get(),dsts.get());
            sync("Case-2 candidates failed");

            thrust::sort_by_key(dp(keys.get()),dp(keys.get()) + adj,dp(dsts.get()));

            auto unique_end = thrust::unique_by_key(dp(keys.get()),dp(keys.get()) + adj,dp(dsts.get()));

            int valid = static_cast<int>(unique_end.first - dp(keys.get()));

            if(valid > 0)
            {
                std::uint64_t last_key = 0;

                check_cuda(cudaMemcpy(&last_key,keys.get() + valid - 1,sizeof(last_key),cudaMemcpyDeviceToHost),"Failed to read last case-2 key");

                if(last_key == UINT64_MAX)
                {
                    --valid;
                }
            }

            if(valid > 0)
            {
                std::vector<std::uint64_t> hk(valid);
                std::vector<int> hd(valid);

                check_cuda(cudaMemcpy(hk.data(),keys.get(),static_cast<std::size_t>(valid) * sizeof(std::uint64_t),cudaMemcpyDeviceToHost),"Failed to copy case-2 keys");
                check_cuda(cudaMemcpy(hd.data(),dsts.get(),static_cast<std::size_t>(valid) * sizeof(int),cudaMemcpyDeviceToHost),"Failed to copy case-2 destinations");

                for(int i = 0;i < valid;++i)
                {
                    raw.push_back(Edge{static_cast<int>(hk[i] / static_cast<std::uint64_t>(n)),hd[i]});
                }
            }
        }

        // ---- Canonicalize, sort, unique (same as FGV / Compact)
        edges_.reserve(raw.size());

        for(Edge e : raw)
        {
            if(e.source == e.destination)
            {
                continue;
            }

            if(e.source > e.destination)
            {
                std::swap(e.source,e.destination);
            }

            edges_.push_back(e);
        }

        std::sort(edges_.begin(),edges_.end(),[](const Edge& a,const Edge& b)
        {
            return a.source != b.source ? a.source < b.source : a.destination < b.destination;
        });

        edges_.erase(std::unique(edges_.begin(),edges_.end(),[](const Edge& a,const Edge& b)
        {
            return a.source == b.source && a.destination == b.destination;
        }),edges_.end());
    }

    const std::vector<Edge>& FGVCompact::edges() const noexcept
    {
        return edges_;
    }

    int FGVCompact::num_centers() const noexcept
    {
        return num_centers_;
    }

    int FGVCompact::num_residual() const noexcept
    {
        return num_residual_;
    }
}
