// CS (Sec. 3.3) and hybrid FGV-CS (Sec. 4.3) spanner pipelines.
//
// Everything stays on the GPU until the final, already sorted and de-duplicated edge list.
//
// Inter-cluster edges (shared by CS and FGV-CS), for every pair of adjacent clusters A, B with
// tree depths ecc[A], ecc[B]:
//   * if 2*ecc[A] + 2*ecc[B] + 1 <= stretch, ONE edge between A and B already gives every A-B edge a
//     path of length <= stretch (this is the "single edge if the diameters allow it" rule of Sec. 5.2);
//   * otherwise every boundary vertex u adds one edge into each adjacent cluster C, but only if
//     u may act as a source (MPVX-Rule: cid(u) < cid(C); in the hybrid all high-coverage (D) vertices
//     are sources towards residual (B) clusters, exactly Case 2 of Sec. 4.3). Lemma 2.1 only needs one
//     of the two endpoints of every inter-cluster edge to add an edge, so D-D pairs use the id rule.
#include "algorithms/compact/fgv_compact.hpp"
#include "gpu/compact_kernels.hpp"
#include "gpu/edge_finalize.hpp"
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

    // ---------------------------------------------------------------- BFS growth (race-free, level synchronous)
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

    // Only vertices at level-1 are read, and they are never written in this launch.
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

    // Safety net: HCIS-r is maximal, so this should never trigger.
    __global__ void fix_unreached_kernel(int n,int* dist,int* cen,int* par)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v < n && dist[v] < 0)
        {
            dist[v] = 0;
            cen[v] = v;
            par[v] = v;
        }
    }

    // ---------------------------------------------------------------- edge sink (packed (min<<32)|max)
    __device__ __forceinline__ void append_edge(std::uint64_t* sink,int* count,int capacity,int a,int b)
    {
        if(a == b)
        {
            return;
        }

        const int lo = a < b ? a : b;
        const int hi = a < b ? b : a;
        const int p = atomicAdd(count,1);

        if(p < capacity)
        {
            sink[p] = (static_cast<std::uint64_t>(lo) << 32) | static_cast<std::uint64_t>(hi);
        }
    }

    __global__ void tree_append_kernel(int n,const int* dist,const int* par,std::uint64_t* sink,int* count,int capacity)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v < n && dist[v] > 0 && par[v] >= 0 && par[v] != v)
        {
            append_edge(sink,count,capacity,v,par[v]);
        }
    }

    __global__ void ecc_from_bfs_kernel(int n,const int* dist,const int* cen,int* ecc)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v < n && dist[v] >= 0)
        {
            atomicMax(&ecc[cen[v]],dist[v]);
        }
    }

    // ---------------------------------------------------------------- inter-cluster edges
    __global__ void inter_candidates_kernel(int n,const int* offsets,const int* neighbors,const int* cluster,const int* source_mask,const int* ecc,int stretch,std::uint64_t* keys,std::uint64_t* vals,int* candidate_count)
    {
        const int u = blockIdx.x * blockDim.x + threadIdx.x;

        if(u >= n)
        {
            return;
        }

        const int cu = cluster[u];

        if(cu < 0)
        {
            return;
        }

        const bool u_source = source_mask == nullptr || source_mask[u] != 0;

        for(int e = offsets[u];e < offsets[u + 1];++e)
        {
            const int w = neighbors[e];
            const int cw = cluster[w];

            if(cw < 0 || cw == cu)
            {
                continue;
            }

            const bool w_source = source_mask == nullptr || source_mask[w] != 0;

            if(!u_source && !w_source)
            {
                continue;   // residual-residual pairs are handled by FGV's own FGV-Rule (Case 1)
            }

            std::uint64_t key;

            if(2 * ecc[cu] + 2 * ecc[cw] + 1 <= stretch)
            {
                if(cu > cw)
                {
                    continue;   // one direction records the pair
                }

                key = (1ULL << 63) | (static_cast<std::uint64_t>(cu) * static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(cw));
            }
            else
            {
                if(!u_source || (w_source && cu > cw))
                {
                    continue;
                }

                key = static_cast<std::uint64_t>(u) * static_cast<std::uint64_t>(n) + static_cast<std::uint64_t>(cw);
            }

            const int pos = atomicAdd(candidate_count,1);

            keys[pos] = key;
            vals[pos] = (static_cast<std::uint64_t>(u) << 32) | static_cast<std::uint64_t>(w);
        }
    }

    __global__ void inter_emit_kernel(int count,const std::uint64_t* vals,std::uint64_t* sink,int* sink_count,int capacity)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < count)
        {
            append_edge(sink,sink_count,capacity,static_cast<int>(vals[i] >> 32),static_cast<int>(vals[i] & 0xFFFFFFFFULL));
        }
    }

    // ---------------------------------------------------------------- residual graph (hybrid only)
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

    __global__ void residual_edges_append_kernel(const int* count_ptr,const int* f_src,const int* f_dst,const int* old_of_new,std::uint64_t* sink,int* sink_count,int capacity)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < *count_ptr)
        {
            append_edge(sink,sink_count,capacity,old_of_new[f_src[i]],old_of_new[f_dst[i]]);
        }
    }

    // FGV cluster tree depth of i = level(i) - level(center); recorded per (original) center id.
    __global__ void ecc_from_fgv_kernel(int n_res,const int* r_dist,const int* r_cen,const int* old_of_new,int* ecc)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;

        if(i < n_res)
        {
            const int c = r_cen[i];
            atomicMax(&ecc[old_of_new[c]],r_dist[i] - r_dist[c]);
        }
    }

    __global__ void assemble_cluster_kernel(int n,const int* dist,const int* cen,const int* newid,const int* res_cen,const int* old_of_new,int* cluster,int* d_mask)
    {
        const int v = blockIdx.x * blockDim.x + threadIdx.x;

        if(v >= n)
        {
            return;
        }

        const bool in_d = dist[v] >= 0;

        cluster[v] = in_d ? cen[v] : old_of_new[res_cen[newid[v]]];
        d_mask[v] = in_d ? 1 : 0;
    }

    void sync_check(const char* what)
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
        check_cuda(cudaGetLastError(),"BFS initialization failed");

        for(int level = 1;level <= depth;++level)
        {
            check_cuda(cudaMemset(changed.get(),0,sizeof(int)),"Failed to reset BFS flag");

            bfs_level_kernel<<<blocks,kBlockSize>>>(num_vertices,level,offsets,neighbors,distances,centers,parents,changed.get());
            check_cuda(cudaGetLastError(),"BFS level failed");

            int host_changed = 0;

            check_cuda(cudaMemcpy(&host_changed,changed.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read BFS flag");

            if(host_changed == 0)
            {
                break;
            }
        }
    }

    namespace
    {
        void add_inter_cluster_edges(int n,int adj,const int* offsets,const int* neighbors,const int* cluster,const int* source_mask,const int* ecc,int stretch,std::uint64_t* sink,int* sink_count,int capacity)
        {
            if(adj <= 0)
            {
                return;
            }

            DeviceBuffer<std::uint64_t> keys(adj),vals(adj);
            DeviceBuffer<int> candidate_count(1);

            check_cuda(cudaMemset(candidate_count.get(),0,sizeof(int)),"Failed to reset candidate counter");

            inter_candidates_kernel<<<grid_size(n),kBlockSize>>>(n,offsets,neighbors,cluster,source_mask,ecc,stretch,keys.get(),vals.get(),candidate_count.get());
            check_cuda(cudaGetLastError(),"Inter-cluster candidates failed");

            int candidates = 0;

            check_cuda(cudaMemcpy(&candidates,candidate_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read candidate count");

            if(candidates <= 0)
            {
                return;
            }

            thrust::sort_by_key(dp(keys.get()),dp(keys.get()) + candidates,dp(vals.get()));

            const auto unique_end = thrust::unique_by_key(dp(keys.get()),dp(keys.get()) + candidates,dp(vals.get()));
            const int unique_count = static_cast<int>(unique_end.first - dp(keys.get()));

            inter_emit_kernel<<<grid_size(unique_count),kBlockSize>>>(unique_count,vals.get(),sink,sink_count,capacity);
            check_cuda(cudaGetLastError(),"Inter-cluster emit failed");
        }

        std::vector<Edge> collect(DeviceBuffer<std::uint64_t>& sink,DeviceBuffer<int>& sink_count,int capacity)
        {
            int count = 0;

            check_cuda(cudaMemcpy(&count,sink_count.get(),sizeof(int),cudaMemcpyDeviceToHost),"Failed to read edge count");

            if(count > capacity)
            {
                throw std::runtime_error("Spanner exceeded the allocated edge capacity");
            }

            return finalize_packed_edges(sink.get(),count);
        }
    }

    // ------------------------------------------------------------------ CS (Sec. 3.3)
    void run_compact_pipeline(const int* offsets,const int* neighbors,int n,int adj,int radius,std::vector<Edge>& edges,int& num_centers)
    {
        edges.clear();
        num_centers = 0;

        if(n <= 0)
        {
            return;
        }

        const int blocks = grid_size(n);

        DeviceBuffer<int> is_center(n),dist(n),cen(n),par(n),ecc(n);

        run_hcis_r(n,radius,offsets,neighbors,is_center.get());

        num_centers = thrust::reduce(dp(is_center.get()),dp(is_center.get()) + n,0);

        grow_compact_bfs(n,radius,offsets,neighbors,is_center.get(),dist.get(),cen.get(),par.get());

        fix_unreached_kernel<<<blocks,kBlockSize>>>(n,dist.get(),cen.get(),par.get());

        check_cuda(cudaMemset(ecc.get(),0,static_cast<std::size_t>(n) * sizeof(int)),"Failed to clear cluster depths");
        ecc_from_bfs_kernel<<<blocks,kBlockSize>>>(n,dist.get(),cen.get(),ecc.get());

        const int capacity = n + adj + 16;
        DeviceBuffer<std::uint64_t> sink(capacity);
        DeviceBuffer<int> sink_count(1);

        check_cuda(cudaMemset(sink_count.get(),0,sizeof(int)),"Failed to reset edge counter");

        tree_append_kernel<<<blocks,kBlockSize>>>(n,dist.get(),par.get(),sink.get(),sink_count.get(),capacity);

        add_inter_cluster_edges(n,adj,offsets,neighbors,cen.get(),nullptr,ecc.get(),2 * radius + 1,sink.get(),sink_count.get(),capacity);

        sync_check("CS pipeline failed");

        edges = collect(sink,sink_count,capacity);
    }

    // ------------------------------------------------------------------ hybrid FGV-CS (Sec. 4)
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
        const int stretch = 2 * k_ - 1;
        const int* offsets = graph_.offsets();
        const int* neighbors = graph_.neighbors();
        const int blocks = grid_size(n);

        // ---- Stage 1+2: modified HCIS-r (alpha phases, beta boundary edges) and radius-r clusters (set D)
        DeviceBuffer<int> is_center(n),dist(n),cen(n),par(n),ecc(n);

        const double beta_double = std::pow(static_cast<double>(n),1.0 + 1.0 / static_cast<double>(k_));
        const long long beta = static_cast<long long>(std::min(beta_double,9.0e18));

        run_hcis_r(n,r,offsets,neighbors,is_center.get(),alpha_,beta);

        num_centers_ = thrust::reduce(dp(is_center.get()),dp(is_center.get()) + n,0);

        grow_compact_bfs(n,r,offsets,neighbors,is_center.get(),dist.get(),cen.get(),par.get());

        check_cuda(cudaMemset(ecc.get(),0,static_cast<std::size_t>(n) * sizeof(int)),"Failed to clear cluster depths");
        ecc_from_bfs_kernel<<<blocks,kBlockSize>>>(n,dist.get(),cen.get(),ecc.get());

        const int capacity = 2 * n + 2 * adj + 16;
        DeviceBuffer<std::uint64_t> sink(capacity);
        DeviceBuffer<int> sink_count(1);

        check_cuda(cudaMemset(sink_count.get(),0,sizeof(int)),"Failed to reset edge counter");

        tree_append_kernel<<<blocks,kBlockSize>>>(n,dist.get(),par.get(),sink.get(),sink_count.get(),capacity);

        // ---- Stage 3: FGV-Spanner on the vertices not covered by D (set B)
        DeviceBuffer<int> flag(n),newid(n),cluster(n),d_mask(n);

        flag_residual_kernel<<<blocks,kBlockSize>>>(n,dist.get(),flag.get());
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
            thrust::exclusive_scan(dp(deg.get()),dp(deg.get()) + n,dp(pos.get()));

            int last_deg = 0,last_pos = 0;
            check_cuda(cudaMemcpy(&last_deg,deg.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual degree");
            check_cuda(cudaMemcpy(&last_pos,pos.get() + n - 1,sizeof(int),cudaMemcpyDeviceToHost),"Failed to read residual position");

            const int res_adj = last_pos + last_deg;

            DeviceBuffer<int> res_off(n_res + 1),res_nbrs(std::max(1,res_adj)),old_of_new(n_res);

            residual_build_kernel<<<blocks,kBlockSize>>>(n,offsets,neighbors,flag.get(),newid.get(),pos.get(),res_off.get(),res_nbrs.get(),old_of_new.get());
            check_cuda(cudaGetLastError(),"Residual graph build failed");

            check_cuda(cudaMemcpy(res_off.get() + n_res,&res_adj,sizeof(int),cudaMemcpyHostToDevice),"Failed to finish residual offsets");

            // Same steps as FGV::run(), on the residual graph.
            const float probability = static_cast<float>(1.0 - std::pow(static_cast<double>(n_res),-1.0 / static_cast<double>(k_)));
            const int res_max_edges = n_res + res_adj;

            DeviceBuffer<int> shifts(n_res),r_dist(n_res),r_cen(n_res),r_par(n_res);
            DeviceBuffer<int> f_count(1),f_src(res_max_edges),f_dst(res_max_edges);

            initialize_fgv(n_res,r,probability,shifts.get(),r_dist.get(),r_cen.get(),r_par.get());
            build_fgv_clusters(n_res,r,res_off.get(),res_nbrs.get(),shifts.get(),r_dist.get(),r_cen.get(),r_par.get());
            build_fgv_spanner(n_res,res_off.get(),res_nbrs.get(),r_dist.get(),r_cen.get(),r_par.get(),f_count.get(),f_src.get(),f_dst.get(),res_max_edges);

            residual_edges_append_kernel<<<grid_size(res_max_edges),kBlockSize>>>(f_count.get(),f_src.get(),f_dst.get(),old_of_new.get(),sink.get(),sink_count.get(),capacity);
            ecc_from_fgv_kernel<<<grid_size(n_res),kBlockSize>>>(n_res,r_dist.get(),r_cen.get(),old_of_new.get(),ecc.get());
            assemble_cluster_kernel<<<blocks,kBlockSize>>>(n,dist.get(),cen.get(),newid.get(),r_cen.get(),old_of_new.get(),cluster.get(),d_mask.get());
            sync_check("Residual FGV stage failed");
        }

        // ---- Stage 4: inter-cluster edges involving D (Case 2), residual-residual ones came from FGV (Case 1)
        add_inter_cluster_edges(n,adj,offsets,neighbors,cluster.get(),n_res == 0 ? nullptr : d_mask.get(),ecc.get(),stretch,sink.get(),sink_count.get(),capacity);

        sync_check("FGV-CS inter-cluster stage failed");

        edges_ = collect(sink,sink_count,capacity);
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
