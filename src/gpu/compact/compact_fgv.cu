#include "gpu/compact_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <climits>

namespace
{
    constexpr int BLOCK_SIZE = 256;

    __global__ void initialize_fgv_residual_kernel(int num_vertices,const int* active,int* distances,int* clusters,int* parents)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(active[vertex])
        {
            distances[vertex] = 0;
            clusters[vertex] = vertex;
            parents[vertex] = vertex;
        }
        else
        {
            distances[vertex] = INT_MAX;
            clusters[vertex] = -1;
            parents[vertex] = -1;
        }
    }

    __global__ void fgv_relax_kernel(int num_vertices,int round,const int* offsets,const int* neighbors,const int* active,int* distances,int* clusters,int* parents)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices || !active[vertex])
        {
            return;
        }

        int best_distance = distances[vertex];
        int best_cluster = clusters[vertex];
        int best_parent = parents[vertex];

        for(int edge = offsets[vertex];edge < offsets[vertex + 1];++edge)
        {
            const int neighbor = neighbors[edge];

            if(!active[neighbor])
            {
                continue;
            }

            const int neighbor_distance = distances[neighbor];

            if(neighbor_distance == INT_MAX)
            {
                continue;
            }

            const int candidate_distance = neighbor_distance + 1;

            if(candidate_distance > round)
            {
                continue;
            }

            const int candidate_cluster = clusters[neighbor];

            if(candidate_distance < best_distance)
            {
                best_distance = candidate_distance;
                best_cluster = candidate_cluster;
                best_parent = neighbor;
            }
            else if(candidate_distance == best_distance && candidate_cluster >= 0 && (best_cluster < 0 || candidate_cluster < best_cluster))
            {
                best_cluster = candidate_cluster;
                best_parent = neighbor;
            }
        }

        distances[vertex] = best_distance;
        clusters[vertex] = best_cluster;
        parents[vertex] = best_parent;
    }

    __global__ void initialize_fgv_shifts_kernel(int num_vertices,const int* active,int radius,float probability,unsigned long long seed,int* shifts)
    {
        const int vertex = blockIdx.x * blockDim.x + threadIdx.x;

        if(vertex >= num_vertices)
        {
            return;
        }

        if(!active[vertex])
        {
            shifts[vertex] = -1;
            return;
        }

        unsigned long long value = seed ^ (static_cast<unsigned long long>(vertex) * 0x9e3779b97f4a7c15ULL);

        value ^= value >> 30;
        value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27;
        value *= 0x94d049bb133111ebULL;
        value ^= value >> 31;

        const double uniform = static_cast<double>(value & 0xffffffffULL) / 4294967296.0;

        int shift = 0;

        double cumulative = probability;

        while(shift < radius - 1 && uniform > cumulative)
        {
            ++shift;
            cumulative += probability * std::pow(1.0 - probability,static_cast<double>(shift));
        }

        shifts[vertex] = shift;
    }
}

namespace spanner
{
    void run_fgv_residual(int num_vertices,int k,const int* offsets,const int* neighbors,const int* active,int* distances,int* clusters,int* parents)
    {
        const int radius = k - 1;
        const int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

        initialize_fgv_residual_kernel<<<blocks,BLOCK_SIZE>>>(num_vertices,active,distances,clusters,parents);
        cudaDeviceSynchronize();

        if(radius <= 0)
        {
            return;
        }

        int* shifts = nullptr;

        cudaMalloc(&shifts,static_cast<std::size_t>(num_vertices) * sizeof(int));

        const double n = static_cast<double>(num_vertices);
        const double probability_double = 1.0 - std::pow(n,-1.0 / static_cast<double>(k));
        const float probability = static_cast<float>(probability_double);

        initialize_fgv_shifts_kernel<<<blocks,BLOCK_SIZE>>>(num_vertices,active,radius,probability,1234567ULL,shifts);

        cudaDeviceSynchronize();

        for(int round = 0;round < radius;++round)
        {
            fgv_relax_kernel<<<blocks,BLOCK_SIZE>>>(num_vertices,round,offsets,neighbors,active,distances,clusters,parents);
            cudaDeviceSynchronize();
        }

        cudaFree(shifts);
    }
}