#pragma once

#include <vector>

#include "graph/edge.hpp"

namespace spanner
{
    // max_iterations  : stop after this many HCIS phases (alpha); 0 = run until no active vertex remains.
    // max_boundary_edges : reject a phase's centers if their total boundary edges exceed this (beta); <0 = no limit.
    void run_hcis_r(int num_vertices,int radius,const int* offsets,const int* neighbors,int* is_center,int max_iterations = 0,long long max_boundary_edges = -1);

    // Multi-source BFS from is_center up to `depth` hops. Unreached vertices keep distances == -1.
    // (Unlike grow_compact_clusters, this never repairs / extends past `depth`.)
    void grow_compact_bfs(int num_vertices,int depth,const int* offsets,const int* neighbors,const int* is_center,int* distances,int* centers,int* parents);

    // Full CS pipeline (Sec. 3.3): HCIS-r -> radius-r BFS clusters -> tree + inter-cluster edges -> sorted unique edges.
    void run_compact_pipeline(const int* offsets,const int* neighbors,int num_vertices,int num_adjacency_entries,int radius,std::vector<Edge>& edges,int& num_centers);

    void initialize_compact_active_set(int num_vertices,int* active,int* active_neighbors);

    void update_compact_active_neighbors(int num_vertices,const int* offsets,const int* neighbors,const int* active,int* active_neighbors);

    void deactivate_compact_radius(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* is_center,int* active);

    void clear_compact_center_flags(int num_vertices,int* is_center);

    void grow_compact_clusters(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* is_center,int* distances,int* centers,int* parents);

    void build_compact_spanner(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,const int* parents,int* edge_count,int* spanner_sources,int* spanner_destinations,int max_edges);
}