#pragma once

namespace spanner
{
    void initialize_fgv(int vertices, int radius,float probability, int* shifts, int* distances,int* centers, int* parents);
    void build_fgv_clusters(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* shifts,int* distances,int* centers,int* parents);
    void build_fgv_spanner(int num_vertices,const int* offsets,const int* neighbors,const int* distances,const int* centers,const int* parents,int* edge_count,int* spanner_sources, int* spanner_destinations, int max_edges);
}