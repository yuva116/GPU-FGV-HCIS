#include <cassert>
#include <iostream>
#include "graph/csr.hpp"
#include "graph/graph_loader.hpp"
#include "validation/correctness.hpp"
#include "utils/timer.hpp"

int main() {
    using namespace spanner;

    const GraphData data = GraphLoader::load_snap("datasets/test_graph.txt");

    std::cout << "Loaded vertices: "<< data.num_vertices<<'\n';
    std::cout << "Loaded edges: "<< data.edges.size()<<'\n';

    assert(data.num_vertices > 0);
    assert(!data.edges.empty());

    Timer timer;

    CSRGraph graph(data.num_vertices,data.edges,false);

    std::cout << "CSR construction time: "<< timer.elapsed_milliseconds()<< " ms\n";

    assert(graph.num_vertices() == data.num_vertices);
    assert(graph.num_edges() == data.edges.size());
    assert(graph.num_adjacency_entries()== 2 * graph.num_edges());
    assert(validation::validate_csr(graph));

    std::cout << "CSR graph is valid.\n";
    std::cout << "Vertices: "<< graph.num_vertices()<< '\n';
    std::cout << "Edges: "<< graph.num_edges()<< '\n';
    std::cout << "Adjacency entries: "<< graph.num_adjacency_entries()<< '\n';
    std::cout << "CSR test passed.\n";

    return 0;
}