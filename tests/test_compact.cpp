#include "algorithms/compact/compact.hpp"
#include "graph/csr.hpp"
#include "graph/graph_loader.hpp"
#include "gpu/gpu_graph.hpp"
#include "validation/gpu_stretch.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

bool has_input_edge(const spanner::CSRGraph& graph,int source,int destination)
{
    const int begin = static_cast<int>(graph.offsets()[source]);
    const int end = static_cast<int>(graph.offsets()[source + 1]);

    return std::binary_search(graph.neighbors_data().begin() + begin,graph.neighbors_data().begin() + end,destination);
}

}

int main()
{
    try
    {
        const spanner::GraphData graph_data = spanner::GraphLoader::load_edge_list("datasets/test_graph.txt");
        const spanner::CSRGraph csr_graph(graph_data.num_vertices,graph_data.edges,false);

        spanner::GPUGraph gpu_graph(csr_graph);

        // Section 3.3 of the paper: CS with radius r yields a
        // (2r + 1)-stretch spanner (via Lemma 2.1, applied to an LDD of
        // diameter 2r grown from a maximal r-hop independent set).
        constexpr int radius = 2;
        constexpr int stretch_bound = 2 * radius + 1;
        constexpr int validation_sources = 4096;

        spanner::Compact compact(gpu_graph,radius);
        compact.run();

        const std::vector<spanner::Edge>& spanner_edges = compact.edges();

        std::cout << "Vertices: " << csr_graph.num_vertices() << '\n';
        std::cout << "Input edges: " << csr_graph.num_edges() << '\n';
        std::cout << "HCIS-" << radius << " centers selected: " << compact.num_centers() << '\n';
        std::cout << "Spanner edges: " << spanner_edges.size() << '\n';

        if(spanner_edges.empty())
        {
            throw std::runtime_error("Compact produced an empty spanner");
        }

        if(compact.num_centers() <= 0)
        {
            throw std::runtime_error("HCIS-r selected no centers");
        }

        for(const spanner::Edge& edge : spanner_edges)
        {
            if(edge.source == edge.destination)
            {
                throw std::runtime_error("Compact produced a self-loop");
            }

            if(edge.source < 0 || edge.destination < 0 || edge.source >= static_cast<int>(csr_graph.num_vertices()) || edge.destination >= static_cast<int>(csr_graph.num_vertices()))
            {
                throw std::runtime_error("Compact produced an invalid vertex");
            }

            if(edge.source > edge.destination)
            {
                throw std::runtime_error("Compact edge is not canonicalized");
            }

            if(!has_input_edge(csr_graph,edge.source,edge.destination))
            {
                throw std::runtime_error("Compact produced an edge not present in the input graph");
            }
        }

        for(std::size_t index = 1;index < spanner_edges.size();++index)
        {
            const spanner::Edge& previous = spanner_edges[index - 1];
            const spanner::Edge& current = spanner_edges[index];

            if(previous.source > current.source || (previous.source == current.source && previous.destination >= current.destination))
            {
                throw std::runtime_error("Compact edges are not sorted and unique");
            }
        }

        std::cout << "Structural checks: PASS\n";

        const spanner::GPUStretchResult stretch_result = spanner::validate_gpu_stretch(csr_graph,spanner_edges,stretch_bound,validation_sources);

        std::cout << "Checked source vertices: " << stretch_result.checked_sources << " / " << stretch_result.total_sources << '\n';
        std::cout << "Checked edge incidences: " << stretch_result.checked_edge_incidences << '\n';
        std::cout << "Maximum observed stretch: " << stretch_result.maximum_stretch << '\n';
        std::cout << "Required stretch bound: " << stretch_bound << '\n';

        if(!stretch_result.passed)
        {
            std::cerr << "Stretch violation: (" << stretch_result.violating_source << "," << stretch_result.violating_destination << ")\n";
            throw std::runtime_error("Compact stretch bound violated");
        }

        if(stretch_result.full_validation)
        {
            std::cout << "GPU stretch check: PASS\n";
        }
        else
        {
            std::cout << "GPU stretch check: PASS (sampled)\n";
        }

        std::cout << "COMPACT TEST: PASS\n";

        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "COMPACT TEST: FAIL\n";
        std::cerr << error.what() << '\n';
        return 1;
    }
}