#include "algorithms/compact/fgv_compact.hpp"
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

        constexpr int validation_sources = 4096;

        // Hybrid FGV-CS gives a (2k-1)-stretch spanner (Lemma 4.2).
        for(int k : {2,3,4})
        {
            const int stretch_bound = 2 * k - 1;

            spanner::FGVCompact hybrid(gpu_graph,k);
            hybrid.run();

            const std::vector<spanner::Edge>& spanner_edges = hybrid.edges();

            std::cout << "k=" << k << " vertices=" << csr_graph.num_vertices() << " input edges=" << csr_graph.num_edges()
                      << " HCIS centers=" << hybrid.num_centers() << " residual vertices=" << hybrid.num_residual()
                      << " spanner edges=" << spanner_edges.size() << '\n';

            if(spanner_edges.empty())
            {
                throw std::runtime_error("FGV-CS produced an empty spanner");
            }

            for(std::size_t i = 0;i < spanner_edges.size();++i)
            {
                const spanner::Edge& edge = spanner_edges[i];

                if(edge.source >= edge.destination || edge.source < 0 || edge.destination >= static_cast<int>(csr_graph.num_vertices()))
                {
                    throw std::runtime_error("FGV-CS produced an invalid or non-canonical edge");
                }

                if(!has_input_edge(csr_graph,edge.source,edge.destination))
                {
                    throw std::runtime_error("FGV-CS produced an edge not present in the input graph");
                }

                if(i > 0 && !(spanner_edges[i - 1].source < edge.source || (spanner_edges[i - 1].source == edge.source && spanner_edges[i - 1].destination < edge.destination)))
                {
                    throw std::runtime_error("FGV-CS edges are not sorted and unique");
                }
            }

            const spanner::GPUStretchResult result = spanner::validate_gpu_stretch(csr_graph,spanner_edges,stretch_bound,validation_sources);

            std::cout << "  max observed stretch: " << result.maximum_stretch << " (bound " << stretch_bound << ")\n";

            if(!result.passed)
            {
                std::cerr << "Stretch violation: (" << result.violating_source << "," << result.violating_destination << ")\n";
                throw std::runtime_error("FGV-CS stretch bound violated");
            }
        }

        std::cout << "FGV-CS TEST: PASS\n";

        return 0;
    }
    catch(const std::exception& error)
    {
        std::cerr << "FGV-CS TEST: FAIL\n";
        std::cerr << error.what() << '\n';
        return 1;
    }
}
