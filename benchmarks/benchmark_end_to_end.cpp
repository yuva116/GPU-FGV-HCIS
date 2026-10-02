// Usage: run_benchmark <out.csv> <snap_graph.txt> [more graphs...]
// Runs FGV (baseline), Compact (CS) and the hybrid FGV-CS at stretch 3, 5, 7 on every graph,
// plus MPVX5-B (Miller) and MPVX5-CS at stretch 5, and writes one CSV row per (graph, algorithm, stretch).
#include "algorithms/compact/compact.hpp"
#include "algorithms/compact/fgv_compact.hpp"
#include "algorithms/fgv/fgv.hpp"
#include "algorithms/miller/miller.hpp"
#include "algorithms/mpvx-cs/mpvx-cs.hpp"
#include "graph/csr.hpp"
#include "graph/graph_loader.hpp"
#include "gpu/gpu_graph.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <out.csv> <graph.txt>...\n";
        return 2;
    }

    std::ofstream out(argv[1]);
    out << "graph,vertices,edges,algorithm,stretch,spanner_edges,time_ms\n";

    for (int i = 2; i < argc; ++i)
    {
        const std::string name = std::filesystem::path(argv[i]).stem().string();
        try
        {
            spanner::GraphData data = spanner::GraphLoader::load_snap(argv[i]);

            // Treat as undirected: canonicalize, drop self-loops and duplicates.
            for (auto& e : data.edges) if (e.source > e.destination) std::swap(e.source, e.destination);
            data.edges.erase(std::remove_if(data.edges.begin(), data.edges.end(),
                [](const spanner::Edge& e) { return e.source == e.destination; }), data.edges.end());
            std::sort(data.edges.begin(), data.edges.end(), [](const spanner::Edge& a, const spanner::Edge& b) {
                return a.source != b.source ? a.source < b.source : a.destination < b.destination; });
            data.edges.erase(std::unique(data.edges.begin(), data.edges.end(),
                [](const spanner::Edge& a, const spanner::Edge& b) {
                    return a.source == b.source && a.destination == b.destination; }), data.edges.end());

            const spanner::CSRGraph csr(data.num_vertices, data.edges, false);
            spanner::GPUGraph gpu(csr);

            for (int stretch : {3, 5, 7})
            {
                auto record = [&](const char* algo, std::size_t size, double ms)
                {
                    out << name << ',' << csr.num_vertices() << ',' << csr.num_edges() << ','
                        << algo << ',' << stretch << ',' << size << ',' << ms << '\n';
                    out.flush();
                    std::cout << name << ' ' << algo << " s=" << stretch << " edges=" << size
                              << " (" << ms << " ms)\n";
                };

                spanner::Timer t;
                spanner::FGV fgv(gpu, (stretch + 1) / 2);   // stretch = 2k-1
                fgv.run();
                record("FGV", fgv.edges().size(), t.elapsed_milliseconds());

                t.reset();
                spanner::Compact cs(gpu, (stretch - 1) / 2); // stretch = 2r+1
                cs.run();
                record("CS", cs.edges().size(), t.elapsed_milliseconds());

                t.reset();
                spanner::FGVCompact hybrid(gpu, (stretch + 1) / 2); // stretch = 2k-1
                hybrid.run();
                record("FGV-CS", hybrid.edges().size(), t.elapsed_milliseconds());

                if (stretch == 5)
                {
                    // MPVX baseline and its compact variant; guarded so a failure here
                    // does not discard the FGV/CS rows already recorded for this graph.
                    try
                    {
                        t.reset();
                        spanner::Miller mpvx(gpu, (stretch - 1) / 4);      // MPVX stretch = 4k+1 -> k=1 for stretch 5
                        mpvx.run();
                        record("MPVX5-B", mpvx.edges().size(), t.elapsed_milliseconds());

                        t.reset();
                        spanner::MPVXCS mpvx_cs(gpu, (stretch - 1) / 4);   // MPVX-CS stretch = 4k+1 -> k=1 for stretch 5
                        mpvx_cs.run();
                        record("MPVX5-CS", mpvx_cs.edges().size(), t.elapsed_milliseconds());
                    }
                    catch (const std::exception& e)
                    {
                        std::cerr << "skipping MPVX on " << name << ": " << e.what() << '\n';
                    }
                }
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "skipping " << name << ": " << e.what() << '\n';
        }
    }
    return 0;
}
