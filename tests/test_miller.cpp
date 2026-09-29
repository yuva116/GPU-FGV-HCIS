#include "algorithms/miller/miller.hpp"

#include "gpu/gpu_graph.hpp"
#include "graph/graph_loader.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

bool has_input_edge(const spanner::CSRGraph& graph, int source, int destination) {
    const auto begin = graph.neighbors_data().begin() + static_cast<std::ptrdiff_t>(graph.offsets()[source]);
    const auto end = graph.neighbors_data().begin() + static_cast<std::ptrdiff_t>(graph.offsets()[source + 1]);
    return std::binary_search(begin, end, destination);
}

std::vector<spanner::Edge> make_grid_edges(int width, int height) {
    std::vector<spanner::Edge> edges;
    edges.reserve(static_cast<std::size_t>((width * height) * 2));

    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            const int source = row * width + col;
            if (col + 1 < width) {
                edges.push_back({source, source + 1});
            }
            if (row + 1 < height) {
                edges.push_back({source, source + width});
            }
        }
    }

    return edges;
}

std::vector<spanner::Edge> make_ring_ladder_edges(int n) {
    std::vector<spanner::Edge> edges;
    edges.reserve(static_cast<std::size_t>(3 * n));

    for (int i = 0; i < n; ++i) {
        const int left = i;
        const int right = (i + 1) % n;
        const int left_upper = i + n;
        const int right_upper = ((i + 1) % n) + n;

        edges.push_back({left, right});
        edges.push_back({left, left_upper});
        edges.push_back({left_upper, right_upper});
    }

    return edges;
}

std::vector<spanner::Edge> make_random_like_graph(int num_vertices, int attachment_count, std::uint32_t seed) {
    std::vector<spanner::Edge> edges;
    std::unordered_set<std::uint64_t> seen;
    std::mt19937 rng(seed);

    const auto key = [](int u, int v) {
        const std::uint64_t a = static_cast<std::uint64_t>(u);
        const std::uint64_t b = static_cast<std::uint64_t>(v);
        return (a < b) ? ((a << 32) | b) : ((b << 32) | a);
    };

    std::vector<int> active_vertices;
    for (int v = 0; v < std::min(num_vertices, attachment_count); ++v) {
        active_vertices.push_back(v);
        if (v > 0) {
            const std::uint64_t edge_key = key(v - 1, v);
            seen.insert(edge_key);
            edges.push_back({v - 1, v});
        }
    }

    for (int v = std::min(num_vertices, attachment_count); v < num_vertices; ++v) {
        std::vector<int> candidates;
        candidates.reserve(static_cast<std::size_t>(active_vertices.size()));
        for (int u : active_vertices) {
            candidates.push_back(u);
        }

        std::shuffle(candidates.begin(), candidates.end(), rng);
        const int chosen_count = std::min(attachment_count, static_cast<int>(candidates.size()));

        for (int i = 0; i < chosen_count; ++i) {
            const int u = candidates[i];
            const std::uint64_t edge_key = key(u, v);
            if (seen.insert(edge_key).second) {
                edges.push_back({u, v});
            }
        }

        active_vertices.push_back(v);
    }

    return edges;
}

void validate_edges(const spanner::CSRGraph& graph, const std::vector<spanner::Edge>& edges) {
    if (edges.empty() && graph.num_edges() > 0) {
        throw std::runtime_error("Miller produced an empty spanner");
    }

    for (const spanner::Edge& edge : edges) {
        if (edge.source == edge.destination) {
            throw std::runtime_error("Miller produced a self-loop");
        }
        if (edge.source < 0 || edge.destination < 0 || edge.source >= static_cast<int>(graph.num_vertices()) || edge.destination >= static_cast<int>(graph.num_vertices())) {
            throw std::runtime_error("Miller produced an invalid vertex");
        }
        if (edge.source > edge.destination) {
            throw std::runtime_error("Miller edge is not canonicalized");
        }
        if (!has_input_edge(graph, edge.source, edge.destination)) {
            throw std::runtime_error("Miller produced an edge not present in the input graph");
        }
    }

    for (std::size_t index = 1; index < edges.size(); ++index) {
        const spanner::Edge& previous = edges[index - 1];
        const spanner::Edge& current = edges[index];
        if (previous.source > current.source || (previous.source == current.source && previous.destination >= current.destination)) {
            throw std::runtime_error("Miller edges are not sorted and unique");
        }
    }
}

std::vector<int> connected_components(int num_vertices, const std::vector<spanner::Edge>& edges) {
    std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(num_vertices));
    for (const spanner::Edge& edge : edges) {
        adjacency[edge.source].push_back(edge.destination);
        adjacency[edge.destination].push_back(edge.source);
    }

    std::vector<int> component_ids(static_cast<std::size_t>(num_vertices), -1);
    int component_count = 0;
    std::vector<int> frontier;

    for (int start = 0; start < num_vertices; ++start) {
        if (component_ids[start] != -1) {
            continue;
        }

        component_ids[start] = component_count;
        frontier.clear();
        frontier.push_back(start);

        for (std::size_t index = 0; index < frontier.size(); ++index) {
            const int vertex = frontier[index];
            for (int neighbor : adjacency[vertex]) {
                if (component_ids[neighbor] == -1) {
                    component_ids[neighbor] = component_count;
                    frontier.push_back(neighbor);
                }
            }
        }

        ++component_count;
    }

    return component_ids;
}

int validate_connectivity(const spanner::CSRGraph& graph, const std::vector<spanner::Edge>& spanner_edges) {
    const int num_vertices = static_cast<int>(graph.num_vertices());
    const std::vector<int> input_components = connected_components(num_vertices, graph.neighbors_data().empty()
        ? std::vector<spanner::Edge>{}
        : [&graph]() {
            std::vector<spanner::Edge> edges;
            edges.reserve(graph.num_edges());
            for (int vertex = 0; vertex < static_cast<int>(graph.num_vertices()); ++vertex) {
                for (std::size_t index = graph.offsets()[vertex]; index < graph.offsets()[vertex + 1]; ++index) {
                    const int neighbor = graph.neighbors_data()[index];
                    if (vertex < neighbor) {
                        edges.push_back({vertex, neighbor});
                    }
                }
            }
            return edges;
        }());
    const std::vector<int> spanner_components = connected_components(num_vertices, spanner_edges);

    std::vector<int> input_to_spanner(static_cast<std::size_t>(num_vertices), -1);
    std::vector<int> spanner_to_input(static_cast<std::size_t>(num_vertices), -1);
    for (int vertex = 0; vertex < num_vertices; ++vertex) {
        const int input_component = input_components[vertex];
        const int spanner_component = spanner_components[vertex];

        if (input_to_spanner[input_component] == -1) {
            input_to_spanner[input_component] = spanner_component;
        } else if (input_to_spanner[input_component] != spanner_component) {
            throw std::runtime_error("Miller spanner disconnects an input component");
        }

        if (spanner_to_input[spanner_component] == -1) {
            spanner_to_input[spanner_component] = input_component;
        } else if (spanner_to_input[spanner_component] != input_component) {
            throw std::runtime_error("Miller spanner connects distinct input components");
        }
    }

    return *std::max_element(input_components.begin(), input_components.end()) + 1;
}

void run_case(const std::string& name, const spanner::CSRGraph& graph, int k) {
    const auto algorithm_start = std::chrono::steady_clock::now();
    spanner::GPUGraph gpu_graph(graph);
    spanner::Miller miller(gpu_graph, k);
    miller.run();
    const auto algorithm_end = std::chrono::steady_clock::now();

    const std::vector<spanner::Edge>& spanner_edges = miller.edges();
    validate_edges(graph, spanner_edges);

    const auto validation_start = std::chrono::steady_clock::now();
    const int component_count = validate_connectivity(graph, spanner_edges);
    const auto validation_end = std::chrono::steady_clock::now();

    const auto algorithm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(algorithm_end - algorithm_start);
    const auto validation_ms = std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start);

    std::cout << name <<": n=" << graph.num_vertices()
                <<", k=" << k
                << ", original edges=" << graph.num_edges()
              << ", spanner edges=" << spanner_edges.size()
                            << ", connected components=" << component_count
              << ", algorithm_ms=" << algorithm_ms.count()
                            << ", connectivity_ms=" << validation_ms.count() << '\n';
}

} // namespace

int main() {
    const spanner::CSRGraph grid_graph(16, make_grid_edges(4, 4), false);
    const spanner::CSRGraph ladder_graph(12, make_ring_ladder_edges(6), false);
    const spanner::CSRGraph single_vertex_graph(1, std::vector<spanner::Edge>{}, false);
    const auto dataset_data = spanner::GraphLoader::load_snap("datasets/test_graph.txt");
    const spanner::CSRGraph dataset_graph(dataset_data.num_vertices, dataset_data.edges, false);

    const std::vector<spanner::Edge> random_edges = make_random_like_graph(2048, 6, 20240621U);
    const spanner::CSRGraph random_graph(2048, random_edges, false);

    run_case("single_vertex", single_vertex_graph, 3);
    run_case("grid_4x4", grid_graph, 3);
    run_case("ladder_6", ladder_graph, 3);
    run_case("dataset_test_graph", dataset_graph, 3);
    run_case("random_like_2048", random_graph, 3);

    std::cout << "MILLER MULTI-GRAPH TEST: PASS\n";
    return 0;
}
