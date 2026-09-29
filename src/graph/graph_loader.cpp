#include "graph/graph_loader.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace spanner {

GraphData GraphLoader::load_edge_list(
    const std::string& filename) {

    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open graph file: "+ filename);
    }

    GraphData graph;

    std::string line;
    int max_vertex = -1;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream input(line);

        int source;
        int destination;

        if (!(input >> source >> destination)) {
            throw std::runtime_error("Invalid edge: " +line);
        }

        if (source < 0 || destination < 0) {
            throw std::runtime_error("Vertex IDs must not be negative");
        }

        graph.edges.push_back({source, destination});

        if (source > max_vertex) {
            max_vertex = source;
        }

        if (destination > max_vertex) {
            max_vertex = destination;
        }
    }

    if (max_vertex >= 0) {
        graph.num_vertices =
            static_cast<std::size_t>(max_vertex + 1);
    }

    return graph;
}

GraphData GraphLoader::load_snap(
    const std::string& filename) {

    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error("Could not open SNAP graph file: " + filename);
    }

    std::vector<Edge> input_edges;

    std::string line;

    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream input(line);

        int source;
        int destination;

        if (!(input >> source >> destination)) {
            throw std::runtime_error("Invalid SNAP edge: " +line);
        }

        if (source < 0 || destination < 0) {
            throw std::runtime_error("SNAP vertex IDs must not be negative");
        }

        input_edges.push_back({source, destination});
    }

    // Convert SNAP vertex IDs to consecutive IDs:
    // 0, 1, 2, ..., V-1.
    std::unordered_map<int, int> vertex_map;

    int next_vertex = 0;

    for (const Edge& edge : input_edges) {
        if (vertex_map.find(edge.source) ==
            vertex_map.end()) {
            vertex_map[edge.source] = next_vertex++;
        }

        if (vertex_map.find(edge.destination) ==
            vertex_map.end()) {
            vertex_map[edge.destination] = next_vertex++;
        }
    }

    GraphData graph;

    graph.num_vertices = static_cast<std::size_t>(next_vertex);

    graph.edges.reserve(input_edges.size());

    for (const Edge& edge : input_edges) {
        graph.edges.push_back({vertex_map.at(edge.source),vertex_map.at(edge.destination)});
    }

    return graph;
}

} 