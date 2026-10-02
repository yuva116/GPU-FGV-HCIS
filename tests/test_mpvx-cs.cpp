#include "algorithms/mpvx-cs/mpvx-cs.hpp"

#include "graph/csr.hpp"
#include "graph/graph_loader.hpp"
#include "gpu/gpu_graph.hpp"
#include "validation/gpu_stretch.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace
{
	bool has_input_edge(const spanner::CSRGraph& graph,int source,int destination)
	{
		const auto begin = graph.neighbors_data().begin() + static_cast<std::ptrdiff_t>(graph.offsets()[source]);
		const auto end = graph.neighbors_data().begin() + static_cast<std::ptrdiff_t>(graph.offsets()[source + 1]);
		return std::binary_search(begin,end,destination);
	}

	std::vector<spanner::Edge> make_grid_edges(int width,int height)
	{
		std::vector<spanner::Edge> edges;
		for(int row = 0;row < height;++row)
		{
			for(int col = 0;col < width;++col)
			{
				const int vertex = row * width + col;
				if(col + 1 < width)
				{
					edges.push_back({vertex,vertex + 1});
				}
				if(row + 1 < height)
				{
					edges.push_back({vertex,vertex + width});
				}
			}
		}
		return edges;
	}

	std::vector<spanner::Edge> make_ring_ladder_edges(int rungs)
	{
		std::vector<spanner::Edge> edges;
		for(int rung = 0;rung < rungs;++rung)
		{
			const int next = (rung + 1) % rungs;
			edges.push_back({rung,next});
			edges.push_back({rung,rung + rungs});
			edges.push_back({rung + rungs,next + rungs});
		}
		return edges;
	}

	std::vector<spanner::Edge> make_random_like_graph(int n,int attachment_count,std::uint32_t seed)
	{
		std::vector<spanner::Edge> edges;
		std::unordered_set<std::uint64_t> seen;
		std::mt19937 random(seed);
		const auto key = [](int a,int b)
		{
			const int low = std::min(a,b);
			const int high = std::max(a,b);
			return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(low)) << 32) | static_cast<std::uint32_t>(high);
		};

		std::vector<int> prior;
		for(int vertex = 0;vertex < std::min(n,attachment_count);++vertex)
		{
			if(vertex > 0)
			{
				seen.insert(key(vertex - 1,vertex));
				edges.push_back({vertex - 1,vertex});
			}
			prior.push_back(vertex);
		}

		for(int vertex = static_cast<int>(prior.size());vertex < n;++vertex)
		{
			std::shuffle(prior.begin(),prior.end(),random);
			const int count = std::min(attachment_count,static_cast<int>(prior.size()));
			for(int index = 0;index < count;++index)
			{
				const int neighbor = prior[index];
				if(seen.insert(key(vertex,neighbor)).second)
				{
					edges.push_back({neighbor,vertex});
				}
			}
			prior.push_back(vertex);
		}
		return edges;
	}

	std::vector<int> connected_components(int num_vertices,const std::vector<spanner::Edge>& edges)
	{
		std::vector<std::vector<int>> adjacency(static_cast<std::size_t>(num_vertices));
		for(const spanner::Edge& edge : edges)
		{
			adjacency[edge.source].push_back(edge.destination);
			adjacency[edge.destination].push_back(edge.source);
		}

		std::vector<int> component_ids(static_cast<std::size_t>(num_vertices),-1);
		int component_count = 0;
		std::vector<int> frontier;

		for(int start = 0;start < num_vertices;++start)
		{
			if(component_ids[start] != -1)
			{
				continue;
			}

			component_ids[start] = component_count;
			frontier.clear();
			frontier.push_back(start);

			for(std::size_t index = 0;index < frontier.size();++index)
			{
				for(int neighbor : adjacency[frontier[index]])
				{
					if(component_ids[neighbor] == -1)
					{
						component_ids[neighbor] = component_count;
						frontier.push_back(neighbor);
					}
				}
			}
			++component_count;
		}
		return component_ids;
	}

	int validate_connectivity(const spanner::CSRGraph& graph,const std::vector<spanner::Edge>& spanner_edges)
	{
		const int num_vertices = static_cast<int>(graph.num_vertices());
		std::vector<spanner::Edge> input_edges;
		input_edges.reserve(graph.num_edges());
		for(int vertex = 0;vertex < num_vertices;++vertex)
		{
			for(std::size_t index = graph.offsets()[vertex];index < graph.offsets()[vertex + 1];++index)
			{
				const int neighbor = graph.neighbors_data()[index];
				if(vertex < neighbor)
				{
					input_edges.push_back({vertex,neighbor});
				}
			}
		}

		const std::vector<int> input_components = connected_components(num_vertices,input_edges);
		const std::vector<int> spanner_components = connected_components(num_vertices,spanner_edges);
		std::vector<int> input_to_spanner(static_cast<std::size_t>(num_vertices),-1);
		std::vector<int> spanner_to_input(static_cast<std::size_t>(num_vertices),-1);

		for(int vertex = 0;vertex < num_vertices;++vertex)
		{
			const int input_component = input_components[vertex];
			const int spanner_component = spanner_components[vertex];
			if(input_to_spanner[input_component] == -1)
			{
				input_to_spanner[input_component] = spanner_component;
			}
			else if(input_to_spanner[input_component] != spanner_component)
			{
				throw std::runtime_error("MPVX-CS disconnects an input component");
			}

			if(spanner_to_input[spanner_component] == -1)
			{
				spanner_to_input[spanner_component] = input_component;
			}
			else if(spanner_to_input[spanner_component] != input_component)
			{
				throw std::runtime_error("MPVX-CS connects distinct input components");
			}
		}

		return *std::max_element(input_components.begin(),input_components.end()) + 1;
	}

	void run_case(const std::string& name,const spanner::CSRGraph& graph,int k)
	{
		spanner::GPUGraph gpu_graph(graph);
		spanner::MPVXCS algorithm(gpu_graph,k);
		algorithm.run();

		const auto& edges = algorithm.edges();
		if(graph.num_edges() > 0 && edges.empty())
		{
			throw std::runtime_error(name + ": MPVX-CS returned an empty spanner");
		}

		for(const spanner::Edge& edge : edges)
		{
			if(edge.source < 0 || edge.destination < 0 || edge.source >= static_cast<int>(graph.num_vertices()) || edge.destination >= static_cast<int>(graph.num_vertices()))
			{
				throw std::runtime_error(name + ": MPVX-CS emitted an invalid vertex");
			}
			if(edge.source >= edge.destination || !has_input_edge(graph,edge.source,edge.destination))
			{
				throw std::runtime_error(name + ": MPVX-CS emitted a noncanonical or non-input edge");
			}
		}
		for(std::size_t index = 1;index < edges.size();++index)
		{
			if(edges[index - 1].source > edges[index].source ||
			   (edges[index - 1].source == edges[index].source && edges[index - 1].destination >= edges[index].destination))
			{
				throw std::runtime_error(name + ": MPVX-CS edges are not sorted and unique");
			}
		}
		const int component_count = validate_connectivity(graph,edges);

		int stretch_bound = 4*k + 1;
		const auto stretch = spanner::validate_gpu_stretch(graph,edges,stretch_bound,4096);
		if(!stretch.passed)
		{
			throw std::runtime_error(name + ": MPVX-CS exceeded the 4k+1 stretch bound");
		}

		std::cout << name << ": n=" << graph.num_vertices()
                    << ", k=" << k
				  << ", input_edges=" << graph.num_edges()
				  << ", high_coverage_centers=" << algorithm.num_centers()
				  << ", residual_vertices=" << algorithm.num_residual()
				  << ", connected_components=" << component_count
				  << ", spanner_edges=" << edges.size()
				  << ", maximum_stretch=" << stretch.maximum_stretch
				  << (stretch.full_validation ? " (full)\n" : " (sampled)\n");
	}
}

int main()
{
	try
	{
		int k = 1;
		const spanner::CSRGraph single_vertex(1,{},false);
		const spanner::CSRGraph grid(16,make_grid_edges(4,4),false);
		const spanner::CSRGraph ladder(12,make_ring_ladder_edges(6),false);
		const spanner::GraphData dataset_data = spanner::GraphLoader::load_edge_list("datasets/test_graph.txt");
		const spanner::CSRGraph dataset(dataset_data.num_vertices,dataset_data.edges,false);
		const spanner::CSRGraph random_graph(2048,make_random_like_graph(2048,6,20240621U),false);

		run_case("single_vertex",single_vertex,k);
		run_case("grid_4x4",grid,k);
		run_case("ladder_6",ladder,k);
		run_case("dataset_test_graph",dataset,k);
		run_case("random_like_2048",random_graph,k);

		std::cout << "MPVX-CS MULTI-GRAPH TEST: PASS\n";
		return 0;
	}
	catch(const std::exception& error)
	{
		std::cerr << "MPVX-CS MULTI-GRAPH TEST: FAIL\n" << error.what() << '\n';
		return 1;
	}
}
