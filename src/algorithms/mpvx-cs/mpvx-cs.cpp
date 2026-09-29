#include "algorithms/mpvx-cs/mpvx-cs.hpp"

#include "gpu/mpvx_cs.hpp"

#include <stdexcept>

namespace spanner
{
	MPVXCS::MPVXCS(const GPUGraph& graph,int k)
		: graph_(graph),k_(k)
	{
		if(k_ < 1 || k_ > 6)
		{
			throw std::invalid_argument("MPVXCS requires 1 <= k <= 6 (HCIS radius is limited to 12)");
		}
	}

	void MPVXCS::run()
	{
		run_mpvx_cs(graph_.offsets(),graph_.neighbors(),static_cast<int>(graph_.num_vertices()),static_cast<int>(graph_.num_adjacency_entries()),k_,edges_,num_centers_,num_residual_);
	}

	const std::vector<Edge>& MPVXCS::edges() const noexcept
	{
		return edges_;
	}

	int MPVXCS::num_centers() const noexcept
	{
		return num_centers_;
	}

	int MPVXCS::num_residual() const noexcept
	{
		return num_residual_;
	}
}
