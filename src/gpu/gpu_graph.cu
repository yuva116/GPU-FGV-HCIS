#include "gpu/gpu_graph.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " +cudaGetErrorString(status));
    }
}

} 

namespace spanner {

GPUGraph::GPUGraph(const CSRGraph& graph): num_vertices_(graph.num_vertices()),num_edges_(graph.num_edges()),num_adjacency_entries_(graph.num_adjacency_entries()) {

    std::vector<int> offsets(num_vertices_ + 1);

    for (std::size_t i = 0; i <= num_vertices_; ++i) {
        offsets[i] =
            static_cast<int>(graph.offsets()[i]);
    }

    check_cuda(cudaMalloc(&d_offsets_,offsets.size() * sizeof(int)),"Failed to allocate device offsets");

    check_cuda(cudaMalloc(&d_neighbors_,num_adjacency_entries_ * sizeof(int)),"Failed to allocate device neighbors");

    check_cuda(cudaMemcpy(d_offsets_,offsets.data(),offsets.size() * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy offsets to device");

    check_cuda(cudaMemcpy(d_neighbors_,graph.neighbors_data().data(),num_adjacency_entries_ * sizeof(int),cudaMemcpyHostToDevice),"Failed to copy neighbors to device");
}

GPUGraph::~GPUGraph() 
{
    cudaFree(d_offsets_);
    cudaFree(d_neighbors_);
}

std::size_t GPUGraph::num_vertices() const noexcept {
    return num_vertices_;
}

std::size_t GPUGraph::num_edges() const noexcept {
    return num_edges_;
}

std::size_t GPUGraph::num_adjacency_entries() const noexcept {
    return num_adjacency_entries_;
}

const int* GPUGraph::offsets() const noexcept {
    return d_offsets_;
}

const int* GPUGraph::neighbors() const noexcept {
    return d_neighbors_;
}

} // namespace spanner