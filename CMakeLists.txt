cmake_minimum_required(VERSION 3.20)

project(GPU_Graph_Spanner_Framework VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(spanner_graph STATIC
    src/graph/graph.cpp
    src/graph/csr.cpp
    src/graph/graph_loader.cpp
    src/validation/correctness.cpp
    src/utils/timer.cpp
)

target_include_directories(spanner_graph PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_library(spanner_gpu STATIC
    src/gpu/gpu_graph.cu
    src/gpu/edge_finalize.cu

    src/gpu/fgv/fgv_shift.cu
    src/gpu/fgv/fgv_cluster.cu
    src/gpu/fgv/fgv_spanner.cu

    src/gpu/compact/compact_centers.cu
    src/gpu/compact/compact_independent_set.cu
    src/gpu/compact/compact_cluster.cu
    src/gpu/compact/compact_spanner.cu
    src/gpu/compact/compact_hybrid.cu

    src/validation/gpu_stretch.cu

    src/algorithms/fgv/fgv.cpp
    src/algorithms/compact/compact.cpp
)

target_include_directories(spanner_gpu PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(spanner_gpu PUBLIC
    spanner_graph
    CUDA::cudart
    CUDA::curand
)

set_target_properties(spanner_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(test_csr
    tests/test_csr.cpp
)

target_link_libraries(test_csr PRIVATE
    spanner_graph
)

add_executable(test_fgv
    tests/test_fgv.cpp
)

target_link_libraries(test_fgv PRIVATE
    spanner_gpu
)

add_executable(test_compact
    tests/test_compact.cpp
)

target_link_libraries(test_compact PRIVATE
    spanner_gpu
)

if(NOT MSVC)

    target_compile_options(spanner_graph PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

    target_compile_options(test_csr PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )

endif()


add_executable(test_fgv_compact
    tests/test_fgv_compact.cpp
)

target_link_libraries(test_fgv_compact PRIVATE
    spanner_gpu
)

add_executable(run_benchmark
    benchmarks/benchmark_end_to_end.cpp
)

target_link_libraries(run_benchmark PRIVATE
    spanner_gpu
)
