#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace spanner
{
    namespace detail
    {
        constexpr int kBlockSize = 256;

        inline int grid_size(int n)
        {
            return (n + kBlockSize - 1) / kBlockSize;
        }

        inline void check_cuda(cudaError_t status,const char* message)
        {
            if(status != cudaSuccess)
            {
                throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
            }
        }
        template<typename T>
        class DeviceBuffer
        {
            public:
                DeviceBuffer() = default;

                explicit DeviceBuffer(std::size_t count)
                {
                    allocate(count);
                }

                ~DeviceBuffer()
                {
                    reset();
                }

                DeviceBuffer(const DeviceBuffer&) = delete;
                DeviceBuffer& operator=(const DeviceBuffer&) = delete;

                DeviceBuffer(DeviceBuffer&& other) noexcept
                    : ptr_(other.ptr_)
                {
                    other.ptr_ = nullptr;
                }

                DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
                {
                    if(this != &other)
                    {
                        reset();
                        ptr_ = other.ptr_;
                        other.ptr_ = nullptr;
                    }

                    return *this;
                }

                void allocate(std::size_t count)
                {
                    reset();

                    check_cuda(cudaMalloc(&ptr_,count * sizeof(T)),"Failed to allocate device buffer");
                }

                void reset()
                {
                    if(ptr_ != nullptr)
                    {
                        cudaFree(ptr_);
                        ptr_ = nullptr;
                    }
                }

                T* get() const noexcept
                {
                    return ptr_;
                }

            private:
                T* ptr_ = nullptr;
        };

        void calc_coverage(int num_vertices,int radius,const int* offsets,const int* neighbors,const int* active,const long long* active_neighbor_counts,long long* coverage,int* max_flag);
    }
}