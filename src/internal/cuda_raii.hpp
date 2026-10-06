#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include <type_traits>

namespace rfdetr {
namespace detail {

struct DevFree {
    void operator()(void* p) const noexcept { if (p) cudaFree(p); }
};

struct HostFree {
    void operator()(void* p) const noexcept { if (p) cudaFreeHost(p); }
};

struct StreamDestroy {
    void operator()(cudaStream_t s) const noexcept { if (s) cudaStreamDestroy(s); }
};

}  // namespace detail

// Owning device-memory pointer.  Drop-in for void* managed with cudaMalloc/cudaFree.
using DevPtr  = std::unique_ptr<void, detail::DevFree>;

// Owning pinned-host-memory pointer.  Managed with cudaMallocHost/cudaFreeHost.
using HostPtr = std::unique_ptr<void, detail::HostFree>;

// Owning CUDA stream handle.
using StreamPtr = std::unique_ptr<std::remove_pointer_t<cudaStream_t>, detail::StreamDestroy>;

}  // namespace rfdetr
