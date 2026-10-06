#include "internal/mask_decode.cuh"
#include "internal/cuda_check.hpp"
#include "internal/cuda_raii.hpp"

#include "rfdetr/core/types.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace rfdetr {

namespace {

// Each thread: one output pixel (x,y) for one detection (blockIdx.z).
// Grid: (ceil(imgW/TX), ceil(imgH/TY), num_dets)
__global__ void maskDecodeKernel(const float*         __restrict__ d_logits,
                                  const std::int32_t*  __restrict__ d_query_indices,
                                  int mH, int mW,
                                  int imgH, int imgW,
                                  std::uint8_t* __restrict__ d_masks_out) {
    const int x   = blockIdx.x * blockDim.x + threadIdx.x;
    const int y   = blockIdx.y * blockDim.y + threadIdx.y;
    const int det = blockIdx.z;

    if (x >= imgW || y >= imgH) return;

    const int q = d_query_indices[det];
    if (q < 0) {
        d_masks_out[det * imgH * imgW + y * imgW + x] = 0;
        return;
    }

    // Bilinear upsample: map output pixel center to source coordinates
    // using OpenCV INTER_LINEAR semantics, matching the CPU decode_masks path.
    const float scale_x = static_cast<float>(mW) / static_cast<float>(imgW);
    const float scale_y = static_cast<float>(mH) / static_cast<float>(imgH);
    const float sx_f = (static_cast<float>(x) + 0.5f) * scale_x - 0.5f;
    const float sy_f = (static_cast<float>(y) + 0.5f) * scale_y - 0.5f;

    const int sx0 = max(0, min(mW - 1, static_cast<int>(floorf(sx_f))));
    const int sy0 = max(0, min(mH - 1, static_cast<int>(floorf(sy_f))));
    const int sx1 = min(mW - 1, sx0 + 1);
    const int sy1 = min(mH - 1, sy0 + 1);
    const float fx = fmaxf(0.0f, fminf(1.0f, sx_f - static_cast<float>(sx0)));
    const float fy = fmaxf(0.0f, fminf(1.0f, sy_f - static_cast<float>(sy0)));

    const int base = q * mH * mW;
    const float v00 = __ldg(d_logits + base + sy0 * mW + sx0);
    const float v01 = __ldg(d_logits + base + sy0 * mW + sx1);
    const float v10 = __ldg(d_logits + base + sy1 * mW + sx0);
    const float v11 = __ldg(d_logits + base + sy1 * mW + sx1);

    const float logit = v00 * (1.0f - fx) * (1.0f - fy)
                      + v01 *         fx  * (1.0f - fy)
                      + v10 * (1.0f - fx) *         fy
                      + v11 *         fx  *         fy;

    // logit > 0 ↔ sigmoid(logit) > 0.5
    d_masks_out[det * imgH * imgW + y * imgW + x] = (logit > 0.0f) ? 255u : 0u;
}

}  // namespace

// cudaMalloc/cudaMallocHost synchronise the whole device, stalling every other
// segmenter's stream in the process, so a decode must not allocate. The logits
// and indices are sized at construction from the engine's bindings. Masks are
// decoded in chunks of kMaskChunk detections through one chunk-sized buffer, so
// its size depends on the frame resolution alone, never on how many detections
// a frame has: it is allocated on the first decode at a resolution, and a
// fixed-resolution camera never allocates again.
constexpr std::size_t kMaskChunk = 16;

struct MaskDecodeScratch {
    HostPtr h_logits, h_idx, h_masks;
    DevPtr  d_logits, d_idx, d_masks;
    std::size_t cap_h_logits{0}, cap_h_idx{0}, cap_h_masks{0};
    std::size_t cap_d_logits{0}, cap_d_idx{0}, cap_d_masks{0};

    static void grow_host(HostPtr& p, std::size_t& cap, std::size_t need) {
        if (cap >= need) return;
        void* raw = nullptr;
        RFDETR_CUDA_CHECK(cudaMallocHost(&raw, need));
        p.reset(raw);  // releases the previous allocation
        cap = need;
    }

    static void grow_dev(DevPtr& p, std::size_t& cap, std::size_t need) {
        if (cap >= need) return;
        void* raw = nullptr;
        RFDETR_CUDA_CHECK(cudaMalloc(&raw, need));
        p.reset(raw);
        cap = need;
    }

    void reserve(std::size_t logit_bytes, std::size_t idx_bytes, std::size_t mask_bytes) {
        grow_host(h_logits, cap_h_logits, logit_bytes);
        grow_host(h_idx,    cap_h_idx,    idx_bytes);
        grow_host(h_masks,  cap_h_masks,  mask_bytes);
        grow_dev (d_logits, cap_d_logits, logit_bytes);
        grow_dev (d_idx,    cap_d_idx,    idx_bytes);
        grow_dev (d_masks,  cap_d_masks,  mask_bytes);
    }
};

void MaskDecodeScratchDeleter::operator()(MaskDecodeScratch* p) const noexcept {
    delete p;
}

MaskDecodeScratchPtr make_mask_decode_scratch() {
    return MaskDecodeScratchPtr(new MaskDecodeScratch());
}

MaskDecodeScratchPtr make_mask_decode_scratch(int num_queries, int mH, int mW) {
    MaskDecodeScratchPtr p(new MaskDecodeScratch());
    const std::size_t q = static_cast<std::size_t>(num_queries > 0 ? num_queries : 0);
    p->reserve(q * static_cast<std::size_t>(mH) * static_cast<std::size_t>(mW) * sizeof(float),
               kMaskChunk * sizeof(std::int32_t), 0);
    return p;
}

std::size_t mask_decode_chunk() noexcept { return kMaskChunk; }

void gpu_decode_masks(const float*            h_masks_logits,
                      const std::vector<int>& query_indices,
                      int mH, int mW,
                      int imgH, int imgW,
                      Detections&             detections,
                      MaskDecodeScratch*      scratch,
                      void*                   cuda_stream) {
    const std::size_t num_dets = detections.size();
    if (num_dets == 0 || !h_masks_logits) return;
    if (mH <= 0 || mW <= 0 || imgH <= 0 || imgW <= 0) return;
    // The index-staging loop below walks num_dets entries; a short vector would
    // read out of range. Callers reaching this directly skip decode_masks()'
    // checks, so validate here too.
    if (query_indices.size() != num_dets) return;

    auto stream = static_cast<cudaStream_t>(cuda_stream);

    // Find the highest referenced query index so we upload only those planes.
    int max_q = 0;
    for (int q : query_indices) if (q > max_q) max_q = q;
    const std::size_t num_q_needed = static_cast<std::size_t>(max_q + 1);

    const std::size_t logit_bytes = num_q_needed *
                                    static_cast<std::size_t>(mH) * mW * sizeof(float);
    const std::size_t mask_plane  = static_cast<std::size_t>(imgH) * imgW;

    // Reuse the caller's staging when provided; otherwise fall back to a
    // call-scoped scratch so the standalone entry point keeps working. A
    // persistent scratch takes a full chunk at once, so a frame with more
    // detections than the last one does not allocate.
    MaskDecodeScratch  local;
    MaskDecodeScratch& buf = scratch ? *scratch : local;
    const std::size_t chunk = scratch ? kMaskChunk : std::min(kMaskChunk, num_dets);
    buf.reserve(logit_bytes, chunk * sizeof(std::int32_t), chunk * mask_plane);

    std::memcpy(buf.h_logits.get(), h_masks_logits, logit_bytes);
    RFDETR_CUDA_CHECK(cudaMemcpyAsync(buf.d_logits.get(), buf.h_logits.get(), logit_bytes,
                                       cudaMemcpyHostToDevice, stream));

    auto* h_idx_i32 = static_cast<std::int32_t*>(buf.h_idx.get());
    const auto* src = static_cast<const std::uint8_t*>(buf.h_masks.get());
    constexpr int TX = 16, TY = 16;
    const dim3 block(TX, TY, 1);

    for (std::size_t first = 0; first < num_dets; first += chunk) {
        const std::size_t n = std::min(chunk, num_dets - first);
        // The previous chunk's stream synchronise has retired every copy out
        // of h_idx and h_masks, so both are free to reuse here.
        for (std::size_t i = 0; i < n; ++i) {
            h_idx_i32[i] = static_cast<std::int32_t>(query_indices[first + i]);
        }
        RFDETR_CUDA_CHECK(cudaMemcpyAsync(buf.d_idx.get(), buf.h_idx.get(),
                                           n * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, stream));

        const dim3 grid((imgW + TX - 1) / TX,
                        (imgH + TY - 1) / TY,
                        static_cast<unsigned>(n));
        maskDecodeKernel<<<grid, block, 0, stream>>>(
            static_cast<const float*>(buf.d_logits.get()),
            static_cast<const std::int32_t*>(buf.d_idx.get()),
            mH, mW, imgH, imgW,
            static_cast<std::uint8_t*>(buf.d_masks.get()));
        RFDETR_CUDA_CHECK(cudaGetLastError());

        RFDETR_CUDA_CHECK(cudaMemcpyAsync(buf.h_masks.get(), buf.d_masks.get(), n * mask_plane,
                                           cudaMemcpyDeviceToHost, stream));
        RFDETR_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Package into cv::Mat per detection (clone from pinned buffer).
        for (std::size_t i = 0; i < n; ++i) {
            if (query_indices[first + i] < 0) continue;
            cv::Mat m(imgH, imgW, CV_8UC1);
            std::memcpy(m.data, src + i * mask_plane, mask_plane);
            detections[first + i].mask = std::move(m);
        }
    }
}

}  // namespace rfdetr
