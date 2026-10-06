// GPU mask decode through a persistent scratch: correct across chunk
// boundaries, and allocation-free once a resolution has been seen.
//
// Needs a GPU but no engine; registered alongside the other GPU tests (see
// tests/CMakeLists.txt).
//
// What is asserted:
//   * every detection's mask matches a CPU reference of the same bilinear
//     upsample + threshold, for counts below, at and above one chunk, and a
//     query index of -1 leaves its mask empty;
//   * after the first decode at a resolution, decoding up to every query
//     changes neither device free memory nor the result -- the decode does not
//     allocate however many detections a frame brings.

#include "internal/mask_decode.cuh"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace {

int g_failures = 0;

void report(bool ok, const char* expr, const char* file, int line) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
}

#define CHECK(expr) report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

constexpr int kQueries = 60;
constexpr int kMaskH = 24, kMaskW = 32;
constexpr int kImgH = 120, kImgW = 160;

// Mirrors maskDecodeKernel, which mirrors OpenCV INTER_LINEAR; returns the
// interpolated logit.
float cpu_logit(const std::vector<float>& logits, int q, int x, int y) {
    const float sx_f = (x + 0.5f) * (static_cast<float>(kMaskW) / kImgW) - 0.5f;
    const float sy_f = (y + 0.5f) * (static_cast<float>(kMaskH) / kImgH) - 0.5f;
    const int sx0 = std::max(0, std::min(kMaskW - 1, static_cast<int>(std::floor(sx_f))));
    const int sy0 = std::max(0, std::min(kMaskH - 1, static_cast<int>(std::floor(sy_f))));
    const int sx1 = std::min(kMaskW - 1, sx0 + 1);
    const int sy1 = std::min(kMaskH - 1, sy0 + 1);
    const float fx = std::max(0.0f, std::min(1.0f, sx_f - sx0));
    const float fy = std::max(0.0f, std::min(1.0f, sy_f - sy0));
    const float* p = logits.data() + static_cast<std::size_t>(q) * kMaskH * kMaskW;
    const float v = p[sy0 * kMaskW + sx0] * (1 - fx) * (1 - fy) +
                    p[sy0 * kMaskW + sx1] * fx * (1 - fy) +
                    p[sy1 * kMaskW + sx0] * (1 - fx) * fy +
                    p[sy1 * kMaskW + sx1] * fx * fy;
    return v;
}

// The GPU contracts to FMA, so a logit within rounding of 0 may threshold
// either way; only pixels clear of it are compared.
constexpr float kAmbiguousLogit = 1e-3f;

// Logits with a different blob per query, far from 0 except at the edges, so
// a mask decoded from the wrong query plane cannot match by accident.
std::vector<float> make_logits() {
    std::vector<float> logits(static_cast<std::size_t>(kQueries) * kMaskH * kMaskW);
    for (int q = 0; q < kQueries; ++q) {
        const float cx = static_cast<float>((q * 7) % kMaskW);
        const float cy = static_cast<float>((q * 5) % kMaskH);
        const float r = 3.0f + static_cast<float>(q % 6);
        for (int y = 0; y < kMaskH; ++y) {
            for (int x = 0; x < kMaskW; ++x) {
                const float d = std::hypot(x - cx, y - cy);
                logits[(static_cast<std::size_t>(q) * kMaskH + y) * kMaskW + x] = r - d;
            }
        }
    }
    return logits;
}

// Decodes `count` detections (query i -> (i * 13) % kQueries, every ninth -1)
// and checks each mask against the CPU reference.
void decode_and_check(rfdetr::MaskDecodeScratch* scratch, cudaStream_t stream,
                      const std::vector<float>& logits, int count) {
    rfdetr::Detections dets(static_cast<std::size_t>(count));
    std::vector<int> query(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) query[i] = (i % 9 == 8) ? -1 : (i * 13) % kQueries;

    rfdetr::gpu_decode_masks(logits.data(), query, kMaskH, kMaskW, kImgH, kImgW, dets,
                             scratch, stream);

    int mismatches = 0;
    for (int i = 0; i < count; ++i) {
        const cv::Mat& m = dets[static_cast<std::size_t>(i)].mask;
        if (query[i] < 0) {
            CHECK(m.empty());
            continue;
        }
        CHECK(m.rows == kImgH && m.cols == kImgW && m.type() == CV_8UC1);
        if (m.rows != kImgH || m.cols != kImgW) continue;
        for (int y = 0; y < kImgH; ++y) {
            for (int x = 0; x < kImgW; ++x) {
                const float v = cpu_logit(logits, query[i], x, y);
                if (std::fabs(v) < kAmbiguousLogit) continue;
                if (m.at<std::uint8_t>(y, x) != (v > 0.0f ? 255u : 0u)) ++mismatches;
            }
        }
    }
    if (mismatches) std::fprintf(stderr, "  %d mismatching pixels at count=%d\n", mismatches, count);
    CHECK(mismatches == 0);
}

std::size_t device_free() {
    std::size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    return free_b;
}

}  // namespace

int main() {
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "no usable CUDA device\n");
        return 1;
    }
    const std::vector<float> logits = make_logits();
    const int chunk = static_cast<int>(rfdetr::mask_decode_chunk());

    {
        auto scratch = rfdetr::make_mask_decode_scratch(kQueries, kMaskH, kMaskW);
        // The first decode at this resolution is the one allowed to allocate.
        decode_and_check(scratch.get(), stream, logits, 1);
        const std::size_t warm = device_free();
        for (int count : {chunk - 1, chunk, chunk + 1, 2 * chunk + 3, kQueries, 1}) {
            decode_and_check(scratch.get(), stream, logits, count);
        }
        CHECK(device_free() == warm);
    }

    // The call-scoped fallback decodes the same masks.
    decode_and_check(nullptr, stream, logits, chunk + 5);

    cudaStreamDestroy(stream);
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("mask_decode: all checks passed\n");
    return 0;
}
