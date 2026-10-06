// Two task objects on one SharedEngine: correct, independent, concurrent, and
// cheaper on the device than two private engines.
//
// Needs a real engine on a real GPU, so it is registered only when
// RFDETR_TEST_ENGINE names one (see tests/CMakeLists.txt). The sidecar is
// <engine>.json; its has_masks picks RFDetrSegmenter or RFDetrDetector.
//
// What is asserted:
//   * each shared-engine instance reproduces a private-engine instance's
//     output exactly, frame for frame, while the two run on different images
//     on two threads at once — so contexts on one engine do not cross-talk;
//   * a moved TrtSession (every shared-engine construction moves one) leaves
//     nothing behind that a second destructor could double-free;
//   * one engine + two contexts costs less device memory than two engines.
// Wall-clock timings for serial vs concurrent are printed, not asserted: how
// much two contexts overlap depends on how much of the GPU one frame fills.

#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/shared_engine.hpp"
#include "rfdetr/tasks/detector.hpp"
#include "rfdetr/tasks/segmenter.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void report(bool ok, const char* expr, const char* file, int line) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
}

#define CHECK(expr) report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

constexpr float kThreshold = 0.05f;  // low, so the comparison sees many detections
constexpr int kConcurrentIters = 40;

// Structured, not noise, so the model produces detections worth comparing,
// and different per seed so two threads demonstrably run different inputs.
cv::Mat synthetic_frame(int seed, int size) {
    cv::Mat img(size, size, CV_8UC3, cv::Scalar(40, 40, 40));
    cv::RNG rng(static_cast<uint64_t>(seed) * 7919u + 17u);
    for (int i = 0; i < 24; ++i) {
        const cv::Point c(rng.uniform(0, size), rng.uniform(0, size));
        const int r = rng.uniform(size / 80, size / 20);
        const int v = rng.uniform(120, 255);
        cv::circle(img, c, r, cv::Scalar(v, v, v), cv::FILLED);
        cv::circle(img, c, r / 3, cv::Scalar(20, 20, 20), cv::FILLED);
    }
    return img;
}

bool same_detections(const rfdetr::Detections& a, const rfdetr::Detections& b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto& x = a[i];
        const auto& y = b[i];
        if (x.class_id != y.class_id || x.score != y.score || x.box.x1 != y.box.x1 ||
            x.box.y1 != y.box.y1 || x.box.x2 != y.box.x2 || x.box.y2 != y.box.y2) {
            return false;
        }
        if (x.mask.empty() != y.mask.empty()) return false;
        if (!x.mask.empty()) {
            if (x.mask.size() != y.mask.size() || x.mask.type() != y.mask.type()) return false;
            if (cv::countNonZero(x.mask != y.mask) != 0) return false;
        }
    }
    return true;
}

// Detector and segmenter share no base class; this gives the test one shape.
class Model {
   public:
    virtual ~Model() = default;
    virtual rfdetr::Detections run(const cv::Mat& img) = 0;
};

template <class Task>
class TaskModel final : public Model {
   public:
    template <class... Args>
    explicit TaskModel(Args&&... args) : task_(std::forward<Args>(args)...) {}
    rfdetr::Detections run(const cv::Mat& img) override;

   private:
    Task task_;
};

template <>
rfdetr::Detections TaskModel<rfdetr::RFDetrDetector>::run(const cv::Mat& img) {
    return task_.detect(img, kThreshold);
}

template <>
rfdetr::Detections TaskModel<rfdetr::RFDetrSegmenter>::run(const cv::Mat& img) {
    return task_.segment(img, kThreshold);
}

std::unique_ptr<Model> make_private(bool masks, const std::filesystem::path& engine,
                                    const std::filesystem::path& meta) {
    if (masks) return std::make_unique<TaskModel<rfdetr::RFDetrSegmenter>>(engine, meta);
    return std::make_unique<TaskModel<rfdetr::RFDetrDetector>>(engine, meta);
}

std::unique_ptr<Model> make_shared(bool masks, const rfdetr::SharedEngine& engine,
                                   const std::filesystem::path& meta) {
    if (masks) return std::make_unique<TaskModel<rfdetr::RFDetrSegmenter>>(engine, meta);
    return std::make_unique<TaskModel<rfdetr::RFDetrDetector>>(engine, meta);
}

std::size_t device_used() {
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    cudaDeviceSynchronize();
    cudaMemGetInfo(&free_b, &total_b);
    return total_b - free_b;
}

// On an integrated GPU (Jetson) device memory is system memory, so
// cudaMemGetInfo moves with every other process and cannot attribute a delta
// to this one.
bool device_memory_is_measurable() {
    int dev = 0;
    cudaDeviceProp prop{};
    return cudaGetDevice(&dev) == cudaSuccess &&
           cudaGetDeviceProperties(&prop, dev) == cudaSuccess && !prop.integrated;
}

double mib(std::size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <engine>\n", argv[0]);
        return 2;
    }
    const std::filesystem::path engine_path = argv[1];
    const std::filesystem::path meta_path = engine_path.string() + ".json";

    try {
        const rfdetr::EngineMeta meta = rfdetr::EngineMeta::from_json_file(meta_path);
        const bool masks = meta.has_masks;
        // The first two synthetic frames this engine finds pins in: which ones
        // clear the threshold depends on the device's numerics, and an empty
        // frame cannot show the two contexts answering for different inputs.
        std::vector<cv::Mat> frames;
        {
            auto probe = make_private(masks, engine_path, meta_path);
            for (int seed = 3; seed < 3 + 32 && frames.size() < 2; ++seed) {
                cv::Mat img = synthetic_frame(seed, meta.input_h);
                if (!probe->run(img).empty()) frames.push_back(std::move(img));
            }
        }
        CHECK(frames.size() == 2);
        if (frames.size() != 2) return 1;
        const cv::Mat& frame_a = frames[0];
        const cv::Mat& frame_b = frames[1];

        if (device_memory_is_measurable()) {
            // --- Device memory: two private engines vs one shared engine ----------
            const std::size_t base = device_used();
            std::size_t two_private = 0;
            {
                auto p1 = make_private(masks, engine_path, meta_path);
                auto p2 = make_private(masks, engine_path, meta_path);
                (void)p1->run(frame_a);
                (void)p2->run(frame_a);
                two_private = device_used() - base;
            }
            std::size_t two_shared = 0;
            {
                const rfdetr::SharedEngine engine(engine_path);
                auto s1 = make_shared(masks, engine, meta_path);
                auto s2 = make_shared(masks, engine, meta_path);
                (void)s1->run(frame_a);
                (void)s2->run(frame_a);
                two_shared = device_used() - base;
            }
            std::printf("device memory for two instances: private engines %.1f MiB, shared engine "
                        "%.1f MiB (saves %.1f MiB)\n",
                        mib(two_private), mib(two_shared),
                        mib(two_private) - mib(two_shared));
            CHECK(two_shared < two_private);
        } else {
            std::printf("device memory comparison skipped: integrated GPU\n");
        }

        // --- Reference output from a private engine ---------------------------
        rfdetr::Detections ref_a;
        rfdetr::Detections ref_b;
        {
            auto ref = make_private(masks, engine_path, meta_path);
            ref_a = ref->run(frame_a);
            ref_b = ref->run(frame_b);
        }
        std::printf("reference detections: frame A %zu, frame B %zu\n", ref_a.size(),
                    ref_b.size());
        CHECK(!ref_a.empty() && !ref_b.empty());

        const rfdetr::SharedEngine engine(engine_path);
        auto s1 = make_shared(masks, engine, meta_path);
        auto s2 = make_shared(masks, engine, meta_path);
        CHECK(same_detections(s1->run(frame_a), ref_a));
        CHECK(same_detections(s2->run(frame_b), ref_b));

        // --- Serial: one instance, both frames, back to back ------------------
        const auto t_serial = std::chrono::steady_clock::now();
        for (int i = 0; i < kConcurrentIters; ++i) {
            (void)s1->run(frame_a);
            (void)s1->run(frame_b);
        }
        const double serial_s = seconds_since(t_serial);

        // --- Concurrent: two instances, one frame each, two threads -----------
        std::atomic<int> mismatches_a{0};
        std::atomic<int> mismatches_b{0};
        const auto worker = [](Model& m, const cv::Mat& img, const rfdetr::Detections& ref,
                               std::atomic<int>& mismatches) {
            for (int i = 0; i < kConcurrentIters; ++i) {
                if (!same_detections(m.run(img), ref)) ++mismatches;
            }
        };
        const auto t_conc = std::chrono::steady_clock::now();
        std::thread ta(worker, std::ref(*s1), std::cref(frame_a), std::cref(ref_a),
                       std::ref(mismatches_a));
        std::thread tb(worker, std::ref(*s2), std::cref(frame_b), std::cref(ref_b),
                       std::ref(mismatches_b));
        ta.join();
        tb.join();
        const double conc_s = seconds_since(t_conc);

        std::printf("%d frame pairs: serial on one context %.3f s, concurrent on two contexts "
                    "%.3f s (%.2fx)\n",
                    kConcurrentIters, serial_s, conc_s, serial_s / conc_s);
        std::printf("concurrent mismatches vs reference: A %d, B %d\n", mismatches_a.load(),
                    mismatches_b.load());
        CHECK(mismatches_a.load() == 0);
        CHECK(mismatches_b.load() == 0);

        // Instances outliving the handle they were built from keep the engine.
        auto orphan = make_shared(masks, rfdetr::SharedEngine(engine_path), meta_path);
        CHECK(same_detections(orphan->run(frame_a), ref_a));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
        return 1;
    }

    if (g_failures == 0) std::printf("test_shared_engine: all checks passed\n");
    return g_failures == 0 ? 0 : 1;
}
