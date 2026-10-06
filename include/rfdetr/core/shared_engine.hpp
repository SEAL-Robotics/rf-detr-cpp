#pragma once

#include <filesystem>
#include <memory>

namespace rfdetr {

namespace detail {
struct EngineHandle;
}  // namespace detail

// A deserialised engine that several RFDetrDetector / RFDetrSegmenter instances
// share. The weights live on the device once; every task object built from the
// handle creates its own execution context, CUDA stream and buffers, so
// instances driven from different threads infer concurrently.
//
// Thread safety: the handle may be copied and used from any thread. Each task
// object built from it is still single-threaded — one instance per thread.
// Copies refer to the same engine, which lives until the last copy and the last
// task object built from it are gone.
class SharedEngine {
   public:
    explicit SharedEngine(const std::filesystem::path& engine_path);

    [[nodiscard]] const std::filesystem::path& path() const noexcept;

    // Library-internal: what the task classes build their sessions from.
    [[nodiscard]] const std::shared_ptr<detail::EngineHandle>& handle() const noexcept {
        return handle_;
    }

   private:
    std::shared_ptr<detail::EngineHandle> handle_;
};

}  // namespace rfdetr
