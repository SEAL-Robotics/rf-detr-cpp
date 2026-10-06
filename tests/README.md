# tests

## With CMake (needs TensorRT, because the root `CMakeLists.txt` does)

```sh
cmake -S . -B build -DRFDETR_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

## Standalone (no TensorRT, no CUDA, no CMake)

`test_engine_meta` deliberately has no dependency beyond nlohmann_json, so it
can be built on any machine — including the one the ONNX exporter runs on,
which is where the sidecar schema actually gets changed:

```sh
g++ -std=c++17 -Iinclude -o /tmp/test_engine_meta \
    tests/test_engine_meta.cpp src/core/engine_meta.cpp
/tmp/test_engine_meta
```

## GPU tests

`test_shared_engine` builds two task objects on one `SharedEngine` and checks
that, run concurrently on two threads with different images, each reproduces a
private-engine instance exactly; it also checks the shared pair uses less
device memory than two private engines, and prints serial vs concurrent timings.
`test_mask_decode` checks the GPU mask decode against a CPU reference across
chunk boundaries, and that once a resolution has been decoded, frames with more
detections allocate no further device memory. It needs no engine, only a GPU.

Both are built only when an engine is named:

```sh
cmake -S . -B build -DRFDETR_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
    -DRFDETR_TEST_ENGINE=/path/to/model.engine   # sidecar at model.engine.json
```

## What is NOT covered

`require_meta_matches_engine` / `resolve_bg_class_index`
(`src/internal/meta_validate.hpp`) need a live `TrtSession`, i.e. a real engine
on a real GPU, so they are not tested here. The part of the resolution check
that carries the logic — `describe_input_resolution_mismatch` — was deliberately
given a plain-`int` signature so that it could be, and is.
