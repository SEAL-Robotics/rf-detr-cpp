# tests/CMakeLists.txt — included from root when RFDETR_BUILD_TESTS=ON.
#
# test_engine_meta compiles src/core/engine_meta.cpp directly rather than
# linking rfdetr::rfdetr. EngineMeta's JSON handling is pure C++17 +
# nlohmann_json with no CUDA or TensorRT in it, and keeping the test out of
# librfdetr's link closure is what lets it build and run on a developer machine
# that has no TensorRT at all — which is where sidecar-schema mistakes are made.
# (The root CMakeLists still requires TensorRT to configure; building this file
# standalone is documented in tests/README.md.)

add_executable(test_engine_meta
    test_engine_meta.cpp
    ${PROJECT_SOURCE_DIR}/src/core/engine_meta.cpp
)
target_include_directories(test_engine_meta PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(test_engine_meta PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(test_engine_meta PRIVATE cxx_std_17)

add_test(NAME engine_meta COMMAND test_engine_meta)
