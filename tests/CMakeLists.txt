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

# test_shared_engine needs a real engine on a real GPU, so it exists only when
# one is named: -DRFDETR_TEST_ENGINE=/path/to/model.engine (sidecar alongside).
set(RFDETR_TEST_ENGINE "" CACHE FILEPATH "Engine for GPU tests (empty: GPU tests not built)")
if(RFDETR_TEST_ENGINE)
    add_executable(test_shared_engine test_shared_engine.cpp)
    target_link_libraries(test_shared_engine PRIVATE rfdetr::rfdetr)
    add_test(NAME shared_engine COMMAND test_shared_engine ${RFDETR_TEST_ENGINE})

    # Needs the GPU but not the engine. Compiles the decode directly because
    # its symbols are internal to librfdetr.
    add_executable(test_mask_decode test_mask_decode.cpp ${PROJECT_SOURCE_DIR}/src/core/mask_decode.cu)
    target_include_directories(test_mask_decode PRIVATE
        ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(test_mask_decode PRIVATE CUDA::cudart ${OpenCV_LIBS})
    target_compile_features(test_mask_decode PRIVATE cxx_std_17)
    add_test(NAME mask_decode COMMAND test_mask_decode)
endif()
