#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <optional>
#include <string>

namespace rfdetr {

// Bump when the JSON schema gains fields that would break older readers.
inline constexpr int kEngineMetaSchemaVersion = 1;

struct EngineMeta {
    int schema_version{kEngineMetaSchemaVersion};
    std::string variant;            // canonical name, e.g. "small", "large"
    int input_h{0};
    int input_w{0};
    int num_queries{0};
    int num_classes{0};

    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> std{0.229f, 0.224f, 0.225f};
    std::string color_order{"RGB"};

    std::string precision{"fp16"};  // "fp32" / "fp16" / "int8"
    int patch_size{0};
    bool dynamic_batch{false};
    int min_batch{1};
    int opt_batch{1};
    int max_batch{1};
    bool cuda_graph_compat{false};  // meta label set by rfdetr_build --cuda-graph

    // Segmentation outputs (filled for seg-* variants, zero otherwise).
    bool has_masks{false};
    int  mask_h{0};
    int  mask_w{0};

    // Hierarchical parent/fine class scheme (schema_version >= 2). Defaults keep
    // flat behaviour, so a v1 sidecar decodes exactly as before -- the parser uses
    // j.value(key, default) throughout and never rejects unknown keys, so old and
    // new sidecars are mutually compatible in both directions.
    // parent_class_index < 0 means "no hierarchy"; see PostprocessParams in
    // core/postprocess.hpp for what the decode then does.
    int parent_class_index{-1};
    std::vector<int> fine_class_indices;
    float fine_conf_threshold{0.0f};
    std::vector<std::string> class_names;

    // Monochrome input contract (schema_version >= 4). Nothing here changes
    // preprocessing or the engine: a mono-trained RF-DETR still takes the same
    // (1,3,H,W) input, each of the three channels carrying the same replicated
    // luma. It exists because that makes a mono-trained and a colour-trained
    // engine structurally indistinguishable -- same input shape, same outputs --
    // so feeding one the wrong kind of frame raises no error anywhere, it just
    // degrades detections quietly. This is the only thing a consumer can check
    // the incoming stream against.
    //
    // std::optional, not bool, because "key absent" (every sidecar written
    // before schema 4) has to stay distinguishable from an explicit false.
    // Defaulting to false would relabel every legacy engine as colour-trained,
    // which is a claim this reader has no evidence for and which a consumer
    // would then act on.
    std::optional<bool> input_monochrome;

    // RGB->luma coefficients the training-time export used; only meaningful
    // when input_monochrome is true. Default is BT.601, which is also what
    // OpenCV's COLOR_BGR2GRAY hardcodes -- so a consumer forced to convert
    // colour frames itself reproduces the training transform instead of a
    // near-miss.
    std::array<float, 3> luma_weights{0.299f, 0.587f, 0.114f};

    static EngineMeta from_json_file(const std::filesystem::path& path);
    void to_json_file(const std::filesystem::path& path) const;
};

}  // namespace rfdetr
