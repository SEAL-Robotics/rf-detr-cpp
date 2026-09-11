#pragma once

#include <array>
#include <vector>
#include <filesystem>
#include <optional>
#include <string>

namespace rfdetr {

// Version stamped on a meta this library SYNTHESIZES from nothing (rfdetr_build
// with no input sidecar). A value parsed from a sidecar always wins, so a v1..v4
// file round-trips carrying its own version.
//
// Deliberately NOT bumped to 4 along with the fields below. A synthesized meta
// cannot know input_monochrome, and the exporter's v4 contract is that the key is
// always present -- so stamping 4 on a meta that omits it would destroy the one
// inference "key absent => predates schema 4" that the optional exists to
// support. Nothing here or in the exporter branches on this number anyway; both
// sides key off a field's PRESENCE (see from_json_file), which is what makes
// old and new sidecars mutually readable in both directions.
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

    // Raw logit column the class head uses for no-object. RF-DETR appends it
    // LAST, so the tasks derive it as num_classes_with_bg - 1 off the engine's
    // own `labels` binding. This field is the exporter's explicit statement of
    // the same thing; when both are available the tasks require them to agree
    // rather than silently preferring one, which turns the derivation from an
    // assumption into a checked invariant.
    //
    // std::optional for the same reason as input_monochrome below: a v1..v3
    // sidecar simply does not carry it, and defaulting to 0 (or to -1, which
    // DISABLES background filtering in PostprocessParams) would be a positive
    // claim this reader has no evidence for.
    std::optional<int> bg_class_index;

    // Pin-type (CAD identity) head (schema_version >= 3). This library does not
    // decode the `pin_types` output yet -- these five exist so that
    // rfdetr_build, which parses the exporter's sidecar into this struct and
    // writes it straight back out, stops ERASING them on the way to the engine
    // sidecar the runtime and any downstream consumer read.
    bool has_pin_types{false};
    std::string pin_type_output_name;              // graph output name, e.g. "pin_types"
    std::vector<std::string> pin_type_names;       // logit-column order, index == column
    int num_pin_types{0};

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

// Describe a disagreement between the sidecar's declared input resolution and
// the resolution the TensorRT engine's input binding actually has, or
// std::nullopt when they agree.
//
// This is the only cross-check on a number that is otherwise trusted absolutely:
// meta.input_h sizes the preprocessor, which writes 3*input_h*input_w floats into
// a device buffer sized from the ENGINE's binding. Disagreement is never benign
// in either direction -- too small feeds the engine uninitialised cudaMalloc
// memory for the remainder of its input (detections computed from noise, no
// error anywhere), too large is a multi-megabyte out-of-bounds device write. A
// static-batch engine never reaches set_input_shape(), which is the one call
// that would make TensorRT validate this itself.
//
// engine_h / engine_w <= 0 means the binding is dynamic and unresolved; there is
// nothing to compare, so the result is std::nullopt.
//
// Deliberately takes plain ints, not a TrtSession/nvinfer1 type, so this header
// stays TensorRT-free (see AGENTS.md) and the caller can be a task, rfdetr_build,
// rfdetr_inspect, or a test.
[[nodiscard]] std::optional<std::string> describe_input_resolution_mismatch(
    const EngineMeta& meta, int engine_h, int engine_w);

}  // namespace rfdetr
