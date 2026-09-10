#include "rfdetr/core/engine_meta.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>

namespace rfdetr {

using nlohmann::json;

EngineMeta EngineMeta::from_json_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("rfdetr: cannot open meta sidecar: " + path.string());
    }
    json j;
    in >> j;

    EngineMeta m;
    m.schema_version = j.value("schema_version", 0);
    m.variant     = j.value("variant", std::string{});
    m.input_h     = j.value("input_h", 0);
    m.input_w     = j.value("input_w", 0);
    m.num_queries = j.value("num_queries", 0);
    m.num_classes = j.value("num_classes", 0);
    m.parent_class_index   = j.value("parent_class_index", -1);
    m.fine_class_indices   = j.value("fine_class_indices", std::vector<int>{});
    m.fine_conf_threshold  = j.value("fine_conf_threshold", 0.0f);
    m.class_names          = j.value("class_names", std::vector<std::string>{});

    // Presence, not schema_version, is the signal: the exporter has bumped the
    // version for reasons unrelated to this field before, and a hand-edited or
    // partially-migrated sidecar should still be read for what it actually
    // says. Absent stays absent -- see EngineMeta::input_monochrome.
    if (j.contains("input_monochrome") && j["input_monochrome"].is_boolean()) {
        m.input_monochrome = j["input_monochrome"].get<bool>();
    }
    if (j.contains("luma_weights") && j["luma_weights"].is_array() &&
        j["luma_weights"].size() == 3) {
        m.luma_weights = j["luma_weights"].get<std::array<float, 3>>();
    }

    if (j.contains("mean") && j["mean"].is_array() && j["mean"].size() == 3) {
        m.mean = j["mean"].get<std::array<float, 3>>();
    }
    if (j.contains("std") && j["std"].is_array() && j["std"].size() == 3) {
        m.std = j["std"].get<std::array<float, 3>>();
    }

    m.color_order   = j.value("color_order", std::string{"RGB"});
    m.precision     = j.value("precision", std::string{"fp16"});
    m.patch_size    = j.value("patch_size", 0);
    m.dynamic_batch      = j.value("dynamic_batch", false);
    m.min_batch          = j.value("min_batch", 1);
    m.opt_batch          = j.value("opt_batch", 1);
    m.max_batch          = j.value("max_batch", 1);
    m.cuda_graph_compat  = j.value("cuda_graph_compat", false);
    m.has_masks          = j.value("has_masks", false);
    m.mask_h             = j.value("mask_h", 0);
    m.mask_w             = j.value("mask_w", 0);
    return m;
}

void EngineMeta::to_json_file(const std::filesystem::path& path) const {
    json j = {
        {"schema_version", schema_version},
        {"variant", variant},
        {"input_h", input_h},
        {"input_w", input_w},
        {"num_queries", num_queries},
        {"num_classes", num_classes},
        {"mean", mean},
        {"std", std},
        {"color_order", color_order},
        {"precision", precision},
        {"patch_size", patch_size},
        {"dynamic_batch", dynamic_batch},
        {"min_batch", min_batch},
        {"opt_batch", opt_batch},
        {"max_batch", max_batch},
        {"cuda_graph_compat", cuda_graph_compat},
        {"has_masks", has_masks},
        {"mask_h", mask_h},
        {"mask_w", mask_w},
        {"parent_class_index", parent_class_index},
        {"fine_class_indices", fine_class_indices},
        {"fine_conf_threshold", fine_conf_threshold},
        {"class_names", class_names},
    };

    // Emitted only when actually known. rfdetr_build reads a sidecar into this
    // struct and writes it back out, so anything not represented here is
    // dropped on the way from the exported ONNX meta to the engine meta the
    // runtime loads -- which is exactly why these two live in EngineMeta and
    // not only in the exporter. Writing `false` for an absent key would turn
    // "unknown" into a positive claim of colour training during that copy.
    if (input_monochrome.has_value()) {
        j["input_monochrome"] = *input_monochrome;
        if (*input_monochrome) {
            j["luma_weights"] = luma_weights;
        }
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("rfdetr: cannot write meta sidecar: " + path.string());
    }
    out << j.dump(2) << '\n';
}

}  // namespace rfdetr
