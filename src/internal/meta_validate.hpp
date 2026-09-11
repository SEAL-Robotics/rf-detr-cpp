#pragma once

// Constructor-time agreement checks between an EngineMeta sidecar and the
// TensorRT engine it claims to describe. Shared by RFDetrDetector::Impl and
// RFDetrSegmenter::Impl, which load the sidecar identically and would otherwise
// each carry their own copy of this (they already each carry their own copy of
// the C-1 background derivation, which is how it drifted out of sync with the
// sidecar's explicit bg_class_index in the first place).

#include "rfdetr/core/engine_meta.hpp"
#include "trt_session.hpp"

#include <NvInferRuntime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace rfdetr {

// The library addresses the network input by the literal name "input" --
// device_buffer("input") in every task's hot path, set_input_shape("input") on
// the dynamic-batch path -- and TrtSession::device_buffer throws on an unknown
// name, so an engine whose input is named anything else cannot work here at
// all. Failing in the constructor, naming what the engine actually has, beats
// the same failure surfacing from the first detect() call as a bare
// "no such binding: input".
[[nodiscard]] inline const BindingInfo& require_input_binding(const TrtSession& session,
                                                              const char* task_name) {
    if (const BindingInfo* b = session.find("input")) return *b;

    std::string names;
    for (const int idx : session.input_indices()) {
        if (!names.empty()) names += ", ";
        names += '\'' + session.bindings()[static_cast<std::size_t>(idx)].name + '\'';
    }
    throw std::runtime_error(std::string("rfdetr: ") + task_name +
                             ": engine has no input tensor named 'input' (engine inputs: " +
                             (names.empty() ? std::string("none") : names) +
                             "); re-export the ONNX with that input name");
}

// Throw unless the sidecar's declared input geometry matches the engine's own
// input binding.
//
// This is the only thing standing between a stale/hand-edited sidecar and
// silent garbage: meta.input_h sizes the ImagePreprocessor, which writes
// 3*input_h*input_w floats into a device buffer whose size came from the
// ENGINE's binding instead. Too small and the engine consumes uninitialised
// cudaMalloc memory for the rest of its input -- plausible-looking detections
// computed from noise, no error anywhere. Too large and it is a multi-megabyte
// out-of-bounds device write. TensorRT validates none of this for a
// static-shape engine, because set_input_shape() -- the one call that would --
// is only reached when meta.dynamic_batch is true.
inline void require_meta_matches_engine(const TrtSession& session, const EngineMeta& meta,
                                        const char* task_name) {
    const BindingInfo& in_b = require_input_binding(session, task_name);
    if (in_b.shape.nbDims < 4) {
        throw std::runtime_error(std::string("rfdetr: ") + task_name +
                                 ": engine input 'input' has " +
                                 std::to_string(in_b.shape.nbDims) +
                                 " dims, expected 4 (N,C,H,W)");
    }

    const int engine_c = static_cast<int>(in_b.shape.d[1]);
    const int engine_h = static_cast<int>(in_b.shape.d[in_b.shape.nbDims - 2]);
    const int engine_w = static_cast<int>(in_b.shape.d[in_b.shape.nbDims - 1]);

    // <= 0 means the dim is dynamic and not yet resolved; nothing to compare.
    if (auto why = describe_input_resolution_mismatch(meta, engine_h, engine_w)) {
        throw std::runtime_error(
            std::string("rfdetr: ") + task_name + ": " + *why +
            ". Rebuild the engine from the ONNX this sidecar belongs to, or correct the "
            "sidecar to match the engine. Do NOT edit it back to the variant's default "
            "resolution -- a fine-tuned checkpoint legitimately overrides that.");
    }

    // ImagePreprocessor always emits 3 planes (see launch_square_resize_normalize),
    // including for monochrome-trained models, which replicate luma across all
    // three. A 1-channel engine would be under-filled by exactly the same
    // mechanism as a resolution mismatch.
    if (engine_c > 0 && engine_c != 3) {
        throw std::runtime_error(std::string("rfdetr: ") + task_name +
                                 ": engine input has " + std::to_string(engine_c) +
                                 " channels, but preprocessing always writes 3");
    }
}

// Reconcile the sidecar's explicit background column with the one derived from
// the engine's own `labels` binding width.
//
// RF-DETR appends the no-object slot LAST, so num_classes_with_bg - 1 is
// correct -- but it is an assumption about an architecture detail, and the
// exporter states the same number outright (schema_version >= 4). Requiring
// them to agree converts the assumption into a checked invariant; the failure
// it catches (a head layout where background is first, i.e. every class id off
// by one) is otherwise completely silent, since both readings produce valid
// in-range class ids.
[[nodiscard]] inline int resolve_bg_class_index(const EngineMeta& meta,
                                                int num_classes_with_bg,
                                                const char* task_name) {
    const int derived = num_classes_with_bg - 1;
    // An unresolved labels binding makes `derived` meaningless; leave the
    // pre-existing behaviour alone rather than inventing a new failure here.
    if (num_classes_with_bg <= 0 || !meta.bg_class_index.has_value()) return derived;

    if (*meta.bg_class_index != derived) {
        throw std::runtime_error(
            std::string("rfdetr: ") + task_name + ": meta sidecar says bg_class_index=" +
            std::to_string(*meta.bg_class_index) + " but the engine's 'labels' tensor is " +
            std::to_string(num_classes_with_bg) +
            " wide, which puts the no-object column at " + std::to_string(derived) +
            "; sidecar and engine describe different class heads");
    }
    return *meta.bg_class_index;
}

}  // namespace rfdetr
