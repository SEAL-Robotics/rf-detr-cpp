#pragma once

#include "rfdetr/core/types.hpp"

#include <cstddef>
#include <vector>

namespace rfdetr {

// RF-DETR's class head emits `num_classes + 1` logits per query. The leading
// slot (index 0) is the no-object / background class — match upstream
// `roboflow/rf-detr` Python `PostProcess` which slices `[..., 1:]`.
inline constexpr int kCocoBackgroundIndex = 0;

// Parameters for RF-DETR detection decoding.
struct PostprocessParams {
    int   num_queries{300};       // model output dim 1 (e.g. 300 for nano/small/medium/base/large)
    int   num_classes_with_bg{91};// model output dim 2 (e.g. 91 = 90 fg + 1 bg)
    int   topk{300};              // num_select; default = num_queries (Python reference)
    float threshold{0.5f};        // score threshold
    int   bg_class_index{kCocoBackgroundIndex};  // raw class id treated as bg; -1 disables filtering

    // --- hierarchical (parent/fine) decode -----------------------------------
    // Set from the engine meta sidecar's `parent_class_index` /
    // `fine_class_indices` / `fine_conf_threshold` (schema_version >= 2).
    //
    // When parent_class_index < 0 (the default) decoding is unchanged: rank every
    // (query, class) pair by its own logit, exactly as before.
    //
    // When it is >= 0 the model was trained with a PARENT class that is positive
    // for every instance plus FINE classes that are supervised only where the
    // distinction exists (pin_any vs pin_up/pin_down; see the pin_detector repo's
    // CLAUDE.md). The columns are independent sigmoids, NOT a softmax, so one
    // query legitimately scores high on the parent AND on a fine class -- and the
    // old all-pairs ranking therefore emits TWO detections for one physical
    // object, with different class ids. Hierarchical decoding instead ranks
    // queries by the parent column alone and reports the fine class as an
    // attribute of that single detection, falling back to the parent id when
    // neither fine column is confident. That fallback is meaningful, not a
    // failure: it is an explicit "object detected, sub-state unclear", which is
    // the honest output on input the fine classifier was never trained for.
    int   parent_class_index{-1};       // raw class id of the parent; <0 = flat decode
    std::vector<int> fine_class_indices;// raw class ids of the fine classes
    float fine_conf_threshold{0.0f};    // fine sigmoid must reach this to be reported
};

// Map raw class id (0..num_classes_with_bg-1) to dense user-facing class id
// (0..num_classes-1), so callers never see the no-object slot. Only shifts
// indices that sit *after* the no-object slot -- correct regardless of
// whether it's first (bg_class_index==0, e.g. some legacy/COCO-pretrained
// checkpoints) or last (bg_class_index==num_classes_with_bg-1, RF-DETR's
// actual architecture: the class head always appends a +1 no-object slot at
// the end -- see rfdetr's own criterion.py and its ONNX reference decoder,
// which universally drops the *last* logit column). The previous unconditional
// `raw_class - 1` only handled the background-first case.
inline int dense_class_from_raw(int raw_class, int bg_class_index) noexcept {
    if (bg_class_index < 0) return raw_class;
    return (raw_class < bg_class_index) ? raw_class : raw_class - 1;
}

// Decode RF-DETR detection outputs (CPU implementation).
//
// Inputs (host pointers, contiguous):
//   dets   : (num_queries, 4) float32, cxcywh normalized to [0,1]
//   labels : (num_queries, num_classes_with_bg) float32, raw logits (focal-style head)
//
// img_w / img_h: original-image dims for box rescaling.
//
// Output: dense Detections with xyxy boxes in original-image pixel coordinates.
Detections decode_detections(const float* dets, const float* labels, int img_w,
                              int img_h, const PostprocessParams& params);

// Same as decode_detections, but also returns the query index for each kept
// detection (parallel to the returned vector). Needed for segmentation mask
// gathering — the upstream Python PostProcess uses `topk_boxes = topk_idx // C`
// to gather the corresponding per-query mask.
Detections decode_detections_with_queries(const float* dets, const float* labels,
                                           int img_w, int img_h,
                                           const PostprocessParams& params,
                                           std::vector<int>& out_query_idx);

// Decode RF-DETR segmentation masks for an existing list of detections.
//
// Inputs:
//   masks_logits  : (num_queries, mask_h, mask_w) float32, raw logits, host pointer.
//   query_indices : per-detection query index (parallel to `detections`).
//   mask_h/mask_w : mask resolution from the engine bindings.
//   img_w/img_h   : original image size.
//   stream        : when non-null, offloads bilinear upsample + threshold to the GPU.
//                   Pass nullptr to use the CPU fallback.
//
// Each detection's `.mask` field is populated with a CV_8UC1 binary mask of
// size (img_h, img_w), values 0 or 255. Matches the upstream Python behavior:
// bilinear upsample to full image, threshold at logit > 0 (sigmoid > 0.5),
// masks are NOT bbox-cropped.
void decode_masks(const float* masks_logits, int mask_h, int mask_w,
                  const std::vector<int>& query_indices,
                  int img_w, int img_h, Detections& detections,
                  void* stream = nullptr);

}  // namespace rfdetr
