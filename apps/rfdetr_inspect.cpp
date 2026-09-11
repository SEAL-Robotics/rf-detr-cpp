// rfdetr_inspect — load a serialized TensorRT engine and print its IO bindings.
// Optional second arg: a meta-sidecar JSON, which is parsed and cross-checked
// against the engine.

#include "rfdetr/core/engine_meta.hpp"
#include "rfdetr/core/trt_logger.hpp"
#include "rfdetr/core/variant.hpp"
#include "rfdetr/version.hpp"

#include <NvInferRuntime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("could not open: " + path.string());
    }
    const std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!in.read(buf.data(), size)) {
        throw std::runtime_error("short read: " + path.string());
    }
    return buf;
}

const char* mode_name(nvinfer1::TensorIOMode m) noexcept {
    switch (m) {
        case nvinfer1::TensorIOMode::kINPUT:  return "INPUT ";
        case nvinfer1::TensorIOMode::kOUTPUT: return "OUTPUT";
        default:                              return "NONE  ";
    }
}

const char* dtype_name(nvinfer1::DataType d) noexcept {
    switch (d) {
        case nvinfer1::DataType::kFLOAT:  return "fp32";
        case nvinfer1::DataType::kHALF:   return "fp16";
        case nvinfer1::DataType::kINT8:   return "int8";
        case nvinfer1::DataType::kINT32:  return "int32";
        case nvinfer1::DataType::kBOOL:   return "bool";
        case nvinfer1::DataType::kUINT8:  return "uint8";
#if NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kFP8:    return "fp8";
        case nvinfer1::DataType::kBF16:   return "bf16";
        case nvinfer1::DataType::kINT64:  return "int64";
#endif
        default:                          return "?";
    }
}

void print_shape(const nvinfer1::Dims& d) {
    std::fputc('[', stdout);
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) std::fputc(',', stdout);
        std::printf("%ld", static_cast<long>(d.d[i]));
    }
    std::fputc(']', stdout);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::fprintf(stderr,
                     "usage: %s <engine.plan> [meta.json]\n"
                     "  rfdetr v%s\n",
                     argv[0], rfdetr::version());
        return 2;
    }

    const std::filesystem::path engine_path{argv[1]};
    std::vector<char> blob;
    try {
        blob = read_file(engine_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    rfdetr::TrtLogger logger{nvinfer1::ILogger::Severity::kWARNING};
    std::unique_ptr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(logger)};
    if (!runtime) {
        std::fprintf(stderr, "error: createInferRuntime returned null\n");
        return 1;
    }
    std::unique_ptr<nvinfer1::ICudaEngine> engine{
        runtime->deserializeCudaEngine(blob.data(), blob.size())};
    if (!engine) {
        std::fprintf(stderr, "error: deserializeCudaEngine returned null\n");
        return 1;
    }

    std::printf("engine: %s\n", engine_path.c_str());
    std::printf("  trt header: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
                NV_TENSORRT_PATCH);
    std::printf("  bindings:\n");
    const int nb = engine->getNbIOTensors();
    int input_count = 0;
    int output_count = 0;
    // Engine-side facts the meta section below cross-checks the sidecar against.
    // -1 = no 4-D input binding found, or its spatial dims are dynamic.
    int engine_in_h = -1;
    int engine_in_w = -1;
    std::vector<std::string> output_names;
    for (int i = 0; i < nb; ++i) {
        const char* name = engine->getIOTensorName(i);
        const auto mode = engine->getTensorIOMode(name);
        const auto dtype = engine->getTensorDataType(name);
        const auto shape = engine->getTensorShape(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            ++input_count;
            // Prefer the binding the rest of the library addresses by name; fall
            // back to the first 4-D input for engines that name it otherwise.
            const bool is_named_input = std::string_view(name) == "input";
            if (shape.nbDims >= 4 && (is_named_input || engine_in_h < 0)) {
                engine_in_h = static_cast<int>(shape.d[shape.nbDims - 2]);
                engine_in_w = static_cast<int>(shape.d[shape.nbDims - 1]);
            }
        }
        if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            ++output_count;
            output_names.emplace_back(name);
        }

        std::printf("    [%d] %-12s  %s  %-5s  shape=", i, name, mode_name(mode),
                    dtype_name(dtype));
        print_shape(shape);
        std::putchar('\n');
    }

    if (argc == 3) {
        const std::filesystem::path meta_path{argv[2]};
        rfdetr::EngineMeta meta;
        try {
            meta = rfdetr::EngineMeta::from_json_file(meta_path);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "error reading meta: %s\n", e.what());
            return 1;
        }

        std::printf("\nmeta sidecar: %s\n", meta_path.c_str());
        std::printf("  variant      = %s\n", meta.variant.c_str());
        std::printf("  input_hxw    = %dx%d\n", meta.input_h, meta.input_w);
        std::printf("  num_queries  = %d\n", meta.num_queries);
        std::printf("  num_classes  = %d\n", meta.num_classes);
        std::printf("  color_order  = %s\n", meta.color_order.c_str());
        std::printf("  precision    = %s\n", meta.precision.c_str());
        std::printf("  mean         = [%.4f, %.4f, %.4f]\n", meta.mean[0], meta.mean[1],
                    meta.mean[2]);
        std::printf("  std          = [%.4f, %.4f, %.4f]\n", meta.std[0], meta.std[1], meta.std[2]);

        const std::string bg_str =
            meta.bg_class_index ? std::to_string(*meta.bg_class_index) : std::string("(absent)");
        std::printf("  bg_class_idx = %s\n", bg_str.c_str());
        if (meta.parent_class_index >= 0) {
            std::printf("  hierarchy    = parent %d, fine [", meta.parent_class_index);
            for (std::size_t i = 0; i < meta.fine_class_indices.size(); ++i) {
                std::printf("%s%d", i ? "," : "", meta.fine_class_indices[i]);
            }
            std::printf("], fine_thresh %.3f\n", meta.fine_conf_threshold);
        }
        if (meta.has_masks) {
            std::printf("  masks        = %dx%d\n", meta.mask_h, meta.mask_w);
        }
        if (meta.has_pin_types) {
            std::printf("  pin_types    = %d via output '%s'\n", meta.num_pin_types,
                        meta.pin_type_output_name.c_str());
        }
        if (meta.input_monochrome) {
            std::printf("  monochrome   = %s\n", *meta.input_monochrome ? "true" : "false");
        }

        // The check that matters: the sidecar's resolution against the ENGINE's,
        // not against a variant default. meta.input_h sizes preprocessing while
        // the buffer it fills is sized from the binding below, and a static-shape
        // engine never reaches setInputShape, so TensorRT reports nothing.
        if (auto why = rfdetr::describe_input_resolution_mismatch(meta, engine_in_h,
                                                                 engine_in_w)) {
            std::printf("  ERROR: %s\n", why->c_str());
            std::printf("         This engine and this sidecar do not belong together. "
                        "Rebuild from the matching ONNX.\n");
        } else if (engine_in_h > 0) {
            std::printf("  input matches engine binding: %dx%d  OK\n", engine_in_h,
                        engine_in_w);
        } else {
            std::printf("  NOTE: engine input dims are dynamic/unresolved — "
                        "cannot cross-check the sidecar resolution\n");
        }

        // Informational only. A fine-tuned checkpoint legitimately trains at a
        // resolution other than the variant's stock one, and the exporter follows
        // the CHECKPOINT. Warning here used to invert the whole check: a correct
        // 1200x1200 sidecar looked broken, a genuine engine/sidecar mismatch where
        // both happened to read 432 looked fine, and the obvious reaction --
        // editing the sidecar back to 432 -- created exactly the silent-corruption
        // case above.
        const auto& vmeta = rfdetr::variant_meta(meta.variant);
        if (vmeta.variant == rfdetr::Variant::Unknown) {
            std::printf("  WARNING: variant '%s' not in built-in variant table\n",
                        meta.variant.c_str());
        } else if (vmeta.input_size != meta.input_h || vmeta.input_size != meta.input_w) {
            std::printf("  note: variant '%s' default resolution is %dx%d; this model uses "
                        "%dx%d (normal for a fine-tuned checkpoint)\n",
                        meta.variant.c_str(), vmeta.input_size, vmeta.input_size,
                        meta.input_h, meta.input_w);
        }

        // RF-DETR always emits dets + labels; seg adds masks, a pin-type head adds
        // pin_types. Derive the expectation from what this model says it is rather
        // than from a constant 2, which warned on every segmentation engine ever
        // built and would warn on every pin-type one.
        const auto has_named = [&](const char* n) {
            return std::any_of(output_names.begin(), output_names.end(),
                               [n](const std::string& o) { return o == n; });
        };
        const bool engine_has_masks = has_named("masks");
        const char* pin_out = meta.pin_type_output_name.empty() ? "pin_types"
                                                                : meta.pin_type_output_name.c_str();
        const bool engine_has_pin_types = has_named(pin_out);

        const int expected_outputs = 2 + (meta.has_masks ? 1 : 0) + (meta.has_pin_types ? 1 : 0);
        if (output_count != expected_outputs) {
            std::printf("  WARNING: engine has %d outputs; sidecar implies %d "
                        "(dets, labels%s%s)\n",
                        output_count, expected_outputs,
                        meta.has_masks ? ", masks" : "",
                        meta.has_pin_types ? ", pin_types" : "");
        }
        if (meta.has_masks != engine_has_masks) {
            std::printf("  WARNING: sidecar says has_masks=%s but the engine %s a "
                        "'masks' output\n",
                        meta.has_masks ? "true" : "false",
                        engine_has_masks ? "HAS" : "has NO");
        }
        if (meta.has_pin_types != engine_has_pin_types) {
            std::printf("  WARNING: sidecar says has_pin_types=%s but the engine %s a "
                        "'%s' output\n",
                        meta.has_pin_types ? "true" : "false",
                        engine_has_pin_types ? "HAS" : "has NO", pin_out);
        }
        if (input_count != 1) {
            std::printf("  WARNING: engine has %d inputs, expected 1\n", input_count);
        }
    }

    return 0;
}
