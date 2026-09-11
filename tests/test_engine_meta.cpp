// EngineMeta JSON round-trip and cross-check tests.
//
// The round trip matters because rfdetr_build is a LOSSY COPY by construction:
// it parses the exporter's <onnx>.meta.json into an EngineMeta and writes that
// struct back out as <engine>.engine.json, which is the only description of the
// engine the ROS node ever reads. Any field the struct does not model is erased
// in transit, silently, and the erasure is invisible in every other test because
// nothing else reads the field either. Five fields were being dropped exactly
// this way (bg_class_index, has_pin_types, pin_type_output_name, pin_type_names,
// num_pin_types) -- "write every field, read it back, compare every field" is
// the only shape of test that catches that class of bug.
//
// No test framework on purpose: this has to stay buildable and runnable on a
// machine with no TensorRT (see tests/CMakeLists.txt).

#include "rfdetr/core/engine_meta.hpp"

#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

namespace {

int g_failures = 0;

void report(bool ok, const char* expr, const char* file, int line) {
    if (ok) return;
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
}

#define CHECK(expr) report(static_cast<bool>(expr), #expr, __FILE__, __LINE__)

std::filesystem::path unique_temp_path(const char* stem) {
    static int counter = 0;
    return std::filesystem::temp_directory_path() /
           ("rfdetr_test_" + std::string(stem) + "_" + std::to_string(counter++) + ".json");
}

// Unique per-test path under the system temp dir; removed by TempFile's dtor.
class TempFile {
   public:
    explicit TempFile(const char* stem) : path_(unique_temp_path(stem)) {}
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    const std::filesystem::path& path() const noexcept { return path_; }

    void write(const std::string& contents) const {
        std::ofstream out(path_);
        out << contents;
    }

   private:
    std::filesystem::path path_;
};

// Every field set to something distinguishable from its default, so a dropped
// field shows up as a mismatch rather than coincidentally matching.
rfdetr::EngineMeta fully_populated() {
    rfdetr::EngineMeta m;
    m.schema_version = 4;
    m.variant        = "seg-medium";
    m.input_h        = 1200;
    m.input_w        = 1200;
    m.num_queries    = 200;
    m.num_classes    = 3;
    m.mean           = {0.1f, 0.2f, 0.3f};
    m.std            = {0.4f, 0.5f, 0.6f};
    m.color_order    = "BGR";
    m.precision      = "int8";
    m.patch_size     = 12;
    m.dynamic_batch  = true;
    m.min_batch      = 1;
    m.opt_batch      = 2;
    m.max_batch      = 4;
    m.cuda_graph_compat = true;
    m.has_masks      = true;
    m.mask_h         = 300;
    m.mask_w         = 300;
    m.parent_class_index  = 2;
    m.fine_class_indices  = {0, 1};
    m.fine_conf_threshold = 0.25f;
    m.class_names         = {"pin_up", "pin_down", "pin_any"};
    m.bg_class_index      = 3;
    m.has_pin_types       = true;
    m.pin_type_output_name = "pin_types";
    m.pin_type_names = {"pin_standard", "pin_lugged", "pin_fixed", "pin_standard_slide",
                        "pin_lugged_slide"};
    m.num_pin_types    = 5;
    m.input_monochrome = true;
    m.luma_weights     = {0.25f, 0.5f, 0.25f};
    return m;
}

void expect_equal(const rfdetr::EngineMeta& a, const rfdetr::EngineMeta& b) {
    CHECK(a.schema_version == b.schema_version);
    CHECK(a.variant == b.variant);
    CHECK(a.input_h == b.input_h);
    CHECK(a.input_w == b.input_w);
    CHECK(a.num_queries == b.num_queries);
    CHECK(a.num_classes == b.num_classes);
    CHECK(a.mean == b.mean);
    CHECK(a.std == b.std);
    CHECK(a.color_order == b.color_order);
    CHECK(a.precision == b.precision);
    CHECK(a.patch_size == b.patch_size);
    CHECK(a.dynamic_batch == b.dynamic_batch);
    CHECK(a.min_batch == b.min_batch);
    CHECK(a.opt_batch == b.opt_batch);
    CHECK(a.max_batch == b.max_batch);
    CHECK(a.cuda_graph_compat == b.cuda_graph_compat);
    CHECK(a.has_masks == b.has_masks);
    CHECK(a.mask_h == b.mask_h);
    CHECK(a.mask_w == b.mask_w);
    CHECK(a.parent_class_index == b.parent_class_index);
    CHECK(a.fine_class_indices == b.fine_class_indices);
    CHECK(a.fine_conf_threshold == b.fine_conf_threshold);
    CHECK(a.class_names == b.class_names);
    CHECK(a.bg_class_index == b.bg_class_index);
    CHECK(a.has_pin_types == b.has_pin_types);
    CHECK(a.pin_type_output_name == b.pin_type_output_name);
    CHECK(a.pin_type_names == b.pin_type_names);
    CHECK(a.num_pin_types == b.num_pin_types);
    CHECK(a.input_monochrome == b.input_monochrome);
    CHECK(a.luma_weights == b.luma_weights);
}

void test_roundtrip_preserves_every_field() {
    const TempFile f("roundtrip");
    const rfdetr::EngineMeta original = fully_populated();
    original.to_json_file(f.path());
    expect_equal(original, rfdetr::EngineMeta::from_json_file(f.path()));
}

// rfdetr_build's actual behaviour: read the exporter's sidecar, mutate the few
// build-time fields, write it back. Two hops, because a field that survives one
// serialisation but is normalised on the next would still corrupt a rebuild.
void test_double_roundtrip_is_stable() {
    const TempFile f1("stable1");
    const TempFile f2("stable2");
    const rfdetr::EngineMeta original = fully_populated();
    original.to_json_file(f1.path());
    const auto once = rfdetr::EngineMeta::from_json_file(f1.path());
    once.to_json_file(f2.path());
    expect_equal(once, rfdetr::EngineMeta::from_json_file(f2.path()));
}

// "Absent" must survive as absent. Inventing a bg_class_index would make the
// tasks' cross-check compare a derived value against itself; inventing
// input_monochrome=false claims a legacy engine was colour-trained.
void test_absent_optionals_stay_absent() {
    const TempFile f("absent");
    rfdetr::EngineMeta m = fully_populated();
    m.bg_class_index.reset();
    m.input_monochrome.reset();
    m.to_json_file(f.path());

    const auto back = rfdetr::EngineMeta::from_json_file(f.path());
    CHECK(!back.bg_class_index.has_value());
    CHECK(!back.input_monochrome.has_value());

    std::ifstream in(f.path());
    const std::string text{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    CHECK(text.find("bg_class_index") == std::string::npos);
    CHECK(text.find("input_monochrome") == std::string::npos);
}

// The exporter writes an explicit JSON `null` -- not an absent key -- for
// pin_type_output_name and luma_weights when they do not apply. nlohmann's
// j.value(key, default) THROWS type_error.302 on a null instead of returning
// the default, so reading these with value() would make every sidecar from a
// model without a pin-type head (i.e. the one in production) fail to load.
void test_explicit_nulls_do_not_throw() {
    const TempFile f("nulls");
    f.write(R"({
      "schema_version": 4,
      "variant": "seg-medium",
      "input_h": 1200,
      "input_w": 1200,
      "num_queries": 200,
      "num_classes": 3,
      "bg_class_index": 3,
      "has_pin_types": false,
      "pin_type_output_name": null,
      "pin_type_names": [],
      "num_pin_types": 0,
      "input_monochrome": true,
      "luma_weights": null
    })");

    bool threw = false;
    rfdetr::EngineMeta m;
    try {
        m = rfdetr::EngineMeta::from_json_file(f.path());
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(!threw);
    CHECK(m.has_pin_types == false);
    CHECK(m.pin_type_output_name.empty());
    CHECK(m.num_pin_types == 0);
    CHECK(m.bg_class_index.has_value() && *m.bg_class_index == 3);
    // luma_weights: null must leave the BT.601 default standing, not zero it.
    CHECK(m.luma_weights[0] == 0.299f);
}

// A pre-hierarchy sidecar must still decode flat, and must not acquire fields
// it never claimed.
void test_legacy_v1_sidecar() {
    const TempFile f("v1");
    f.write(R"({
      "schema_version": 1,
      "variant": "nano",
      "input_h": 384,
      "input_w": 384,
      "num_queries": 300,
      "num_classes": 90
    })");

    const auto m = rfdetr::EngineMeta::from_json_file(f.path());
    CHECK(m.schema_version == 1);
    CHECK(m.parent_class_index == -1);   // flat decode
    CHECK(m.fine_class_indices.empty());
    CHECK(!m.bg_class_index.has_value());
    CHECK(!m.input_monochrome.has_value());
    CHECK(!m.has_pin_types);
    CHECK(m.color_order == "RGB");
}

void test_input_resolution_mismatch() {
    rfdetr::EngineMeta m;
    m.input_h = 1200;
    m.input_w = 1200;

    CHECK(!rfdetr::describe_input_resolution_mismatch(m, 1200, 1200).has_value());

    // Unresolved (dynamic) engine dims: nothing to compare, must not fire.
    CHECK(!rfdetr::describe_input_resolution_mismatch(m, -1, -1).has_value());
    CHECK(!rfdetr::describe_input_resolution_mismatch(m, 0, 0).has_value());

    // The case that publishes detections computed from uninitialised device
    // memory: sidecar smaller than the engine.
    const auto under = rfdetr::describe_input_resolution_mismatch(m, 432, 432);
    CHECK(under.has_value());
    // Both numbers must appear, or the message cannot be acted on.
    CHECK(under && under->find("1200") != std::string::npos);
    CHECK(under && under->find("432") != std::string::npos);

    // The out-of-bounds device write case: sidecar larger than the engine.
    rfdetr::EngineMeta big;
    big.input_h = 1200;
    big.input_w = 1200;
    CHECK(rfdetr::describe_input_resolution_mismatch(big, 432, 432).has_value());

    // Non-square disagreement on one axis only.
    CHECK(rfdetr::describe_input_resolution_mismatch(m, 1200, 800).has_value());
}

}  // namespace

int main() {
    test_roundtrip_preserves_every_field();
    test_double_roundtrip_is_stable();
    test_absent_optionals_stay_absent();
    test_explicit_nulls_do_not_throw();
    test_legacy_v1_sidecar();
    test_input_resolution_mismatch();

    if (g_failures != 0) {
        std::fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("all engine_meta checks passed\n");
    return 0;
}
