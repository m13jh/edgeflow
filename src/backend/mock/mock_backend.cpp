#include "edgeflow/backend/mock_backend.h"
#include <charconv>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

namespace edgeflow {
namespace {

constexpr std::uint32_t kMaxGeneratedOutputs = 64;

struct MockOptions {
    std::uint32_t num_outputs = 0;
    bool fail_run = false;
};

/**
 * 校验 MockBackend 私有配置
 */
Status ParseOptions(const ModelConfig& config, MockOptions& out) {
    MockOptions options;
    for (const auto& [key, value] : config.backend_options) {
        if (key == "num_outputs") {
            std::uint32_t count = 0;
            const char* end = value.data() + value.size();
            const auto result = std::from_chars(value.data(), end, count);
            if (result.ec != std::errc{} || result.ptr != end ||
                count == 0 || count > kMaxGeneratedOutputs) {
                return Status::InvalidArgument("MockBackend: num_outputs must be an integer in [1, 64]");
            }
            options.num_outputs = count;
        } else if (key == "fail_run") {
            if (value == "true") {
                options.fail_run = true;
            } else if (value == "false") {
                options.fail_run = false;
            } else {
                return Status::InvalidArgument("MockBackend: fail_run must be true or false");
            }
        } else {
            return Status::InvalidArgument("MockBackend: unknown option: " + key);
        }
    }
    out = options;
    return Status::Ok();
}

Status ValidateDescs(const std::vector<TensorDesc>& descs, bool is_output) {
    if (descs.empty()) {
        return Status::InvalidArgument("MockBackend: empty tensor descriptors");
    }
    for (const auto& desc : descs) {
        if (desc.ByteSize() == 0) {
            return Status::InvalidArgument("MockBackend: invalid tensor: " + desc.name);
        }
        if (desc.fmt != TensorFormat::Undefined &&
            desc.fmt != TensorFormat::NHWC && desc.fmt != TensorFormat::NCHW) {
            return Status::Unsupported("MockBackend: unsupported format: " + desc.name);
        }
        if (is_output && desc.dtype != DataType::Float32) {
            return Status::Unsupported("MockBackend: output must be Float32: " + desc.name);
        }
    }
    return Status::Ok();
}

} // namespace

Status MockBackend::Load(const ModelConfig& config) {
    try {
        if (config.model_path.empty()) {
            return Status::InvalidArgument("MockBackend: empty model path");
        }
        if (!config.expected_sha256.empty()) {
            return Status::Unsupported("MockBackend: SHA-256 verification is not implemented");
        }
        MockOptions options;
        auto st = ParseOptions(config, options);
        if (!st.ok())
            return st;
        if (!spec_.forced_load_status.ok()) {
            return spec_.forced_load_status;
        }
        if (spec_.check_model_file_exists) {
            std::error_code ec;
            const bool exists = std::filesystem::exists(config.model_path, ec);
            if (ec) {
                return Status::IoError("MockBackend: cannot stat model: " + ec.message());
            }
            if (!exists) {
                return Status::NotFound("MockBackend: model not found: " + config.model_path);
            }
            const bool regular = std::filesystem::is_regular_file(config.model_path, ec);
            if (ec) {
                return Status::IoError("MockBackend: cannot inspect model: " + ec.message());
            }
            if (!regular) {
                return Status::InvalidArgument("MockBackend: model is not a regular file");
            }
        }
        auto new_inputs = spec_.input_descs;
        auto new_outputs = spec_.output_descs;
        if (options.num_outputs != 0) {
            new_outputs.clear();
            new_outputs.reserve(options.num_outputs);
            for (std::uint32_t i = 0; i < options.num_outputs; ++i) {
                new_outputs.push_back({ "output_" + std::to_string(i), { 1, 1001 }, DataType::Float32, TensorFormat::Undefined });
            }
        }
        st = ValidateDescs(new_inputs, false);
        if (!st.ok()) {
            return st;
        }
        st = ValidateDescs(new_outputs, true);
        if (!st.ok()) {
            return st;
        }
        input_descs_ = std::move(new_inputs);
        output_descs_ = std::move(new_outputs);
        fail_run_ = options.fail_run;
        run_count_ = 0;
        loaded_ = true;
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("MockBackend::Load: out of memory");
    } catch (const std::length_error&) {
        return Status::RuntimeError("MockBackend::Load: invalid allocation size");
    }
}

void MockBackend::Reset() noexcept {
    input_descs_.clear();
    output_descs_.clear();
    fail_run_ = false;
    run_count_ = 0;
    loaded_ = false;
}

BackendCapabilities MockBackend::Capabilities() const {
    BackendCapabilities caps;
    caps.backend_name = "mock";
    caps.preferred_memory_type = MemoryType::Host;
    caps.max_concurrency = 1;
    caps.max_batch_size = 1;
    caps.supports_multi_output = true;
    return caps;
}

Status MockBackend::Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    try {
        if (!loaded_) {
            return Status::RuntimeError("MockBackend: model is not loaded");
        }
        if (&inputs == &outputs) {
            return Status::InvalidArgument("MockBackend: inputs and outputs must not alias");
        }
        if (inputs.size() != input_descs_.size()) {
            return Status::InvalidArgument("MockBackend: input count mismatch");
        }
        std::uint32_t seed = 0;
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto& tensor = inputs[i];
            const auto st = tensor.Validate();
            if (!st.ok()) {
                return Status::InvalidArgument("MockBackend: invalid input at index " + std::to_string(i));
            }
            const auto& actual = tensor.Desc();
            const auto& expected = input_descs_[i];
            if (actual.dims != expected.dims || actual.dtype != expected.dtype ||
                actual.fmt != expected.fmt) {
                return Status::InvalidArgument("MockBackend: input descriptor mismatch at index " + std::to_string(i));
            }
            const auto* bytes = static_cast<const unsigned char*>(tensor.RawData());
            seed = (seed * 31u + bytes[0]) % 1000u;
        }
        if (!spec_.forced_run_status.ok()) {
            auto st = spec_.forced_run_status;
            outputs.clear();
            return st;
        }
        if (fail_run_) {
            auto st = Status::RuntimeError("MockBackend: injected run failure");
            outputs.clear();
            return st;
        }
        if (run_count_ == std::numeric_limits<std::size_t>::max()) {
            return Status::RuntimeError("MockBackend: run count overflow");
        }
        std::vector<Tensor> temporary;
        temporary.reserve(output_descs_.size());
        for (std::size_t i = 0; i < output_descs_.size(); ++i) {
            const auto& desc = output_descs_[i];
            Tensor output;
            auto st = Tensor::Allocate(desc, output);
            if (!st.ok())
                return st;
            auto* values = output.MutableData<float>();
            if (!values)
                return Status::RuntimeError("MockBackend: invalid output storage");
            const auto count = desc.NumElements();
            for (std::size_t j = 0; j < count; ++j) {
                const auto v = (seed + (i % 1000u) * 31u + (j % 1000u) * 17u) % 1000u;
                values[j] = static_cast<float>(v) / 1000.0f;
            }
            temporary.push_back(std::move(output));
        }
        outputs.swap(temporary);
        ++run_count_;
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("MockBackend::Run: out of memory");
    } catch (const std::length_error&) {
        return Status::RuntimeError("MockBackend::Run: invalid allocation size");
    }
}

} // namespace edgeflow
