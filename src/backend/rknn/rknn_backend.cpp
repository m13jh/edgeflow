#include "edgeflow/backend/rknn_backend.h"
#include "edgeflow/backend/inference_backend.h"
#include "edgeflow/backend/rknn_context.h"
#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/status.h"
#include <limits>
#include <stdexcept>
#include <utility>

namespace edgeflow {
namespace {

/**
 * 字符串转为对应枚举, 因为 RknnTensorInfo 中的 format 是 string
 * 而 TensorDesc 中的 fmt 是枚举类型
 */
TensorFormat MapFormat(std::string_view format) noexcept {
    if (format == "NCHW") {
        return TensorFormat::NCHW;
    }
    if (format == "NHWC") {
        return TensorFormat::NHWC;
    }
    if (format == "NC1HWC2") {
        return TensorFormat::NC1HWC2;
    }
    return TensorFormat::Undefined;
}

/**
 * 校验 RknnBackend 私有配置
 */
Status ParseBackendOptions(const ModelConfig& config, std::string_view& core_mask) {
    core_mask = {};
    // 遍历私有配置
    for (const auto& [key, value] : config.backend_options) {
        if (key != "core_mask") {
            return Status::InvalidArgument("Unknown RKNN backend option: " + key);
        }
        if (value == "auto") {
            core_mask = "auto";
        } else if (value == "0") {
            core_mask = "core0";
        } else if (value == "1") {
            core_mask = "core1";
        } else if (value == "0_1") {
            core_mask = "core0_1";
        } else {
            return Status::InvalidArgument("Invalid RKNN core_mask: " + value);
        }
    }
    return Status::Ok();
}

/**
 * 构建 RknnBackend 对外张量描述
 */
Status BuildTensorDescs(const std::vector<RknnTensorInfo>& attrs, bool is_input, std::vector<TensorDesc>& out) {
    if (attrs.empty()) {
        return Status::RuntimeError("RKNN tensor attributes are empty");
    }
    std::vector<TensorDesc> temporary;
    temporary.reserve(attrs.size());
    for (const auto& attr : attrs) {
        TensorDesc desc;
        desc.name = attr.name;
        desc.dims = attr.dims;
        desc.fmt = MapFormat(attr.format);
        // 当前 P1 不支持分块布局
        if (desc.fmt == TensorFormat::NC1HWC2) {
            return Status::Unsupported("Unsupported RKNN tensor format: " + attr.format);
        }
        // 不允许未知格式被静默转换为 Undefined
        if (desc.fmt == TensorFormat::Undefined && attr.format != "UNDEFINED") {
            return Status::Unsupported("Unknown RKNN tensor format: " + attr.format);
        }
        if (is_input) {
            // 与当前 RunTensors 支持的输入布局保持一致
            if (desc.fmt != TensorFormat::NHWC && desc.fmt != TensorFormat::NCHW) {
                return Status::Unsupported("Unsupported RKNN input format: " + attr.format);
            }
            // 输入描述采用推荐的应用层数据类型
            // 这里我没有直接把所有模型的输入类型都设置成 UInt8,
            // 因为后面可能部署深度估计, SSC, 视觉几何融合等模型,
            // 其中部分输入本身就是浮点数据
            if (attr.type == "INT8" || attr.type == "UINT8") {
                desc.dtype = DataType::UInt8;
            } else if (attr.type == "FLOAT16" || attr.type == "FLOAT32") {
                desc.dtype = DataType::Float32;
            } else {
                return Status::Unsupported("Unsupported RKNN native input dtype: " + attr.type);
            }
        } else {
            // 底层 want_float = 1
            // 输出始终按照实际返回的 Float32 描述
            desc.dtype = DataType::Float32;
        }
        // 检查形状与元素数量
        if (desc.NumElements() == 0 || desc.NumElements() != attr.n_elems) {
            return Status::RuntimeError("Invalid RKNN tensor shape: " + attr.name);
        }
        // 检查字节数及 RKNN SDK 的 uint32_t 大小限制
        const auto bytes = desc.ByteSize();
        if (bytes == 0 || bytes > std::numeric_limits<std::uint32_t>::max()) {
            return Status::RuntimeError("Invalid RKNN tensor byte size: " + attr.name);
        }
        temporary.push_back(std::move(desc));
    }
    out = std::move(temporary);
    return Status::Ok();
}

} // namespace

/**
 * 加载模型
 */
Status RknnBackend::Load(const ModelConfig& config) {
    try {
        // 1. 检查模型路径
        if (config.model_path.empty()) {
            return Status::InvalidArgument("RKNN model path must not be empty");
        }
        // Day 9 实现 SHA-256 校验
        // 当前不能静默忽略用户明确要求的完整性校验
        if (!config.expected_sha256.empty()) {
            return Status::Unsupported("Model SHA-256 verification is not implemented yet");
        }
        // 2. 校验后端私有配置
        std::string_view core_mask;
        Status status = ParseBackendOptions(config, core_mask);
        if (!status.ok()) {
            return status;
        }
        // 3. 创建临时 Context
        RknnContext candidate;
        status = candidate.LoadFromFile(config.model_path);
        if (!status.ok()) {
            return status;
        }
        // 4. 设置 NPU Core Mask
        if (!core_mask.empty()) {
            status = candidate.SetCoreMask(core_mask);
            if (!status.ok()) {
                return status;
            }
        }
        // 5. 构造临时输入输出描述
        std::vector<TensorDesc> new_input_descs;
        std::vector<TensorDesc> new_output_descs;
        status = BuildTensorDescs(candidate.inputs(), true, new_input_descs);
        if (!status.ok()) {
            return status;
        }
        status = BuildTensorDescs(candidate.outputs(), false, new_output_descs);
        if (!status.ok()) {
            return status;
        }
        // 6. 全部成功后, 提交新模型和张量描述
        ctx_ = std::move(candidate);
        input_descs_ = std::move(new_input_descs);
        output_descs_ = std::move(new_output_descs);
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("RknnBackend::Load: out of memory");
    } catch (const std::length_error& e) {
        return Status::RuntimeError("RknnBackend::Load: allocation failed: " + std::string(e.what()));
    }
}

/**
 * 判断是否加载模型
 */
bool RknnBackend::IsLoaded() const noexcept {
    return ctx_.IsLoaded();
}

/**
 * 清除配置
 */
void RknnBackend::Reset() noexcept {
    ctx_.Reset();
    input_descs_.clear();
    output_descs_.clear();
}

BackendCapabilities RknnBackend::Capabilities() const {
    BackendCapabilities caps;
    caps.backend_name = "rknn";
    caps.preferred_memory_type = MemoryType::Host;
    caps.max_batch_size = 1;
    caps.max_concurrency = 1;
    caps.supports_multi_output = true;
    return caps;
}

const std::vector<TensorDesc>& RknnBackend::InputDescs() const noexcept {
    return input_descs_;
}

const std::vector<TensorDesc>& RknnBackend::OutputDescs() const noexcept {
    return output_descs_;
}

Status RknnBackend::QuerySdkVersion(RknnSdkVersion& out) const {
    return ctx_.QuerySdkVersion(out);
}

Status RknnBackend::Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) {
    if (&inputs == &outputs) {
        return Status::InvalidArgument("RknnBackend: inputs and outputs must not alias");
    }
    return ctx_.RunTensors(inputs, outputs);
}

} // namespace edgeflow