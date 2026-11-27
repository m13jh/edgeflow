#pragma once

#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/status.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace edgeflow {

/**
 * 模型加载配置
 */
struct ModelConfig {
    std::string model_path;                                       // 模型路径
    std::string expected_sha256;                                  // 模型文件预期的 SHA-256
    std::unordered_map<std::string, std::string> backend_options; // 后端私有配置参数
};

/**
 * 推理后端能力描述
 */
struct BackendCapabilities {
    std::string backend_name;                            // 后端名称, RKNN, TENSORRT
    MemoryType preferred_memory_type = MemoryType::Host; // 首选内存类型
    std::uint32_t max_batch_size = 1;                    // 最大批处理大小
    std::uint32_t max_concurrency = 1;                   // 最大并发数
    bool supports_multi_output = true;                   // 是否支持多输出
};

/**
 * 通用推理后端抽象接口
 */
class IInferenceBackend {
public:
    virtual ~IInferenceBackend() = default;
    IInferenceBackend(const IInferenceBackend&) = delete;
    IInferenceBackend& operator=(const IInferenceBackend&) = delete;
    // 加载模型并初始化推理资源
    virtual Status Load(const ModelConfig& config) = 0;
    // 是否加载成功
    virtual bool IsLoaded() const noexcept = 0;
    // 卸载模型并释放推理资源
    virtual void Reset() noexcept = 0;
    // 获取当前后端的能力描述
    virtual BackendCapabilities Capabilities() const = 0;
    // 获取模型输入张量描述
    virtual const std::vector<TensorDesc>& InputDescs() const noexcept = 0;
    // 获取模型输出张量描述
    virtual const std::vector<TensorDesc>& OutputDescs() const noexcept = 0;
    // 执行模型推理
    virtual Status Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) = 0;

protected:
    IInferenceBackend() = default;
};

} // namespace edgeflow