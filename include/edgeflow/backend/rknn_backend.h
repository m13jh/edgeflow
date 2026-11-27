#pragma once

#include "edgeflow/backend/inference_backend.h"
#include "edgeflow/backend/rknn_context.h"

namespace edgeflow {

class RknnBackend final : public IInferenceBackend {
public:
    RknnBackend() = default;
    ~RknnBackend() override = default;
    RknnBackend(const RknnBackend&) = delete;
    RknnBackend& operator=(const RknnBackend&) = delete;
    Status Load(const ModelConfig& config) override;
    bool IsLoaded() const noexcept override;
    void Reset() noexcept override;
    BackendCapabilities Capabilities() const override;
    const std::vector<TensorDesc>& InputDescs() const noexcept override;
    const std::vector<TensorDesc>& OutputDescs() const noexcept override;
    Status Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;
    // 查询底层 RKNN SDK 版本
    Status QuerySdkVersion(RknnSdkVersion& out) const;
    // 保留原生模型属性，用于诊断与日志打印
    const RknnContext& raw_context() const noexcept {
        return ctx_;
    }

private:
    RknnContext ctx_;
    std::vector<TensorDesc> input_descs_;
    std::vector<TensorDesc> output_descs_;
};

} // namespace edgeflow