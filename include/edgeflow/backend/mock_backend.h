#pragma once

#include "edgeflow/backend/inference_backend.h"
#include <cstddef>
#include <vector>

namespace edgeflow {

struct MockBackendSpec {
    // 输入张量描述
    std::vector<TensorDesc> input_descs{
        { "input", { 1, 224, 224, 3 }, DataType::UInt8, TensorFormat::NHWC }
    };
    // 输出张量描述
    std::vector<TensorDesc> output_descs{
        { "output", { 1, 1001 }, DataType::Float32, TensorFormat::Undefined }
    };
    bool check_model_file_exists = false;     // 是否检查模型文件存在
    Status forced_load_status = Status::Ok(); // 强制的加载状态
    Status forced_run_status = Status::Ok();
};

class MockBackend final : public IInferenceBackend {
public:
    explicit MockBackend(MockBackendSpec spec = {})
        : spec_(std::move(spec)) {}
    ~MockBackend() override = default;

    Status Load(const ModelConfig& config) override;
    void Reset() noexcept override;
    BackendCapabilities Capabilities() const override;

    bool IsLoaded() const noexcept override {
        return loaded_;
    }

    const std::vector<TensorDesc>& InputDescs() const noexcept override {
        return input_descs_;
    }

    const std::vector<TensorDesc>& OutputDescs() const noexcept override {
        return output_descs_;
    }

    Status Run(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs) override;

    /**
     * 成功运行次数
     */
    std::size_t run_count() const noexcept {
        return run_count_;
    }

private:
    MockBackendSpec spec_;
    bool loaded_ = false;
    bool fail_run_ = false;
    std::size_t run_count_ = 0;
    std::vector<TensorDesc> input_descs_;
    std::vector<TensorDesc> output_descs_;
};

} // namespace edgeflow
