#pragma once

#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/status.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace edgeflow {

struct RknnSdkVersion {
    std::string api_version; // api 版本
    std::string drv_version; // npu 驱动版本
};

// 模型 Tensor 的信息
struct RknnTensorInfo {
    std::uint32_t index = 0;         // Tensor 的编号
    std::string name;                // Tensor 的名称
    std::vector<std::uint32_t> dims; // Tensor 的维度
    std::uint32_t n_elems = 0;       // Tensor 中的元素个数
    std::uint32_t size = 0;          // Tensor 字节数, size = n_elems * sizeof(type)
    std::string format;              // Tensor 数据布局, 如 NHWC, NCHW
    std::string type;                // 元素数据类型
    // Xfloat = (Xint - zp) * scale
    // 量化相关属性
    std::string qnt_type; // 量化类型, 如 qnt_type=AFFINE
    std::int32_t zp = 0;  // AFFINE 零点
    float scale = 0.0f;   // AFFINE 缩放因子
    // Xfloat = Xint * 2^-fl
    std::int8_t fl = 0; // DFP 量化的分数位长
    // 内存布局相关属性
    std::uint32_t w_stride = 0;         // 宽度方向的 stride
    std::uint32_t size_with_stride = 0; // 考虑 stride 后的字节数
};

class RknnContext final { // final 表示这个类不能被继承
public:
    RknnContext() noexcept = default;
    ~RknnContext() noexcept;
    RknnContext(const RknnContext&) = delete;
    RknnContext& operator=(const RknnContext&) = delete;
    RknnContext(RknnContext&& other) noexcept;
    RknnContext& operator=(RknnContext&& other) noexcept;
    Status LoadFromFile(const std::string& model_path);
    Status QuerySdkVersion(RknnSdkVersion& out) const;
    Status RunRgbUint8(const std::uint8_t* input_data, std::size_t input_size, std::vector<std::vector<float>>& outputs);
    void Reset() noexcept;
    // 通用推理接口，支持多输入和多输出
    Status RunTensors(const std::vector<Tensor>& inputs, std::vector<Tensor>& outputs);
    // 设置 NPU 核心运行模式
    Status SetCoreMask(std::string_view core_mask);

    /**
     * 判断当前上下文是否已成功加载 RKNN 模型
     */
    bool IsLoaded() const noexcept {
        return handle_ != 0;
    }

    /**
     * 获取已加载模型的所有输入张量属性列表
     */
    const std::vector<RknnTensorInfo>& inputs() const noexcept {
        return input_attrs_;
    }

    /**
     * 获取已加载模型的所有输出张量属性列表
     */
    const std::vector<RknnTensorInfo>& outputs() const noexcept {
        return output_attrs_;
    }

private:
    std::uint64_t handle_ = 0;                 // RKNN Runtime 模型上下文句柄
    std::vector<RknnTensorInfo> input_attrs_;  // 模型所有输入 Tensor 的信息
    std::vector<RknnTensorInfo> output_attrs_; // 模型所有输出 Tensor 的信息
};

} // namespace edgeflow