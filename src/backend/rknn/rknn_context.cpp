#include "edgeflow/backend/rknn_context.h"
#include "edgeflow/core/status.h"
#include "rknn_api.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace edgeflow {
namespace {
namespace fs = std::filesystem;

static_assert(sizeof(rknn_context) <= sizeof(std::uint64_t), "rknn_context cannot fit into RknnContext::handle_");
static_assert(std::is_integral<rknn_context>::value, "rknn_context must be an integer handle");

/**
 * 将错误类型转为字符串
 */
const char* RknnErrorName(int ret) noexcept {
    switch (ret) {
    case RKNN_SUCC:
        return "RKNN_SUCC";
    case RKNN_ERR_FAIL:
        return "RKNN_ERR_FAIL";
    case RKNN_ERR_TIMEOUT:
        return "RKNN_ERR_TIMEOUT";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
        return "RKNN_ERR_DEVICE_UNAVAILABLE";
    case RKNN_ERR_MALLOC_FAIL:
        return "RKNN_ERR_MALLOC_FAIL";
    case RKNN_ERR_PARAM_INVALID:
        return "RKNN_ERR_PARAM_INVALID";
    case RKNN_ERR_MODEL_INVALID:
        return "RKNN_ERR_MODEL_INVALID";
    case RKNN_ERR_CTX_INVALID:
        return "RKNN_ERR_CTX_INVALID";
    case RKNN_ERR_INPUT_INVALID:
        return "RKNN_ERR_INPUT_INVALID";
    case RKNN_ERR_OUTPUT_INVALID:
        return "RKNN_ERR_OUTPUT_INVALID";
    case RKNN_ERR_DEVICE_UNMATCH:
        return "RKNN_ERR_DEVICE_UNMATCH";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
        return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
        return "RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
        return "RKNN_ERR_TARGET_PLATFORM_UNMATCH";
#ifdef RKNN_ERR_KEY_INVALID
    case RKNN_ERR_KEY_INVALID:
        return "RKNN_ERR_KEY_INVALID";
#endif
#ifdef RKNN_ERR_DECRYPT_FAILED
    case RKNN_ERR_DECRYPT_FAILED:
        return "RKNN_ERR_DECRYPT_FAILED";
#endif
#ifdef RKNN_ERR_SOC_BIND_FAILED
    case RKNN_ERR_SOC_BIND_FAILED:
        return "RKNN_ERR_SOC_BIND_FAILED";
#endif
    default:
        return "RKNN_ERR_UNKNOWN";
    }
}

/**
 * 将 RKNN SDK 返回码映射为带接口名与错误详情的 Status
 */
Status MapRknnError(int ret, const std::string& api_name) {
    if (ret == RKNN_SUCC) {
        return Status::Ok();
    }
    const std::string detail = api_name + " failed " + RknnErrorName(ret) +
                               " (ret=" + std::to_string(ret) + ")";
    switch (ret) {
    case RKNN_ERR_PARAM_INVALID:
    case RKNN_ERR_INPUT_INVALID:
    case RKNN_ERR_OUTPUT_INVALID:
        return Status::InvalidArgument(detail);
    default:
        return Status::RuntimeError(detail);
    }
}

/**
 * 将 Tensor 类型转为字符串
 */
std::string TensorFormatName(rknn_tensor_format format) {
    switch (format) {
    case RKNN_TENSOR_NCHW:
        return "NCHW";
    case RKNN_TENSOR_NHWC:
        return "NHWC";
    case RKNN_TENSOR_NC1HWC2:
        return "NC1HWC2";
    case RKNN_TENSOR_UNDEFINED:
        return "UNDEFINED";
    default:
        return "UNKNOWN(" + std::to_string(static_cast<int>(format)) + ")";
    }
}

/**
 * 模板推导数组长度, 同时解决 C 风格数组没有 '\0' 的问题
 */
template <std::size_t N>
std::string BoundedCString(const char (&bytes)[N]) {
    const char* end = std::find(bytes, bytes + N, '\0');
    return std::string(bytes, end);
}

/**
 * 将 Tensor 类型转为字符串
 */
std::string TensorTypeName(rknn_tensor_type type) {
    switch (type) {
    case RKNN_TENSOR_FLOAT32:
        return "FLOAT32";
    case RKNN_TENSOR_FLOAT16:
        return "FLOAT16";
    case RKNN_TENSOR_INT8:
        return "INT8";
    case RKNN_TENSOR_INT4:
        return "INT4";
    case RKNN_TENSOR_UINT8:
        return "UINT8";
    case RKNN_TENSOR_INT16:
        return "INT16";
    case RKNN_TENSOR_UINT16:
        return "UINT16";
    case RKNN_TENSOR_INT32:
        return "INT32";
    case RKNN_TENSOR_UINT32:
        return "UINT32";
    case RKNN_TENSOR_INT64:
        return "INT64";
    case RKNN_TENSOR_BOOL:
        return "BOOL";
    default:
        return "UNKNOWN(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

/**
 * 量化类型转为字符串
 */
std::string QuantizationName(rknn_tensor_qnt_type type) {
    switch (type) {
    case RKNN_TENSOR_QNT_NONE:
        return "NONE";
    case RKNN_TENSOR_QNT_DFP:
        return "DFP";
    case RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC:
        return "AFFINE_ASYMMETRIC";
    default:
        return "UNKNOWN(" + std::to_string(static_cast<int>(type)) + ")";
    }
}

/**
 * 将 rknn_tensor_attr 转为自定义数据类型 RknnTensorInfo
 */
RknnTensorInfo ConvertTensorAttr(const rknn_tensor_attr& attr) {
    RknnTensorInfo result;
    result.index = attr.index;
    result.name = BoundedCString(attr.name);
    result.dims.assign(attr.dims, attr.dims + attr.n_dims);
    result.n_elems = attr.n_elems;
    result.size = attr.size;
    result.format = TensorFormatName(attr.fmt);
    result.type = TensorTypeName(attr.type);
    result.qnt_type = QuantizationName(attr.qnt_type);
    result.zp = attr.zp;
    result.scale = attr.scale;
    return result;
}

/**
 * 根据 Tensor 数量, 逐个调用 RKNN SDK, 获取每个 Tensor 的具体信息,
 * 并转换成 EdgeFlow 自己的 RknnTensorInfo
 */
Status QueryAttrs(
    rknn_context ctx,
    std::uint32_t count,
    rknn_query_cmd command,
    const char* command_name,
    std::vector<RknnTensorInfo>& result
) {
    result.clear();
    result.reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        rknn_tensor_attr attr{};
        attr.index = i;
        const int ret = rknn_query(ctx, command, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) {
            const std::string api_name = std::string("rknn_query(") + command_name +
                                         ", index=" + std::to_string(i) + ")";
            return MapRknnError(ret, api_name);
        }
        if (attr.n_dims > RKNN_MAX_DIMS) {
            return Status::RuntimeError(
                std::string("rknn_query(") + command_name +
                ", index=" + std::to_string(i) +
                "): invalid n_dims=" + std::to_string(attr.n_dims)
            );
        }
        result.push_back(ConvertTensorAttr(attr));
    }
    return Status::Ok();
}

/**
 * 先查询模型有多少个输入, 多少个输出, 然后分别调用 QueryAttrs()
 */
Status QueryAllTensorAttrs(
    rknn_context ctx,
    std::vector<RknnTensorInfo>& inputs,
    std::vector<RknnTensorInfo>& outputs
) {
    rknn_input_output_num counts{};
    const int ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts));
    if (ret != RKNN_SUCC) {
        return MapRknnError(ret, "rknn_query(RKNN_QUERY_IN_OUT_NUM)");
    }
    constexpr std::uint32_t kMaxTensorCount = 4096;
    if (
        counts.n_input == 0 ||
        counts.n_output == 0 ||
        counts.n_input > kMaxTensorCount ||
        counts.n_output > kMaxTensorCount
    ) {
        return Status::RuntimeError(
            "rknn_query(RKNN_QUERY_IN_OUT_NUM): invalid counts: n_input=" +
            std::to_string(counts.n_input) +
            ", n_output=" + std::to_string(counts.n_output)
        );
    }
    Status status = QueryAttrs(ctx, counts.n_input, RKNN_QUERY_INPUT_ATTR, "RKNN_QUERY_INPUT_ATTR", inputs);
    if (!status.ok()) {
        return status;
    }
    return QueryAttrs(ctx, counts.n_output, RKNN_QUERY_OUTPUT_ATTR, "RKNN_QUERY_OUTPUT_ATTR", outputs);
}

/**
 * RAII 守卫, 确保 rknn_outputs_get 分配的输出缓冲区一定通过 rknn_outputs_release 释放
 */
class OutputReleaseGuard {
public:
    /**
     * 构造输出释放守卫, armed 为 true 时析构或显式调用 Release 会释放输出缓冲区
     */
    OutputReleaseGuard(rknn_context ctx, std::uint32_t count, rknn_output* outputs, bool armed) noexcept
        : ctx_(ctx), count_(count), outputs_(outputs), armed_(armed) {}

    /**
     * 析构时若尚未释放则自动释放 RKNN 输出缓冲区
     */
    ~OutputReleaseGuard() noexcept {
        if (armed_) {
            (void)Release();
        }
    }
    OutputReleaseGuard(const OutputReleaseGuard&) = delete;
    OutputReleaseGuard& operator=(const OutputReleaseGuard&) = delete;

    /**
     * 显式释放 RKNN 输出缓冲区并解除守卫, 返回 rknn_outputs_release 的错误码
     */
    int Release() noexcept {
        if (!armed_) {
            return RKNN_SUCC;
        }
        armed_ = false;
        return rknn_outputs_release(ctx_, count_, outputs_);
    }

private:
    rknn_context ctx_;     // RKNN 模型上下文句柄
    std::uint32_t count_;  // 保存本次获取的输出 Tensor 数量
    rknn_output* outputs_; // 指向保存输出信息的数组
    bool armed_ = true;    // 标记是否还需要执行释放操作
};

} // namespace

/**
 * 析构时释放持有的 RKNN 上下文资源
 */
RknnContext::~RknnContext() noexcept {
    Reset();
}

/**
 * 移动构造, 接管源对象的 RKNN 句柄与张量属性并置空源对象
 */
RknnContext::RknnContext(RknnContext&& other) noexcept
    : handle_(std::exchange(other.handle_, 0)),
      input_attrs_(std::move(other.input_attrs_)),
      output_attrs_(std::move(other.output_attrs_)) {
    other.input_attrs_.clear();
    other.output_attrs_.clear();
}

/**
 * 移动赋值, 释放当前资源后接管源对象的 RKNN 句柄与张量属性
 */
RknnContext& RknnContext::operator=(RknnContext&& other) noexcept {
    if (this != &other) {
        Reset();
        handle_ = std::exchange(other.handle_, 0);
        input_attrs_ = std::move(other.input_attrs_);
        output_attrs_ = std::move(other.output_attrs_);
        other.input_attrs_.clear();
        other.output_attrs_.clear();
    }
    return *this;
}

/**
 * 销毁当前 RKNN 上下文句柄并清空输入输出属性缓存
 */
void RknnContext::Reset() noexcept {
    if (handle_ != 0) {
        (void)rknn_destroy(static_cast<rknn_context>(handle_));
        handle_ = 0;
    }
    input_attrs_.clear();
    output_attrs_.clear();
}

/**
 * 校验并加载指定路径的 RKNN 模型, 初始化上下文并缓存输入输出张量属性
 */
Status RknnContext::LoadFromFile(const std::string& model_path) {
    if (model_path.empty()) {
        return Status::InvalidArgument("model path must not be empty");
    }
    try {
        const fs::path path(model_path);
        std::error_code ec;
        // 1. 判断是否存在
        const bool exists = fs::exists(path, ec);
        if (ec) {
            return Status::IoError("cannot stat model '" + model_path + "': " + ec.message());
        }
        if (!exists) {
            return Status::NotFound("model file does not exist: " + model_path);
        }
        // 2. 判断是否为普通文件
        const bool regular = fs::is_regular_file(path, ec);
        if (ec) {
            return Status::IoError("cannot inspect model '" + model_path + "': " + ec.message());
        }
        if (!regular) {
            return Status::InvalidArgument("model is not a regular file: " + model_path);
        }
        // 3. 获取模型大小
        const std::uintmax_t file_size = fs::file_size(path, ec);
        if (ec) {
            return Status::IoError("cannot read model size '" + model_path + "': " + ec.message());
        }
        if (file_size == 0) {
            return Status::InvalidArgument("model file is empty: " + model_path);
        }
        if (file_size > std::numeric_limits<std::uint32_t>::max() ||
            file_size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
            return Status::InvalidArgument("model file exceeds rknn_init size limit: " + model_path);
        }
        // 4. 二进制方式读取模型
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            return Status::IoError("cannot open model file: " + model_path);
        }
        std::vector<std::uint8_t> model_bytes(static_cast<std::size_t>(file_size));
        const auto read_size = static_cast<std::streamsize>(file_size);
        in.read(reinterpret_cast<char*>(model_bytes.data()), read_size);
        if (in.gcount() != read_size) {
            return Status::IoError("short read while loading model: " + model_path);
        }
        // 5. 初始化模型
        rknn_context new_ctx = 0;
        const int ret = rknn_init(&new_ctx, model_bytes.data(), static_cast<std::uint32_t>(model_bytes.size()), 0, nullptr);
        if (ret != RKNN_SUCC) {
            if (new_ctx != 0) {
                (void)rknn_destroy(new_ctx);
            }
            return MapRknnError(ret, "rknn_init");
        }
        if (new_ctx == 0) {
            return Status::RuntimeError("rknn_init returned RKNN_SUCC but context is zero");
        }
        // 6. 转换为自己的 RknnContext 类型管理
        RknnContext candidate;
        candidate.handle_ = static_cast<std::uint64_t>(new_ctx);
        Status status = QueryAllTensorAttrs(new_ctx, candidate.input_attrs_, candidate.output_attrs_);
        if (!status.ok()) {
            return status;
        }
        *this = std::move(candidate);
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("out of memory while loading RKNN model: " + model_path);
    } catch (const std::length_error& e) {
        return Status::RuntimeError("invalid allocation while loading RKNN model: " + std::string(e.what()));
    }
}

/**
 * 查询已加载 RKNN 上下文的 SDK API 版本与驱动版本
 */
Status RknnContext::QuerySdkVersion(RknnSdkVersion& out) const {
    if (!IsLoaded()) {
        return Status::RuntimeError("QuerySdkVersion: context is not loaded");
    }

    rknn_sdk_version ver{};
    const int ret = rknn_query(static_cast<rknn_context>(handle_), RKNN_QUERY_SDK_VERSION, &ver, sizeof(ver));
    if (ret != RKNN_SUCC) {
        return MapRknnError(ret, "rknn_query(RKNN_QUERY_SDK_VERSION)");
    }

    RknnSdkVersion temporary;
    temporary.api_version = BoundedCString(ver.api_version);
    temporary.drv_version = BoundedCString(ver.drv_version);
    out = std::move(temporary);
    return Status::Ok();
}

/**
 * 使用单路 NHWC 格式的 RGB uint8 输入执行一次 RKNN 推理, 并以 float 数组返回所有输出张量
 */
Status RknnContext::RunRgbUint8(
    const std::uint8_t* input_data,
    std::size_t input_size,
    std::vector<std::vector<float>>& outputs
) {
    // 1. 判断模型是否加载
    if (!IsLoaded()) {
        return Status::RuntimeError("RunRgbUint8: RKNN model is not loaded");
    }
    // 2. 检查数据
    if (input_attrs_.size() != 1) {
        return Status::InvalidArgument(
            "RunRgbUint8 requires exactly one input tensor; got " +
            std::to_string(input_attrs_.size())
        );
    }
    if (input_data == nullptr) {
        return Status::InvalidArgument("RunRgbUint8: input_data is null");
    }
    // TODO: 当前同时校验 input_size == size 与 input_size == n_elems, 会拦截 FP16/FP32 等非 1 字节输入的模型
    //       (pass_through=0 + RKNN_TENSOR_UINT8 时输入缓冲区期望大小应按 n_elems 校验);
    //       且下方 input_size > uint32_t::max() 位于与 uint32_t 成员比较之后属于死代码, 后续需调整顺序或精简。
    if (input_size != input_attrs_[0].size) {
        return Status::InvalidArgument(
            "RunRgbUint8: input size mismatch: got " +
            std::to_string(input_size) + ", expected " +
            std::to_string(input_attrs_[0].size)
        );
    }
    if (input_size == 0 || input_size != input_attrs_[0].n_elems) {
        return Status::InvalidArgument("RunRgbUint8 requires a tightly packed 1-byte-per-element input");
    }
    if (input_size > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidArgument("RunRgbUint8: input exceeds RKNN size limit");
    }
    if (output_attrs_.empty() || output_attrs_.size() > std::numeric_limits<std::uint32_t>::max()) {
        return Status::RuntimeError("RunRgbUint8: invalid output tensor count");
    }
    const auto ctx = static_cast<rknn_context>(handle_);
    try {
        // 3. 配置输入数据结构体
        rknn_input input{};
        input.index = 0;
        input.buf = const_cast<std::uint8_t*>(input_data);
        input.size = static_cast<std::uint32_t>(input_size);
        input.pass_through = 0; // 不使用 pass_through 方式
        input.type = RKNN_TENSOR_UINT8;
        input.fmt = RKNN_TENSOR_NHWC;
        int ret = rknn_inputs_set(ctx, 1, &input);
        if (ret != RKNN_SUCC) {
            return MapRknnError(ret, "rknn_inputs_set");
        }
        // 4. 执行 rknn_run
        ret = rknn_run(ctx, nullptr);
        if (ret != RKNN_SUCC) {
            return MapRknnError(ret, "rknn_run");
        }
        // 5. 配置输出结构体
        std::vector<rknn_output> raw_outputs(output_attrs_.size());
        for (std::size_t i = 0; i < raw_outputs.size(); ++i) {
            raw_outputs[i].index = static_cast<std::uint32_t>(i);
            raw_outputs[i].want_float = 1;  // 输出为 float 类型
            raw_outputs[i].is_prealloc = 0; // 自动申请内存
        }
        const auto count = static_cast<std::uint32_t>(raw_outputs.size());
        // 6. 获得输出数据结构体
        ret = rknn_outputs_get(ctx, count, raw_outputs.data(), nullptr);
        const bool any_buffer = std::any_of(raw_outputs.begin(), raw_outputs.end(), [](const rknn_output& item) {
            return item.buf != nullptr;
        });
        // RAII 管理资源, rknn_outputs_get 要配合 rknn_outputs_release 使用
        // ret == RKNN_SUCC || any_buffer, rknn_outputs_get 成功或者部分成功
        OutputReleaseGuard guard(ctx, count, raw_outputs.data(), ret == RKNN_SUCC || any_buffer);
        if (ret != RKNN_SUCC) {
            return MapRknnError(ret, "rknn_outputs_get");
        }
        // 7. 提取输出数据
        std::vector<std::vector<float>> new_outputs;
        new_outputs.reserve(raw_outputs.size());
        for (std::size_t i = 0; i < raw_outputs.size(); ++i) {
            const auto& attr = output_attrs_[i];
            const auto& raw = raw_outputs[i];
            // 不为 0, 不能超出 uint32_t 表示范围
            if (
                attr.n_elems == 0 ||
                attr.n_elems > std::numeric_limits<std::uint32_t>::max() / sizeof(float)
            ) {
                return Status::RuntimeError("RunRgbUint8: invalid element count for output " + std::to_string(i));
            }
            const std::size_t expected_bytes = static_cast<std::size_t>(attr.n_elems) * sizeof(float);
            if (raw.buf == nullptr || raw.size != expected_bytes) {
                return Status::RuntimeError(
                    "RunRgbUint8: invalid float output " + std::to_string(i) +
                    ": expected " + std::to_string(expected_bytes) +
                    " bytes, got " + std::to_string(raw.size) +
                    (raw.buf == nullptr ? " (null buffer)" : "")
                );
            }
            std::vector<float> values(attr.n_elems);
            std::memcpy(values.data(), raw.buf, expected_bytes);
            new_outputs.push_back(std::move(values));
        }
        // 这里不是重复释放, 而是检查是否释放成功
        const int release_ret = guard.Release();
        if (release_ret != RKNN_SUCC) {
            return MapRknnError(release_ret, "rknn_outputs_release");
        }
        outputs = std::move(new_outputs);
        return Status::Ok();
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("RunRgbUint8: out of memory");
    } catch (const std::length_error& e) {
        return Status::RuntimeError("RunRgbUint8: allocation failed: " + std::string(e.what()));
    }
}

} // namespace edgeflow