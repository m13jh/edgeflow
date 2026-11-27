#include "edgeflow/backend/rknn_backend.h"
#include "edgeflow/backend/rknn_context.h"
#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/image_io.h"
#include "edgeflow/core/status.h"
#include "edgeflow/core/system_probe.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

constexpr const char* kNpuRenderNode = "/dev/dri/renderD129";
constexpr const char* kDogImagePath = "/usr/share/model/dog_224x224.jpg";

int g_failures = 0;

// 轻量断言宏：Release 模式下也不会被 NDEBUG 屏蔽
#define CHECK(expr)                                               \
    do {                                                          \
        if (!(expr)) {                                            \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ \
                      << "  CHECK(" #expr ") failed\n";           \
            ++g_failures;                                         \
        }                                                         \
    } while (0)

/**
 * RAII 临时文件管理类：用于构造空模型、损坏模型等错误注入用例，离开作用域自动清理
 */
class ScopedTempFile {
public:
    /**
     * 在系统临时目录下按标签与进程 PID 创建临时模型文件并写入字节内容
     */
    ScopedTempFile(const std::string& tag, const std::vector<std::uint8_t>& bytes) {
        const std::string filename = "edgeflow_rknn_test_" + tag + "_" + std::to_string(::getpid()) + ".rknn";
        path_ = (fs::temp_directory_path() / filename).string();
        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        if (!bytes.empty()) {
            out.write(
                reinterpret_cast<const char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size())
            );
        }
    }

    /**
     * 析构时自动移除临时测试模型文件并忽略删除异常
     */
    ~ScopedTempFile() noexcept {
        std::error_code ec;
        fs::remove(path_, ec);
    }

    ScopedTempFile(const ScopedTempFile&) = delete;
    ScopedTempFile& operator=(const ScopedTempFile&) = delete;

    /**
     * 获取临时测试模型文件的完整路径
     */
    const std::string& path() const noexcept {
        return path_;
    }

private:
    std::string path_;
};

/**
 * 判断当前运行环境是否具备可访问的 RK3576 NPU 设备与默认测试模型
 */
bool HasBoardNpuAndModel() {
    const bool npu_accessible = (::access(kNpuRenderNode, R_OK | W_OK) == 0);
    std::error_code ec;
    const bool model_exists =
        ((fs::is_regular_file(edgeflow::SystemProbe::kDefaultModelPath, ec)) &&
         (!ec) &&
         (::access(edgeflow::SystemProbe::kDefaultModelPath, R_OK) == 0));
    return npu_accessible && model_exists;
}

/**
 * 1. 测试空上下文的默认构造、重复 Reset、空移动以及未加载状态调用拦截
 */
void TestEmptyContextLifecycle() {
    edgeflow::RknnContext ctx;
    CHECK(!ctx.IsLoaded());
    CHECK(ctx.inputs().empty());
    CHECK(ctx.outputs().empty());

    // 空状态下重复调用 Reset() 应当完全幂等、安全
    ctx.Reset();
    ctx.Reset();
    CHECK(!ctx.IsLoaded());

    // 未加载模型时调用查询与推理必须返回 RuntimeError，不能把 0 句柄传给底层 C API
    edgeflow::RknnSdkVersion ver;
    const auto ver_status = ctx.QuerySdkVersion(ver);
    CHECK(!ver_status.ok());
    CHECK(ver_status.code() == edgeflow::ErrorCode::RuntimeError);

    const std::uint8_t dummy_byte = 0;
    std::vector<std::vector<float>> outputs;
    const auto run_status = ctx.RunRgbUint8(&dummy_byte, 1, outputs);
    CHECK(!run_status.ok());
    CHECK(run_status.code() == edgeflow::ErrorCode::RuntimeError);

    // 空对象的移动构造与移动赋值
    edgeflow::RknnContext moved_ctor(std::move(ctx));
    CHECK(!moved_ctor.IsLoaded());
    edgeflow::RknnContext moved_assign;
    moved_assign = std::move(moved_ctor);
    CHECK(!moved_assign.IsLoaded());
}

/**
 * 2. 测试 LoadFromFile 的文件系统校验与损坏模型错误注入
 */
void TestLoadFromFileErrorPaths() {
    edgeflow::RknnContext ctx;

    // 2.1 空路径 -> InvalidArgument
    const auto empty_path_status = ctx.LoadFromFile("");
    CHECK(!empty_path_status.ok());
    CHECK(empty_path_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(!ctx.IsLoaded());

    // 2.2 不存在的文件 -> NotFound
    const auto missing_status = ctx.LoadFromFile("/tmp/edgeflow_nonexistent_model_99999.rknn");
    CHECK(!missing_status.ok());
    CHECK(missing_status.code() == edgeflow::ErrorCode::NotFound);
    CHECK(!ctx.IsLoaded());

    // 2.3 传入目录而非普通文件 -> InvalidArgument
    const auto dir_status = ctx.LoadFromFile("/tmp");
    CHECK(!dir_status.ok());
    CHECK(dir_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(!ctx.IsLoaded());

    // 2.4 0 字节空文件 -> InvalidArgument
    ScopedTempFile zero_model("zero", {});
    const auto zero_status = ctx.LoadFromFile(zero_model.path());
    CHECK(!zero_status.ok());
    CHECK(zero_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(!ctx.IsLoaded());

    // 2.5 伪造魔数的损坏模型文件 -> rknn_init 拦截并返回 RuntimeError，句柄保持未加载
    const std::vector<std::uint8_t> corrupt_bytes = {
        'B', 'A', 'D', '_', 'R', 'K', 'N', 'N', '_', 'M', 'O', 'D', 'E', 'L'
    };
    ScopedTempFile corrupt_model("corrupt", corrupt_bytes);
    const auto corrupt_status = ctx.LoadFromFile(corrupt_model.path());
    CHECK(!corrupt_status.ok());
    CHECK(corrupt_status.code() == edgeflow::ErrorCode::RuntimeError);
    CHECK(!ctx.IsLoaded());
}

/**
 * 3. 测试板端真实模型加载、RAII 移动语义与强异常安全 (加载失败不破坏原已加载上下文)
 */
void TestBoardModelLifecycleAndMoveSemantics() {
    edgeflow::RknnContext ctx;
    const auto load_status = ctx.LoadFromFile(edgeflow::SystemProbe::kDefaultModelPath);
    CHECK(load_status.ok());
    if (!load_status.ok()) {
        return;
    }
    CHECK(ctx.IsLoaded());

    // 3.1 校验 SDK/驱动版本与 MobileNetV1 输入输出张量元数据
    edgeflow::RknnSdkVersion ver;
    const auto ver_status = ctx.QuerySdkVersion(ver);
    CHECK(ver_status.ok());
    CHECK(!ver.api_version.empty());
    CHECK(!ver.drv_version.empty());

    CHECK(ctx.inputs().size() == 1);
    CHECK(ctx.outputs().size() == 1);
    if (ctx.inputs().size() == 1 && ctx.outputs().size() == 1) {
        const auto& in = ctx.inputs()[0];
        const auto& out = ctx.outputs()[0];
        const std::vector<std::uint32_t> expected_dims = { 1, 224, 224, 3 };
        CHECK(in.dims == expected_dims);
        CHECK(in.n_elems == 224U * 224U * 3U);
        CHECK(in.format == "NHWC");
        CHECK(out.n_elems == 1001U);
    }

    // 3.2 移动构造：源对象必须被置空（防止析构时 double-free），新对象接管上下文
    edgeflow::RknnContext moved_ctor(std::move(ctx));
    CHECK(!ctx.IsLoaded());
    CHECK(ctx.inputs().empty());
    CHECK(ctx.outputs().empty());
    CHECK(moved_ctor.IsLoaded());
    CHECK(moved_ctor.inputs().size() == 1);
    CHECK(moved_ctor.outputs().size() == 1);

    // 3.3 移动赋值：新对象接管后依然能正常调用底层 rknn_query
    edgeflow::RknnContext moved_assign;
    moved_assign = std::move(moved_ctor);
    CHECK(!moved_ctor.IsLoaded());
    CHECK(moved_assign.IsLoaded());
    CHECK(moved_assign.QuerySdkVersion(ver).ok());

    // 3.4 强保证验证：在已加载的上下文上加载不存在的模型失败后，原有上下文不应被破坏
    const auto failed_reload =
        moved_assign.LoadFromFile("/tmp/edgeflow_reload_missing_99999.rknn");
    CHECK(!failed_reload.ok());
    CHECK(failed_reload.code() == edgeflow::ErrorCode::NotFound);
    CHECK(moved_assign.IsLoaded());
    CHECK(moved_assign.inputs().size() == 1);

    // 3.5 显式 Reset() 后彻底释放
    moved_assign.Reset();
    CHECK(!moved_assign.IsLoaded());
    CHECK(moved_assign.inputs().empty());
    CHECK(moved_assign.outputs().empty());
}

/**
 * 4. 测试 RunRgbUint8 参数边界拦截与 dog_224x224.jpg 实机推理精度对齐
 */
void TestInferenceValidationAndAccuracy() {
    edgeflow::RknnContext ctx;
    const auto load_status = ctx.LoadFromFile(edgeflow::SystemProbe::kDefaultModelPath);
    CHECK(load_status.ok());
    if (!load_status.ok() || ctx.inputs().empty()) {
        return;
    }

    const std::size_t expected_bytes = ctx.inputs()[0].n_elems;
    std::vector<std::vector<float>> outputs;

    // 4.1 空指针输入 -> InvalidArgument
    const auto null_status = ctx.RunRgbUint8(nullptr, expected_bytes, outputs);
    CHECK(!null_status.ok());
    CHECK(null_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 4.2 0 字节输入 -> InvalidArgument
    const std::uint8_t one_byte = 0;
    const auto zero_len_status = ctx.RunRgbUint8(&one_byte, 0, outputs);
    CHECK(!zero_len_status.ok());
    CHECK(zero_len_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 4.3 长度不匹配输入 -> InvalidArgument
    const std::vector<std::uint8_t> wrong_buf(100, 0);
    const auto wrong_len_status =
        ctx.RunRgbUint8(wrong_buf.data(), wrong_buf.size(), outputs);
    CHECK(!wrong_len_status.ok());
    CHECK(wrong_len_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 4.4 实机样本图推理并与 rknn_common_test 基线（Top-1 类别 156，分数 ~0.935）对齐
    std::error_code ec;
    if (!fs::is_regular_file(kDogImagePath, ec) || ec) {
        std::cout << "[SKIP] sample image not found: " << kDogImagePath << '\n';
        return;
    }

    edgeflow::RgbImage dog;
    const auto img_status = edgeflow::LoadJpegRgb24(kDogImagePath, dog);
    CHECK(img_status.ok());
    if (!img_status.ok()) {
        return;
    }
    CHECK(dog.data.size() == expected_bytes);

    const auto infer_status =
        ctx.RunRgbUint8(dog.data.data(), dog.data.size(), outputs);
    CHECK(infer_status.ok());
    if (!infer_status.ok()) {
        return;
    }

    CHECK(outputs.size() == 1);
    if (outputs.size() == 1) {
        const auto& scores = outputs[0];
        CHECK(scores.size() == 1001U);
        const bool all_finite = std::all_of(scores.begin(), scores.end(), [](float v) {
            return std::isfinite(v);
        });
        CHECK(all_finite);

        if (!scores.empty()) {
            const auto best_it = std::max_element(scores.begin(), scores.end());
            const auto best_index =
                static_cast<std::size_t>(std::distance(scores.begin(), best_it));
            // 对齐 rknn_common_test 基线：Top-1 为类别 156 (Blenheim spaniel)，置信度约 0.935
            CHECK(best_index == 156U);
            CHECK(*best_it > 0.90f);
        }
    }
}

/** 无需真实 NPU 或模型的 RknnBackend 参数校验。 */
void TestRknnBackendErrorPaths() {
    using namespace edgeflow;
    RknnBackend backend;
    CHECK(!backend.IsLoaded());
    CHECK(backend.InputDescs().empty());
    CHECK(backend.OutputDescs().empty());
    CHECK(backend.Capabilities().backend_name == "rknn");
    std::vector<Tensor> inputs, outputs;
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::RuntimeError);
    CHECK(backend.Run(inputs, inputs).code() == ErrorCode::InvalidArgument);
    ModelConfig config;
    CHECK(backend.Load(config).code() == ErrorCode::InvalidArgument);
    config.model_path = "/tmp/edgeflow_unloaded_backend_model.rknn";
    config.expected_sha256 = std::string(64, 'a');
    CHECK(backend.Load(config).code() == ErrorCode::Unsupported);
    config.expected_sha256.clear();
    config.backend_options["threads"] = "4";
    CHECK(backend.Load(config).code() == ErrorCode::InvalidArgument);
    config.backend_options.clear();
    config.backend_options["core_mask"] = "99";
    CHECK(backend.Load(config).code() == ErrorCode::InvalidArgument);
    CHECK(!backend.IsLoaded());
    CHECK(backend.InputDescs().empty());
    CHECK(backend.OutputDescs().empty());
}

/** 通过 IInferenceBackend 验证真实 MobileNetV1 推理及失败重载后的可用性。 */
void TestRknnBackendMobileNetInference() {
    using namespace edgeflow;
    RknnBackend backend;
    IInferenceBackend& iface = backend;
    ModelConfig config;
    config.model_path = SystemProbe::kDefaultModelPath;
    config.backend_options["core_mask"] = "auto";
    const auto load = iface.Load(config);
    CHECK(load.ok());
    if (!load.ok())
        return;
    CHECK(iface.IsLoaded());
    CHECK(iface.InputDescs().size() == 1);
    CHECK(iface.OutputDescs().size() == 1);
    if (iface.InputDescs().size() != 1 || iface.OutputDescs().size() != 1)
        return;
    const std::vector<std::uint32_t> expected_input{ 1, 224, 224, 3 };
    const std::vector<std::uint32_t> expected_output{ 1, 1001 };
    const auto input_desc = iface.InputDescs().front();
    const auto output_desc = iface.OutputDescs().front();
    CHECK(input_desc.dims == expected_input);
    CHECK(input_desc.dtype == DataType::UInt8);
    CHECK(input_desc.fmt == TensorFormat::NHWC);
    CHECK(output_desc.dims == expected_output);
    CHECK(output_desc.dtype == DataType::Float32);

    ModelConfig missing = config;
    missing.model_path = "/tmp/edgeflow_rknn_backend_reload_missing_99999.rknn";
    CHECK(iface.Load(missing).code() == ErrorCode::NotFound);
    CHECK(backend.IsLoaded());
    CHECK(iface.InputDescs().size() == 1);
    CHECK(iface.OutputDescs().size() == 1);
    if (!backend.IsLoaded() || iface.InputDescs().size() != 1 || iface.OutputDescs().size() != 1)
        return;
    CHECK(iface.InputDescs().front().dims == input_desc.dims);
    CHECK(iface.OutputDescs().front().dims == output_desc.dims);

    std::vector<Tensor> inputs, outputs;
    Tensor wrong;
    const auto bad_alloc = Tensor::Allocate(
        { "images", { 1, 100, 100, 3 }, DataType::UInt8, TensorFormat::NHWC }, wrong
    );
    CHECK(bad_alloc.ok());
    if (!bad_alloc.ok())
        return;
    inputs.push_back(std::move(wrong));
    CHECK(iface.Run(inputs, outputs).code() == ErrorCode::InvalidArgument);
    inputs.clear();

    std::error_code ec;
    if (!fs::is_regular_file(kDogImagePath, ec) || ec) {
        std::cout << "[SKIP] RknnBackend sample image not found: " << kDogImagePath << '\n';
        return;
    }
    RgbImage dog;
    const auto image_status = LoadJpegRgb24(kDogImagePath, dog);
    CHECK(image_status.ok());
    if (!image_status.ok())
        return;
    CHECK(dog.width == 224);
    CHECK(dog.height == 224);
    CHECK(dog.channels == 3);
    CHECK(dog.data.size() == 224U * 224U * 3U);
    if (dog.width != 224 || dog.height != 224 || dog.channels != 3 ||
        dog.data.size() != 224U * 224U * 3U)
        return;
    Tensor image;
    const auto copy = Tensor::FromBytes(
        { input_desc.name, { 1, 224, 224, 3 }, DataType::UInt8, TensorFormat::NHWC },
        dog.data.data(),
        dog.data.size(),
        image
    );
    CHECK(copy.ok());
    if (!copy.ok())
        return;
    inputs.push_back(std::move(image));
    const auto run = iface.Run(inputs, outputs);
    CHECK(run.ok());
    if (!run.ok())
        return;
    CHECK(outputs.size() == 1);
    if (outputs.size() != 1)
        return;
    const auto& output = outputs.front();
    CHECK(output.Validate().ok());
    CHECK(output.Desc().dims == expected_output);
    CHECK(output.Desc().dtype == DataType::Float32);
    CHECK(output.Desc().NumElements() == 1001);
    CHECK(output.SizeBytes() == 1001U * sizeof(float));
    const float* scores = output.Data<float>();
    CHECK(scores != nullptr);
    if (!scores || output.SizeBytes() != 1001U * sizeof(float))
        return;
    const bool finite = std::all_of(scores, scores + 1001, [](float v) {
        return std::isfinite(v);
    });
    CHECK(finite);
    if (!finite)
        return;
    const auto* best = std::max_element(scores, scores + 1001);
    CHECK(static_cast<std::size_t>(best - scores) == 156U);
    CHECK(*best > 0.90f);
}

} // namespace

/**
 * RKNN 后端测试入口：始终运行空状态与错误注入测试；具备板端 NPU 时运行真实模型与推理测试
 */
int main() {
    TestEmptyContextLifecycle();
    TestLoadFromFileErrorPaths();
    TestRknnBackendErrorPaths();

    if (HasBoardNpuAndModel()) {
        TestBoardModelLifecycleAndMoveSemantics();
        TestInferenceValidationAndAccuracy();
        TestRknnBackendMobileNetInference();
    } else {
        std::cout << "[SKIP] NPU device or default RKNN model unavailable; "
                     "skipped on-board hardware checks\n";
    }

    if (g_failures == 0) {
        std::cout << "[PASS] all edgeflow_rknn_tests passed\n";
        return 0;
    }
    std::cerr << "[FAIL] " << g_failures << " check(s) failed\n";
    return 1;
}
