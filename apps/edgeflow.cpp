#include "edgeflow/backend/inference_backend.h"
#include "edgeflow/core/latency_stats.h"
#include "edgeflow/postprocess/classification.h"
#include <charconv>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <vector>
#if EDGEFLOW_WITH_RKNN

#include "edgeflow/backend/rknn_backend.h"
#include "edgeflow/backend/rknn_context.h"

#endif

#include "edgeflow/core/image_io.h"
#include "edgeflow/core/system_probe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ratio>
#include <string>
#include <string_view>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitHealthFailed = 1;
constexpr int kExitUsage = 2;
constexpr std::string_view kNpuNode = "/dev/dri/renderD129";

struct InferOptions {
    std::string model_path;
    std::string input_path;
    std::size_t warmup = 0;
    std::size_t runs = 1;
    std::size_t topk = 5;
};

/**
 * 将字符串严格解析为无符号整数, 包含非数字字符、负号或溢出时返回 false
 */
bool ParseSize(std::string_view text, std::size_t& out) {
    if (text.empty()) {
        return false;
    }
    std::size_t value = 0;
    const char* end = text.data() + text.size();
    const auto result = std::from_chars(text.data(), end, value);
    if (result.ec != std::errc{} || result.ptr != end) {
        return false;
    }
    out = value;
    return true;
}

/**
 * 打印 edgeflow 命令行帮助信息与退出码说明
 */
void PrintUsage(std::ostream& os) {
    os << "EdgeFlow - edge AI runtime diagnostic CLI\n\n"
       << "Usage:\n"
       << "  edgeflow version\n"
       << "  edgeflow doctor [--model <path>]\n"
       << "  edgeflow infer [--model <path>] --input <jpeg>\n"
       << "                 [--warmup <N>] [--runs <N>] [--topk <K>]\n"
       << "  edgeflow --help\n\n"
       << "Commands:\n"
       << "  version  Show version, commit, build type, architecture, backend flags\n"
       << "  doctor   Print a JSON device report to stdout\n"
       << "  infer    Run image inference and report latency\n\n"
       << "Infer options:\n"
       << "  --model <path>  RKNN model (default: built-in model path)\n"
       << "  --input <jpeg>  Input JPEG image (required)\n"
       << "  --warmup <N>    Warmup iterations, N >= 0 (default: 0)\n"
       << "  --runs <N>      Timed iterations, N >= 1 (default: 1)\n"
       << "  --topk <K>      Number of predictions for single output, K >= 1 (default: 5)\n\n"
       << "Exit codes:\n"
       << "  0  Success (doctor: required checks passed)\n"
       << "  1  Probe or inference failure\n"
       << "  2  Invalid command-line arguments\n";
}

/**
 * 打印版本号、Git 提交、构建类型、目标架构及后端开关
 */
void PrintVersion() {
    std::cout << "EdgeFlow " << EDGEFLOW_VERSION << '\n'
              << "Git commit: " << EDGEFLOW_GIT_COMMIT << '\n'
              << "Build type: " << EDGEFLOW_BUILD_TYPE << '\n'
              << "Target architecture: " << EDGEFLOW_TARGET_ARCH << '\n'
              << "RKNN backend: " << (EDGEFLOW_WITH_RKNN ? "ON" : "OFF") << '\n'
              << "RGA backend: " << (EDGEFLOW_WITH_RGA ? "ON" : "OFF") << '\n';
}

/**
 * 采集系统诊断报告并以 JSON 输出, 同时校验 NPU 节点与模型文件可用性
 */
int RunDoctor(const std::string& model_path) {
    edgeflow::SystemReport report;
    const edgeflow::Status status = edgeflow::SystemProbe::Collect(report, model_path);
    if (!status.ok()) {
        std::cerr << "edgeflow doctor: probe failed: " << status.message() << '\n';
        return kExitHealthFailed;
    }
    std::cout << report.to_json();
    const auto npu = std::find_if(
        report.devices.begin(), report.devices.end(), [](const edgeflow::DeviceAccess& device) {
            return device.path == kNpuNode;
        }
    );
    bool healthy = true;
    if (npu == report.devices.end() || !npu->exists || !npu->readable || !npu->writable) {
        std::cerr << "edgeflow doctor: expected NPU node " << kNpuNode
                  << " is missing or not readable/writable\n";
        healthy = false;
    }
    if (!report.model.exists || !report.model.readable) {
        std::cerr << "edgeflow doctor: model is missing or unreadable: "
                  << report.model.path << '\n';
        healthy = false;
    }
    return healthy ? kExitOk : kExitHealthFailed;
}

#if EDGEFLOW_WITH_RKNN

/**
 * 格式化打印单个 RKNN 输入或输出张量的维度、数据类型、量化参数与内存对齐属性
 */
void PrintTensor(const char* kind, const edgeflow::RknnTensorInfo& tensor) {
    std::cout << kind << " index=" << tensor.index << " name=" << tensor.name
              << " dims=[";
    for (std::size_t i = 0; i < tensor.dims.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << tensor.dims[i];
    }
    std::cout << "]"
              << ", n_elems=" << tensor.n_elems
              << ", size=" << tensor.size
              << ", fmt=" << tensor.format
              << ", type=" << tensor.type
              << ", qnt_type=" << tensor.qnt_type
              << ", zp=" << tensor.zp
              << ", scale=" << tensor.scale
              << ", fl=" << static_cast<int>(tensor.fl) // fl 为 int8_t 直接输出可能会当成 signed char
              << ", w_stride=" << tensor.w_stride
              << ", size_with_stride=" << tensor.size_with_stride
              << '\n';
}

/**
 * 校验解码后的 RGB 图像尺寸与模型单输入张量的批次、通道数及宽高是否匹配
 */
bool CheckImageShape(const edgeflow::TensorDesc& input, const edgeflow::RgbImage& image) {
    const auto& dims = input.dims;
    const bool nhwc = (input.fmt == edgeflow::TensorFormat::NHWC);
    if ((!nhwc && input.fmt != edgeflow::TensorFormat::NCHW) || dims.size() != 4 ||
        dims[0] != 1 || dims[nhwc ? 3 : 1] != 3) {
        std::cerr << "edgeflow infer: expected a single RGB image tensor "
                     "with batch=1 and 3 channels (NHWC or NCHW)\n";
        return false;
    }
    const auto height = dims[nhwc ? 1 : 2];
    const auto width = dims[nhwc ? 2 : 3];
    if (image.width != width || image.height != height) {
        std::cerr << "edgeflow infer: JPEG is " << image.width << 'x' << image.height
                  << ", model expects " << width << 'x' << height
                  << " (width x height); automatic resizing is disabled\n";
        return false;
    }
    return true;
}

/**
 * 打印应用层实际返回的 Float32 输出张量摘要 (调用前已校验)
 */
void PrintOutputSummary(std::size_t index, const edgeflow::Tensor& tensor) {
    const auto& desc = tensor.Desc();
    const auto count = desc.NumElements();
    const auto* values = tensor.Data<float>();
    std::cout << "output_tensor index=" << index
              << " name=" << desc.name << " dims=[";
    for (std::size_t i = 0; i < desc.dims.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << desc.dims[i];
    }
    std::cout << "]" << ", n_elems=" << count
              << ", dtype=" << edgeflow::DataTypeName(desc.dtype)
              << ", fmt=" << edgeflow::TensorFormatName(desc.fmt)
              << ", first_values=[";
    const auto preview_count = std::min(count, std::size_t{ 5 });
    for (std::size_t j = 0; j < preview_count; ++j) {
        if (j != 0) {
            std::cout << ", ";
        }
        std::cout << values[j];
    }
    if (count > preview_count) {
        std::cout << ", ...";
    }
    std::cout << "]\n";
}

/**
 * 加载 JPEG 图像与 RKNN 模型, 执行预热与多轮计时推理并输出延迟统计与结果
 */
int InferImage(const InferOptions& options) {
    const std::string& input_path = options.input_path;
    const std::string& model_path = options.model_path;
    // 1. 加载图片
    edgeflow::RgbImage image;
    auto status = edgeflow::LoadJpegRgb24(input_path, image);
    if (!status.ok()) {
        std::cerr << "edgeflow infer: reading " << input_path << ": " << status.message() << '\n';
        return kExitHealthFailed;
    }
    // 2. 加载模型
    edgeflow::RknnBackend backend;
    edgeflow::ModelConfig config;
    config.model_path = model_path;
    const auto load_start = std::chrono::steady_clock::now(); // ns
    status = backend.Load(config);
    const auto load_time = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start
    );
    if (!status.ok()) {
        std::cerr << "edgeflow infer: loading " << model_path << ": " << status.message() << '\n';
        return kExitHealthFailed;
    }
    // 3. 查询 rknn sdk 版本
    edgeflow::RknnSdkVersion version;
    status = backend.QuerySdkVersion(version);
    if (!status.ok()) {
        std::cerr << "edgeflow infer: querying SDK version: " << status.message() << '\n';
        return kExitHealthFailed;
    }
    std::cout << "Model load: " << load_time.count() << " ms\n"
              << "RKNN SDK: " << version.api_version
              << "\nDriver: " << version.drv_version << '\n';
    // 4. 输出 Tensor 数据
    for (const auto& tensor : backend.raw_context().inputs()) {
        PrintTensor("input", tensor);
    }
    for (const auto& tensor : backend.raw_context().outputs()) {
        PrintTensor("output", tensor);
    }
    const auto& input_descs = backend.InputDescs();
    if (input_descs.size() != 1) {
        std::cerr << "edgeflow infer: expected 1 input tensor, got " << input_descs.size() << '\n';
        return kExitHealthFailed;
    }
    if (!CheckImageShape(input_descs.front(), image)) {
        return kExitHealthFailed;
    }
    // 5. 准备计时样本和输出容器
    // 在计时循环外将 JPEG 字节封装为通用 Tensor，并改用 Tensor 列表作为输入/输出容器
    // JPEG 解码结果实际为紧凑 NHWC UInt8, 不使用模型原生量化类型
    std::vector<edgeflow::Tensor> inputs;
    std::vector<edgeflow::Tensor> output_tensors;
    std::vector<double> samples_ms;
    try {
        // 构造 NHWC UInt8 输入描述, 并通过 Tensor::FromBytes 拷贝图像像素
        edgeflow::TensorDesc input_desc{
            input_descs.front().name,
            { 1, static_cast<std::uint32_t>(image.height), static_cast<std::uint32_t>(image.width), 3 },
            edgeflow::DataType::UInt8,
            edgeflow::TensorFormat::NHWC
        };
        inputs.emplace_back();
        status = edgeflow::Tensor::FromBytes(input_desc, image.data.data(), image.data.size(), inputs.back());
        if (!status.ok()) {
            std::cerr << "edgeflow infer: creating input tensor: " << status.message() << '\n';
            return kExitHealthFailed;
        }
        samples_ms.reserve(options.runs);
    } catch (const std::length_error&) {
        std::cerr << "edgeflow infer: invalid input or latency allocation size\n";
        return kExitHealthFailed;
    } catch (const std::bad_alloc&) {
        std::cerr << "edgeflow infer: cannot allocate input or latency samples\n";
        return kExitHealthFailed;
    }
    // 6. 预热, 不采集耗时样本
    for (std::size_t i = 0; i < options.warmup; ++i) {
        output_tensors.clear();
        status = backend.Run(inputs, output_tensors);
        if (!status.ok()) {
            std::cerr << "edgeflow infer: warmup " << i + 1 << " failed: " << status.message() << '\n';
            return kExitHealthFailed;
        }
    }
    // 7. 正式计时, 每次调用单独测量
    for (std::size_t i = 0; i < options.runs; ++i) {
        output_tensors.clear();
        const auto start = std::chrono::steady_clock::now();
        status = backend.Run(inputs, output_tensors);
        const auto end = std::chrono::steady_clock::now();
        if (!status.ok()) {
            std::cerr << "edgeflow infer: inference failed: " << status.message() << '\n';
            return kExitHealthFailed;
        }
        samples_ms.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    // 8. 统计正式推理耗时
    edgeflow::LatencyStats stats;
    status = edgeflow::ComputeLatencyStats(samples_ms, stats);
    if (!status.ok()) {
        std::cerr << "edgeflow infer: latency statistics failed: " << status.message() << '\n';
        return kExitHealthFailed;
    }
    // 9. 对最后一次正式推理结果进行分类后处理
    if (output_tensors.empty()) {
        std::cerr << "edgeflow infer: model returned no outputs\n";
        return kExitHealthFailed;
    }
    // 逐个校验输出张量是否为非空有效的 Float32 数据
    for (std::size_t i = 0; i < output_tensors.size(); ++i) {
        if (output_tensors[i].Data<float>() == nullptr || output_tensors[i].Desc().NumElements() == 0) {
            std::cerr << "edgeflow infer: invalid Float32 output at index " << i << '\n';
            return kExitHealthFailed;
        }
    }
    std::vector<edgeflow::ClassPrediction> predictions;
    // 单输出分支: 从 Tensor 拷贝出 std::vector<float>
    // 兼容原有 TopKPredictions 接口, 并捕获内存异常
    if (output_tensors.size() == 1) {
        const auto& tensor = output_tensors.front();
        const auto* values = tensor.Data<float>();
        const auto count = tensor.Desc().NumElements();
        try {
            // 兼容原有 vector<float> 分类接口, 此拷贝不计入推理耗时。
            const std::vector<float> scores(values, values + count);
            status = edgeflow::TopKPredictions(scores, options.topk, predictions);
            if (!status.ok()) {
                std::cerr << "edgeflow infer: classification postprocess failed: "
                          << status.message() << '\n';
                return kExitHealthFailed;
            }
        } catch (const std::length_error&) {
            std::cerr << "edgeflow infer: classification output is too large\n";
            return kExitHealthFailed;
        } catch (const std::bad_alloc&) {
            std::cerr << "edgeflow infer: cannot allocate classification results\n";
            return kExitHealthFailed;
        }
    }
    // 10. 打印耗时统计, 默认单次推理保持原有输出格式
    if (options.runs == 1) {
        if (options.warmup != 0) {
            std::cout << "Warmup: " << options.warmup << '\n';
        }
        std::cout << "RunRgbUint8: " << stats.mean_ms << " ms\n";
    } else {
        std::cout << "---- RunRgbUint8 latency ----\n"
                  << "Runs: " << stats.count << '\n'
                  << "Warmup: " << options.warmup << '\n'
                  << std::fixed << std::setprecision(3)
                  << "Min: " << stats.min_ms << " ms\n"
                  << "Mean: " << stats.mean_ms << " ms\n"
                  << "P50: " << stats.p50_ms << " ms\n"
                  << "P95: " << stats.p95_ms << " ms\n"
                  << "P99: " << stats.p99_ms << " ms\n"
                  << "Max: " << stats.max_ms << " ms\n"
                  << "FPS: " << stats.throughput_fps << '\n';
    }
    // 11. 打印分类结果
    if (output_tensors.size() == 1) {
        std::cout << "---- Top" << predictions.size() << " ----\n"
                  << std::fixed << std::setprecision(6);
        for (const auto& prediction : predictions) {
            std::cout << prediction.score
                      << " - "
                      << prediction.class_id
                      << '\n';
        }
    } else {
        // 多输出张量摘要打印分支
        std::cout << "---- Output tensors (" << output_tensors.size() << ") ----\n"
                  << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < output_tensors.size(); ++i) {
            PrintOutputSummary(i, output_tensors[i]);
        }
    }
    return kExitOk;
}

#endif

/**
 * 解析 infer 子命令参数并执行单张图像推理
 * 1. edgeflow infer --model mobilenet_v1.rknn --input dog.jpg
 * 2. edgeflow infer --help
 */
int RunInfer(int argc, char* argv[]) {
    if (argc == 3 && std::string_view(argv[2]) == "--help") {
        PrintUsage(std::cout);
        return kExitOk;
    }
    InferOptions options;
    std::optional<std::size_t> warmup;
    std::optional<std::size_t> runs;
    std::optional<std::size_t> topk;
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag == "--model" || flag == "--input") {
            auto& value = (flag == "--model" ? options.model_path : options.input_path);
            if (!value.empty()) {
                std::cerr << "edgeflow infer: duplicate option: " << flag << '\n';
                return kExitUsage;
            }
            if (i + 1 == argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
                std::cerr << "edgeflow infer: " << flag << " requires a path\n";
                return kExitUsage;
            }
            value = argv[++i];
        } else if (flag == "--warmup" || flag == "--runs" || flag == "--topk") {
            auto& value = (flag == "--warmup" ? warmup : (flag == "--runs" ? runs : topk));
            if (value.has_value()) {
                std::cerr << "edgeflow infer: duplicate option: " << flag << '\n';
                return kExitUsage;
            }
            // 后面的 ParseSize 带有严格的字符校验
            if (i + 1 == argc) {
                std::cerr << "edgeflow infer: " << flag << " requires an integer\n";
                return kExitUsage;
            }
            std::size_t parsed = 0;
            // 如果当前选项不是 --warmup, 但解析出来的数字等于 0
            if (!ParseSize(argv[++i], parsed) || (flag != "--warmup" && parsed == 0)) {
                std::cerr << "edgeflow infer: " << flag
                          << (flag == "--warmup" ? " requires an integer >= 0\n"
                                                 : " requires an integer >= 1\n");
                return kExitUsage;
            }
            value = parsed;
        } else {
            std::cerr << "edgeflow infer: unknown argument: " << flag << '\n';
            return kExitUsage;
        }
    }
    if (options.input_path.empty()) {
        std::cerr << "edgeflow infer: --input <jpeg> is required\n";
        return kExitUsage;
    }
    options.warmup = warmup.value_or(options.warmup);
    options.runs = runs.value_or(options.runs);
    options.topk = topk.value_or(options.topk);
#if EDGEFLOW_WITH_RKNN
    if (options.model_path.empty()) {
        options.model_path = edgeflow::SystemProbe::kDefaultModelPath;
    }
    return InferImage(options);
#else
    std::cerr << "edgeflow infer: RKNN backend is disabled (EDGEFLOW_WITH_RKNN=OFF)\n";
    return kExitHealthFailed;
#endif
}

} // namespace

/**
 * CLI 入口, 解析命令行参数并分发到 help / version / doctor / infer 子命令
 */
int main(int argc, char* argv[]) {
    if (argc == 1) {
        PrintUsage(std::cout);
        return kExitOk;
    }
    const std::string_view command(argv[1]);
    if (command == "--help" || command == "-h" || command == "help") {
        if (argc != 2) {
            std::cerr << "edgeflow: --help takes no arguments\n";
            PrintUsage(std::cerr);
            return kExitUsage;
        }
        PrintUsage(std::cout);
        return kExitOk;
    }
    if (command == "version") {
        if (argc != 2) {
            std::cerr << "edgeflow: version takes no arguments\n";
            PrintUsage(std::cerr);
            return kExitUsage;
        }
        PrintVersion();
        return kExitOk;
    }
    if (command == "infer") {
        return RunInfer(argc, argv);
    }
    if (command == "doctor") {
        if (argc == 2) {
            return RunDoctor(edgeflow::SystemProbe::kDefaultModelPath);
        }
        if (argc == 3 && std::string_view(argv[2]) == "--help") {
            PrintUsage(std::cout);
            return kExitOk;
        }
        if (argc == 4 && std::string_view(argv[2]) == "--model" &&
            argv[3][0] != '\0') {
            return RunDoctor(argv[3]);
        }
        std::cerr << "edgeflow: invalid doctor arguments\n";
        PrintUsage(std::cerr);
        return kExitUsage;
    }
    std::cerr << "edgeflow: unknown command: " << command << '\n';
    PrintUsage(std::cerr);
    return kExitUsage;
}
