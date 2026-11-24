#if EDGEFLOW_WITH_RKNN

#include "edgeflow/backend/rknn_context.h"

#endif

#include "edgeflow/core/image_io.h"
#include "edgeflow/core/system_probe.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ratio>
#include <string>
#include <string_view>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitHealthFailed = 1;
constexpr int kExitUsage = 2;
constexpr std::string_view kNpuNode = "/dev/dri/renderD129";

/**
 * 打印 edgeflow 命令行帮助信息与退出码说明
 */
void PrintUsage(std::ostream& os) {
    os << "EdgeFlow - edge AI runtime diagnostic CLI\n\n"
       << "Usage:\n"
       << "  edgeflow version\n"
       << "  edgeflow doctor [--model <path>]\n"
       << "  edgeflow --help\n\n"
       << "Commands:\n"
       << "  version  Show version, commit, build type, architecture, backend flags\n"
       << "  doctor   Print a JSON device report to stdout\n\n"
       << "Exit codes:\n"
       << "  0  Success (doctor: required checks passed)\n"
       << "  1  Probe failure, inaccessible NPU node, or missing/unreadable model\n"
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

void PrintTensor(const char* kind, const edgeflow::RknnTensorInfo& tensor) {
    std::cout << kind << " index=" << tensor.index << " name=" << tensor.name
              << " dims=[";
    for (std::size_t i = 0; i < tensor.dims.size(); ++i) {
        if (i != 0) {
            std::cout << ", ";
        }
        std::cout << tensor.dims[i];
    }
    std::cout << "] fmt=" << tensor.format << " type=" << tensor.type
              << " size=" << tensor.size << '\n';
}

bool CheckImageShape(const edgeflow::RknnTensorInfo& input, const edgeflow::RgbImage& image) {
    const auto& dims = input.dims;
    const bool nhwc = input.format == "NHWC";
    if ((!nhwc && input.format != "NCHW") || dims.size() != 4 ||
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

int InferImage(const std::string& model_path, const std::string& input_path) {
    // 1. 加载图片
    edgeflow::RgbImage image;
    auto status = edgeflow::LoadJpegRgb24(input_path, image);
    if (!status.ok()) {
        std::cerr << "edgeflow infer: reading " << input_path << ": " << status.message() << '\n';
        return kExitHealthFailed;
    }
    // 2. 加载模型
    edgeflow::RknnContext ctx;
    const auto load_start = std::chrono::steady_clock::now(); // ns
    status = ctx.LoadFromFile(model_path);
    const auto load_time = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start
    );
    if (!status.ok()) {
        std::cerr << "edgeflow infer: loading " << model_path << ": " << status.message() << '\n';
        return kExitHealthFailed;
    }
    // 3. 查询 rknn sdk 版本
    edgeflow::RknnSdkVersion version;
    status = ctx.QuerySdkVersion(version);
    if (!status.ok()) {
        std::cerr << "edgeflow infer: querying SDK version: " << status.message() << '\n';
        return kExitHealthFailed;
    }
    std::cout << "Model load: " << load_time.count() << " ms\n"
              << "RKNN SDK: " << version.api_version
              << "\nDriver: " << version.drv_version << '\n';
    // 4. 输出 Tensor 数据
    for (const auto& tensor : ctx.inputs()) {
        PrintTensor("input", tensor);
    }
    for (const auto& tensor : ctx.outputs()) {
        PrintTensor("output", tensor);
    }
    if (ctx.inputs().size() != 1) {
        std::cerr << "edgeflow infer: expected 1 input tensor, got " << ctx.inputs().size() << '\n';
        return kExitHealthFailed;
    }
    if (!CheckImageShape(ctx.inputs().front(), image)) {
        return kExitHealthFailed;
    }
    // 5. 执行推理
    std::vector<std::vector<float>> outputs;
    const auto infer_start = std::chrono::steady_clock::now();
    status = ctx.RunRgbUint8(image.data.data(), image.data.size(), outputs);
    const auto infer_time = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - infer_start
    );
    if (!status.ok()) {
        std::cerr << "edgeflow infer: inference failed: " << status.message() << '\n';
        return kExitHealthFailed;
    }
    std::cout << "RunRgbUint8: " << infer_time.count() << " ms\n";
    // 6. 打印前 5 条信息
    // 原本输出为 0: 0.1, 1: 0.9, 2: 0.3, 3: 0.2, 4: 0.01
    constexpr std::size_t kTop = 5;
    if (outputs.empty() || outputs.front().size() < kTop) {
        std::cerr << "edgeflow infer: output[0] must contain at least 5 class scores\n";
        return kExitHealthFailed;
    }
    // 有错误数据直接报错
    const auto& scores = outputs.front();
    if (
        !std::all_of(scores.begin(), scores.end(), [](float s) {
            // 正常数值返回 true
            return std::isfinite(s);
        })
    ) {
        // 不能是非有限小数
        std::cerr << "edgeflow infer: output[0] contains a non-finite class score\n";
        return kExitHealthFailed;
    }
    // 不修改原 scores 数组的前提下, 高效找出分数最高的前 kTop 个类别
    // 排序后, 1: 0.9, 2: 0.3, 3: 0.2, 0: 0.1, 4: 0.01
    std::vector<std::size_t> order(scores.size());
    // std::iota 连续递增填充
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::partial_sort(order.begin(), order.begin() + kTop, order.end(), [&scores](std::size_t a, std::size_t b) {
        if (scores[a] == scores[b])
            return a < b;
        return scores[a] > scores[b];
    });
    std::cout << "---- Top5 ----\n"
              << std::fixed << std::setprecision(6);
    for (std::size_t rank = 0; rank < kTop; ++rank) {
        const auto index = order[rank];
        std::cout << scores[index] << " - " << index << '\n';
    }
    return kExitOk;
}

#endif

/**
 * 1. edgeflow infer --model mobilenet_v1.rknn --input dog.jpg
 * 2. edgeflow infer --help
 */
int RunInfer(int argc, char* argv[]) {
    if (argc == 3 && std::string_view(argv[2]) == "--help") {
        PrintUsage(std::cout);
        return kExitOk;
    }
    std::string model_path;
    std::string input_path;
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (flag != "--model" && flag != "--input") {
            std::cerr << "edgeflow infer: unknown argument: " << flag << '\n';
            return kExitUsage;
        }
        // 这里必须为引用
        auto& value = flag == "--model" ? model_path : input_path;
        if (!value.empty()) {
            std::cerr << "edgeflow infer: duplicate option: " << flag << '\n';
            return kExitUsage;
        }
        if (i + 1 == argc || argv[i + 1][0] == '\0' || argv[i + 1][0] == '-') {
            std::cerr << "edgeflow infer: " << flag << " requires a path\n";
            return kExitUsage;
        }
        value = argv[++i];
    }
    if (input_path.empty()) {
        std::cerr << "edgeflow infer: --input <jpeg> is required\n";
        return kExitUsage;
    }
#if EDGEFLOW_WITH_RKNN
    if (model_path.empty())
        model_path = edgeflow::SystemProbe::kDefaultModelPath;
    return InferImage(model_path, input_path);
#else
    (void)model_path; // 忽略未使用警告
    std::cerr << "edgeflow infer: RKNN backend is disabled (EDGEFLOW_WITH_RKNN=OFF)\n";
    return kExitHealthFailed;
#endif
}

} // namespace

/**
 * CLI 入口, 解析命令行参数并分发到 help / version / doctor 子命令
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
