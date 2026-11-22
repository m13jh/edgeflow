#include "edgeflow/core/system_probe.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <string_view>

namespace {

constexpr int kExitOk = 0;
constexpr int kExitHealthFailed = 1;
constexpr int kExitUsage = 2;
constexpr std::string_view kNpuNode = "/dev/dri/renderD129";

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

void PrintVersion() {
    std::cout << "EdgeFlow " << EDGEFLOW_VERSION << '\n'
              << "Git commit: " << EDGEFLOW_GIT_COMMIT << '\n'
              << "Build type: " << EDGEFLOW_BUILD_TYPE << '\n'
              << "Target architecture: " << EDGEFLOW_TARGET_ARCH << '\n'
              << "RKNN backend: " << (EDGEFLOW_WITH_RKNN ? "ON" : "OFF") << '\n'
              << "RGA backend: " << (EDGEFLOW_WITH_RGA ? "ON" : "OFF") << '\n';
}

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

} // namespace

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
