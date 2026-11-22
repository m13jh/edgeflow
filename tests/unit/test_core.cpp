#include "edgeflow/core/status.h"
#include "edgeflow/core/system_probe.h"
#include <iostream>
#include <string>

namespace {

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

void TestStatus() {
    // 1. 默认构造与 Ok()
    edgeflow::Status default_ok;
    CHECK(default_ok.ok());
    CHECK(default_ok.code() == edgeflow::ErrorCode::Ok);
    CHECK(default_ok.message().empty());

    const auto ok = edgeflow::Status::Ok();
    CHECK(ok.ok());
    CHECK(ok.code() == edgeflow::ErrorCode::Ok);

    // 2. 各类错误工厂方法
    const auto invalid = edgeflow::Status::InvalidArgument("bad arg");
    CHECK(!invalid.ok());
    CHECK(invalid.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(invalid.message() == "bad arg");

    const auto not_found = edgeflow::Status::NotFound("missing");
    CHECK(not_found.code() == edgeflow::ErrorCode::NotFound);

    const auto denied = edgeflow::Status::PermissionDenied("no perm");
    CHECK(denied.code() == edgeflow::ErrorCode::PermissionDenied);

    const auto io_err = edgeflow::Status::IoError("read fail");
    CHECK(io_err.code() == edgeflow::ErrorCode::IoError);

    const auto runtime_err = edgeflow::Status::RuntimeError("npu err");
    CHECK(runtime_err.code() == edgeflow::ErrorCode::RuntimeError);

    const auto unsupported = edgeflow::Status::Unsupported("not impl");
    CHECK(unsupported.code() == edgeflow::ErrorCode::Unsupported);
}

void TestSystemProbeModelChecks() {
    // 1. 空模型路径应返回 InvalidArgument
    edgeflow::SystemReport report;
    const auto empty_status = edgeflow::SystemProbe::Collect(report, "");
    CHECK(!empty_status.ok());
    CHECK(empty_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 2. 不存在的模型路径：Collect 本身成功，但 model.exists / readable 为 false
    const std::string missing_path = "/tmp/edgeflow_definitely_not_exist_12345.rknn";
    const auto missing_status = edgeflow::SystemProbe::Collect(report, missing_path);
    CHECK(missing_status.ok());
    CHECK(report.mem_total_mib > 0.0);
    CHECK(report.model.path == missing_path);
    CHECK(!report.model.exists);
    CHECK(!report.model.readable);
    CHECK(!report.model.writable);

    // 3. 如果你已经在 system_probe.cpp 里修了“目录不能算有效模型”，加上下面这组断言：
    const auto dir_status = edgeflow::SystemProbe::Collect(report, "/tmp");
    CHECK(dir_status.ok());
    CHECK(!report.model.readable);
}

void TestSystemReportToJson() {
    // 手工构造报告，纯离线测试 JSON 序列化与特殊字符转义
    edgeflow::SystemReport report;
    report.mem_total_mib = 4096.0;
    report.mem_available_mib = 2048.5;
    report.devices.push_back({ "/dev/dri/renderD129", true, true, true });
    report.thermal_zones.push_back({ "thermal_zone0", "soc-thermal", 42.5 });
    report.thermal_zones.push_back({ "thermal_zone1", "unknown", std::nullopt });
    report.npu_frequency_mhz = std::nullopt;
    report.model = { "/tmp/a\"b\\c\n.rknn", false, false, false };

    const std::string json = report.to_json();
    CHECK(json.find("\"memory\"") != std::string::npos);
    CHECK(json.find("\"devices\"") != std::string::npos);
    CHECK(json.find("\"thermal_zones\"") != std::string::npos);
    CHECK(json.find("\"frequency_mhz\": null") != std::string::npos);
    CHECK(json.find("\"temperature_c\": null") != std::string::npos);
    // 验证双引号、反斜杠、换行符被正确转义
    CHECK(json.find("/tmp/a\\\"b\\\\c\\n.rknn") != std::string::npos);
}

} // namespace

int main() {
    TestStatus();
    TestSystemProbeModelChecks();
    TestSystemReportToJson();

    if (g_failures == 0) {
        std::cout << "[PASS] all edgeflow_unit_tests passed\n";
        return 0;
    }
    std::cerr << "[FAIL] " << g_failures << " check(s) failed\n";
    return 1;
}