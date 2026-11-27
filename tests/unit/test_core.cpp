#include "edgeflow/backend/mock_backend.h"
#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/image_io.h"
#include "edgeflow/core/latency_stats.h"
#include "edgeflow/core/status.h"
#include "edgeflow/core/system_probe.h"
#include "edgeflow/postprocess/classification.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;

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
 * RAII 临时文件管理类：构造时写入测试字节，析构时自动删除临时文件，防止污染 /tmp
 */
class ScopedTempFile {
public:
    /**
     * 在系统临时目录下按标签与进程 PID 创建临时文件并写入字节内容
     */
    ScopedTempFile(const std::string& tag, const std::vector<std::uint8_t>& bytes) {
        const std::string filename = "edgeflow_test_" + tag + "_" + std::to_string(::getpid()) + ".tmp";
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
     * 析构时自动移除临时测试文件并忽略删除异常
     */
    ~ScopedTempFile() noexcept {
        std::error_code ec;
        fs::remove(path_, ec);
    }

    ScopedTempFile(const ScopedTempFile&) = delete;
    ScopedTempFile& operator=(const ScopedTempFile&) = delete;

    /**
     * 获取临时测试文件的完整路径
     */
    const std::string& path() const noexcept {
        return path_;
    }

private:
    std::string path_;
};

// 标准 2x2 RGB24 JFIF 测试图片（左上红、右上绿、左下蓝、右下白），不依赖任何外部文件
// clang-format off
const std::vector<std::uint8_t> kValid2x2Jpeg = {
    0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01,
    0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xff, 0xdb, 0x00, 0x43,
    0x00, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x02, 0x01, 0x01, 0x01, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x04, 0x03, 0x02, 0x02, 0x02, 0x02, 0x05, 0x04,
    0x04, 0x03, 0x04, 0x06, 0x05, 0x06, 0x06, 0x06, 0x05, 0x06, 0x06, 0x06,
    0x07, 0x09, 0x08, 0x06, 0x07, 0x09, 0x07, 0x06, 0x06, 0x08, 0x0b, 0x08,
    0x09, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x06, 0x08, 0x0b, 0x0c, 0x0b, 0x0a,
    0x0c, 0x09, 0x0a, 0x0a, 0x0a, 0xff, 0xdb, 0x00, 0x43, 0x01, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x05, 0x03, 0x03, 0x05, 0x0a, 0x07, 0x06, 0x07,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
    0x0a, 0x0a, 0xff, 0xc0, 0x00, 0x11, 0x08, 0x00, 0x02, 0x00, 0x02, 0x03,
    0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01, 0xff, 0xc4, 0x00,
    0x1f, 0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
    0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x10, 0x00,
    0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00,
    0x00, 0x01, 0x7d, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
    0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81,
    0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0, 0x24,
    0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a,
    0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56,
    0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a,
    0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86,
    0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3,
    0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6,
    0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9,
    0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xc4, 0x00,
    0x1f, 0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05,
    0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0xff, 0xc4, 0x00, 0xb5, 0x11, 0x00,
    0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04, 0x04, 0x00,
    0x01, 0x02, 0x77, 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31,
    0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08,
    0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0, 0x15,
    0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55,
    0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84,
    0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa,
    0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4,
    0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
    0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa, 0xff, 0xda, 0x00,
    0x0c, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3f, 0x00, 0xfd,
    0x60, 0xff, 0x00, 0x82, 0x70, 0x7c, 0x08, 0xf8, 0x1f, 0xe2, 0x5f, 0xf8,
    0x27, 0x8f, 0xc0, 0x5f, 0x11, 0x78, 0x8b, 0xe0, 0xd7, 0x85, 0x35, 0x0d,
    0x43, 0x50, 0xf8, 0x2f, 0xe1, 0x6b, 0x9b, 0xfb, 0xfb, 0xdf, 0x0e, 0xdb,
    0x4b, 0x35, 0xcc, 0xcf, 0xa4, 0xda, 0xb3, 0xc9, 0x23, 0xb2, 0x16, 0x77,
    0x66, 0x25, 0x8b, 0x12, 0x49, 0x24, 0x93, 0x45, 0x14, 0x57, 0xf3, 0x66,
    0x7b, 0xff, 0x00, 0x23, 0xbc, 0x57, 0xfd, 0x7c, 0x9f, 0xfe, 0x94, 0xcf,
    0xe4, 0x4e, 0x25, 0xff, 0x00, 0x92, 0x8f, 0x1b, 0xff, 0x00, 0x5f, 0x6a,
    0x7f, 0xe9, 0x6c, 0xff, 0xd9
};
// clang-format on

/**
 * 测试 Status 默认构造、Ok() 及各类错误工厂方法
 */
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

/**
 * 测试 SystemProbe::Collect 对空路径、不存在路径和目录路径的校验
 */
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

    // 3. 目录不能被识别为有效普通模型文件
    const auto dir_status = edgeflow::SystemProbe::Collect(report, "/tmp");
    CHECK(dir_status.ok());
    CHECK(!report.model.exists);
    CHECK(!report.model.readable);
}

/**
 * 测试 SystemReport::to_json 的字段输出、null 处理与特殊字符转义
 */
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

/**
 * 测试 LoadJpegRgb24 对正常 JPEG 解码及各种异常输入的错误拦截（不触发 libjpeg exit）
 */
void TestLoadJpegRgb24() {
    edgeflow::RgbImage image;

    // 1. 空路径 -> InvalidArgument
    const auto empty_status = edgeflow::LoadJpegRgb24("", image);
    CHECK(!empty_status.ok());
    CHECK(empty_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 2. 不存在的文件 -> NotFound
    const auto missing_status = edgeflow::LoadJpegRgb24("/tmp/edgeflow_missing_jpeg_98765.jpg", image);
    CHECK(!missing_status.ok());
    CHECK(missing_status.code() == edgeflow::ErrorCode::NotFound);

    // 3. 0 字节空文件 -> IoError
    ScopedTempFile empty_file("empty_jpg", {});
    const auto zero_status = edgeflow::LoadJpegRgb24(empty_file.path(), image);
    CHECK(!zero_status.ok());
    CHECK(zero_status.code() == edgeflow::ErrorCode::IoError);

    // 4. 非 JPEG 伪造内容 -> IoError（验证 setjmp/longjmp 拦截了 libjpeg 默认的 exit）
    const std::vector<std::uint8_t> fake_bytes = { 'n', 'o', 't', '_', 'j', 'p', 'g' };
    ScopedTempFile fake_file("fake_jpg", fake_bytes);
    const auto fake_status = edgeflow::LoadJpegRgb24(fake_file.path(), image);
    CHECK(!fake_status.ok());
    CHECK(fake_status.code() == edgeflow::ErrorCode::IoError);

    // 5. 截断的损坏 JPEG（仅有前 200 字节，缺少压缩图像体与 EOI） -> IoError
    const std::vector<std::uint8_t> truncated_bytes(
        kValid2x2Jpeg.begin(),
        kValid2x2Jpeg.begin() + 200
    );
    ScopedTempFile truncated_file("truncated_jpg", truncated_bytes);
    const auto truncated_status = edgeflow::LoadJpegRgb24(truncated_file.path(), image);
    CHECK(!truncated_status.ok());
    CHECK(truncated_status.code() == edgeflow::ErrorCode::IoError);

    // 6. 合法 2x2 RGB JPEG -> 成功解码为 2x2x3 = 12 字节
    ScopedTempFile valid_file("valid_jpg", kValid2x2Jpeg);
    const auto ok_status = edgeflow::LoadJpegRgb24(valid_file.path(), image);
    CHECK(ok_status.ok());
    CHECK(image.width == 2);
    CHECK(image.height == 2);
    CHECK(image.channels == 3);
    CHECK(image.data.size() == 2U * 2U * 3U);
}

/**
 * 测试 TopKPredictions 的正常排序、边界条件、
 * 非法输入及同分排序行为
 */
void TestTopKPredictions() {
    std::vector<edgeflow::ClassPrediction> predictions;

    // 1. 正常路径：4 个类别取 Top-2
    const std::vector<float> scores = {
        0.1f, 0.9f, 0.3f, 0.5f
    };
    const auto normal_status = edgeflow::TopKPredictions(scores, 2, predictions);

    CHECK(normal_status.ok());
    CHECK(predictions.size() == 2);

    if (predictions.size() == 2) {
        CHECK(predictions[0].class_id == 1);
        CHECK(predictions[0].score == 0.9f);
        CHECK(predictions[1].class_id == 3);
        CHECK(predictions[1].score == 0.5f);
    }

    // 验证原始输入没有被修改
    CHECK((scores == std::vector<float>{ 0.1f, 0.9f, 0.3f, 0.5f }));

    // 2. 空数组 -> InvalidArgument
    const auto empty_status = edgeflow::TopKPredictions(
        {}, 2, predictions
    );

    CHECK(!empty_status.ok());
    CHECK(empty_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(predictions.empty());

    // 3. k = 0 -> InvalidArgument
    const auto zero_status = edgeflow::TopKPredictions(
        scores, 0, predictions
    );

    CHECK(!zero_status.ok());
    CHECK(zero_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(predictions.empty());

    // 4. 包含 NaN -> InvalidArgument
    const float nan = std::numeric_limits<float>::quiet_NaN();

    const auto nan_status = edgeflow::TopKPredictions(
        { 0.1f, nan, 0.9f }, 2, predictions
    );

    CHECK(!nan_status.ok());
    CHECK(nan_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(predictions.empty());

    // 5. 包含正无穷 -> InvalidArgument
    const float inf = std::numeric_limits<float>::infinity();

    const auto inf_status = edgeflow::TopKPredictions(
        { 0.1f, inf, 0.9f }, 2, predictions
    );

    CHECK(!inf_status.ok());
    CHECK(inf_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(predictions.empty());

    // 6. 包含负无穷 -> InvalidArgument
    const auto neg_inf_status = edgeflow::TopKPredictions(
        { 0.1f, -inf, 0.9f }, 2, predictions
    );

    CHECK(!neg_inf_status.ok());
    CHECK(neg_inf_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(predictions.empty());

    // 7. k 大于类别总数：3 个类别取 Top-5
    const auto oversized_status = edgeflow::TopKPredictions(
        { 0.3f, 0.9f, 0.5f }, 5, predictions
    );

    CHECK(oversized_status.ok());
    CHECK(predictions.size() == 3);

    if (predictions.size() == 3) {
        CHECK(predictions[0].class_id == 1);
        CHECK(predictions[0].score == 0.9f);
        CHECK(predictions[1].class_id == 2);
        CHECK(predictions[1].score == 0.5f);
        CHECK(predictions[2].class_id == 0);
        CHECK(predictions[2].score == 0.3f);
    }

    // 8. 分数完全相等时，下标小的排在前面
    const auto tie_status = edgeflow::TopKPredictions(
        { 0.5f, 0.9f, 0.9f, 0.5f }, 4, predictions
    );

    CHECK(tie_status.ok());
    CHECK(predictions.size() == 4);

    if (predictions.size() == 4) {
        CHECK(predictions[0].class_id == 1);
        CHECK(predictions[1].class_id == 2);
        CHECK(predictions[2].class_id == 0);
        CHECK(predictions[3].class_id == 3);
    }

    // 9. 同分截断：两个最高分相同，但只取 Top-1
    const auto cutoff_status = edgeflow::TopKPredictions(
        { 0.9f, 0.9f, 0.3f }, 1, predictions
    );

    CHECK(cutoff_status.ok());
    CHECK(predictions.size() == 1);

    if (predictions.size() == 1) {
        CHECK(predictions[0].class_id == 0);
        CHECK(predictions[0].score == 0.9f);
    }
}

/**
 * 测试 ComputeLatencyStats 的基础统计指标、
 * Nearest-rank 分位数、吞吐量及非法输入校验
 */
void TestComputeLatencyStats() {
    edgeflow::LatencyStats stats;

    // 浮点数比较允许的误差
    constexpr double kEpsilon = 1e-9;

    // 1. 正常路径：构造乱序的 1.0 到 100.0
    std::vector<double> samples(100);

    std::iota(samples.begin(), samples.end(), 1.0);

    // 固定随机种子，保证测试可复现
    std::mt19937 rng(42);
    std::shuffle(samples.begin(), samples.end(), rng);

    // 保存原始数据
    const auto original = samples;

    const auto normal_status = edgeflow::ComputeLatencyStats(
        samples, stats
    );

    CHECK(normal_status.ok());
    CHECK(stats.count == 100);

    CHECK(std::abs(stats.min_ms - 1.0) < kEpsilon);
    CHECK(std::abs(stats.max_ms - 100.0) < kEpsilon);
    CHECK(std::abs(stats.mean_ms - 50.5) < kEpsilon);

    CHECK(std::abs(stats.p50_ms - 50.0) < kEpsilon);
    CHECK(std::abs(stats.p95_ms - 95.0) < kEpsilon);
    CHECK(std::abs(stats.p99_ms - 99.0) < kEpsilon);

    CHECK(
        std::abs(stats.throughput_fps - 1000.0 / 50.5) < kEpsilon
    );

    // 验证原始输入顺序没有被修改
    CHECK(samples == original);

    // 2. 空数组 -> InvalidArgument
    const auto empty_status = edgeflow::ComputeLatencyStats(
        {}, stats
    );

    CHECK(!empty_status.ok());
    CHECK(empty_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 验证旧统计结果已经被清空
    CHECK(stats.count == 0);
    CHECK(stats.min_ms == 0.0);
    CHECK(stats.max_ms == 0.0);
    CHECK(stats.mean_ms == 0.0);
    CHECK(stats.p50_ms == 0.0);
    CHECK(stats.p95_ms == 0.0);
    CHECK(stats.p99_ms == 0.0);
    CHECK(stats.throughput_fps == 0.0);

    // 3. 单个样本：所有延迟指标均为 6.5
    const auto single_status = edgeflow::ComputeLatencyStats(
        { 6.5 }, stats
    );

    CHECK(single_status.ok());
    CHECK(stats.count == 1);

    CHECK(std::abs(stats.min_ms - 6.5) < kEpsilon);
    CHECK(std::abs(stats.max_ms - 6.5) < kEpsilon);
    CHECK(std::abs(stats.mean_ms - 6.5) < kEpsilon);

    CHECK(std::abs(stats.p50_ms - 6.5) < kEpsilon);
    CHECK(std::abs(stats.p95_ms - 6.5) < kEpsilon);
    CHECK(std::abs(stats.p99_ms - 6.5) < kEpsilon);

    CHECK(
        std::abs(stats.throughput_fps - 1000.0 / 6.5) < kEpsilon
    );

    // 4. 包含负数 -> InvalidArgument
    const auto negative_status = edgeflow::ComputeLatencyStats(
        { -1.0 }, stats
    );

    CHECK(!negative_status.ok());
    CHECK(negative_status.code() == edgeflow::ErrorCode::InvalidArgument);
    CHECK(stats.count == 0);

    // 5. 包含 NaN -> InvalidArgument
    const double nan = std::numeric_limits<double>::quiet_NaN();

    const auto nan_status = edgeflow::ComputeLatencyStats(
        { 1.0, nan, 3.0 }, stats
    );

    CHECK(!nan_status.ok());
    CHECK(nan_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 6. 包含正无穷 -> InvalidArgument
    const double inf = std::numeric_limits<double>::infinity();

    const auto inf_status = edgeflow::ComputeLatencyStats(
        { 1.0, inf, 3.0 }, stats
    );

    CHECK(!inf_status.ok());
    CHECK(inf_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 7. 包含负无穷 -> InvalidArgument
    const auto neg_inf_status = edgeflow::ComputeLatencyStats(
        { 1.0, -inf, 3.0 }, stats
    );

    CHECK(!neg_inf_status.ok());
    CHECK(neg_inf_status.code() == edgeflow::ErrorCode::InvalidArgument);

    // 8. 所有样本均为 0，避免计算 FPS 时出现除零
    const auto zero_status = edgeflow::ComputeLatencyStats(
        { 0.0, 0.0, 0.0 }, stats
    );

    CHECK(zero_status.ok());
    CHECK(stats.count == 3);
    CHECK(stats.min_ms == 0.0);
    CHECK(stats.max_ms == 0.0);
    CHECK(stats.mean_ms == 0.0);
    CHECK(stats.p50_ms == 0.0);
    CHECK(stats.p95_ms == 0.0);
    CHECK(stats.p99_ms == 0.0);
    CHECK(stats.throughput_fps == 0.0);

    // 9. 偶数样本：验证 Nearest-rank 分位数计算
    const auto even_status = edgeflow::ComputeLatencyStats(
        { 40.0, 10.0, 30.0, 20.0 }, stats
    );

    CHECK(even_status.ok());
    CHECK(stats.count == 4);
    CHECK(std::abs(stats.mean_ms - 25.0) < kEpsilon);
    CHECK(std::abs(stats.p50_ms - 20.0) < kEpsilon);
    CHECK(std::abs(stats.p95_ms - 40.0) < kEpsilon);
    CHECK(std::abs(stats.p99_ms - 40.0) < kEpsilon);
}
/** 测试张量描述、分配、类型安全、布尔字节校验及移动语义。 */
void TestTensorDescAndAllocation() {
    using namespace edgeflow;
    const TensorDesc desc{ "x", { 1, 2, 3 }, DataType::Float32, TensorFormat::NCHW };
    CHECK(desc.NumElements() == 6);
    CHECK(desc.ByteSize() == 24);

    const auto max_dim = std::numeric_limits<std::uint32_t>::max();
    const TensorDesc empty{ "empty", {}, DataType::Float32, TensorFormat::NCHW };
    const TensorDesc zero{ "zero", { 1, 0, 3 }, DataType::Float32, TensorFormat::NCHW };
    const TensorDesc unknown{ "unknown", { 1, 2, 3 }, DataType::Unknown, TensorFormat::NCHW };
    const TensorDesc elem_overflow{ "overflow", { max_dim, max_dim, max_dim }, DataType::UInt8, TensorFormat::Undefined };
    const TensorDesc byte_overflow{ "overflow", { max_dim, max_dim }, DataType::Float32, TensorFormat::Undefined };
    CHECK(empty.NumElements() == 0);
    CHECK(zero.NumElements() == 0);
    CHECK(unknown.NumElements() == 6);
    CHECK(elem_overflow.NumElements() == 0);
    if constexpr (sizeof(std::size_t) == 8) {
        CHECK(byte_overflow.NumElements() != 0);
    }
    for (const auto& bad : { empty, zero, unknown, elem_overflow, byte_overflow }) {
        CHECK(bad.ByteSize() == 0);
        Tensor tensor;
        CHECK(Tensor::Allocate(bad, tensor).code() == ErrorCode::InvalidArgument);
    }

    Tensor src;
    CHECK(src.Validate().code() == ErrorCode::InvalidArgument);
    CHECK(src.SizeBytes() == 0);
    CHECK(src.Data<float>() == nullptr);
    const auto allocated = Tensor::Allocate(desc, src);
    CHECK(allocated.ok());
    if (!allocated.ok())
        return;
    CHECK(src.Validate().ok());
    CHECK(src.SizeBytes() == 24);
    auto* writable = src.MutableData<float>();
    CHECK(writable != nullptr);
    if (!writable)
        return;
    const float values[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    std::copy(std::begin(values), std::end(values), writable);
    const auto* readable = src.Data<float>();
    CHECK(readable != nullptr);
    if (readable)
        CHECK(std::equal(std::begin(values), std::end(values), readable));
    CHECK(src.Data<std::uint8_t>() == nullptr);
    CHECK(src.MutableData<std::uint8_t>() == nullptr);

    Tensor copied;
    CHECK(Tensor::FromBytes(desc, nullptr, sizeof(values), copied).code() == ErrorCode::InvalidArgument);
    CHECK(Tensor::FromBytes(desc, values, sizeof(values) - 1, copied).code() == ErrorCode::InvalidArgument);
    const auto copy_status = Tensor::FromBytes(desc, values, sizeof(values), copied);
    CHECK(copy_status.ok());
    if (copy_status.ok()) {
        const auto* data = copied.Data<float>();
        CHECK(data != nullptr);
        if (data)
            CHECK(std::equal(std::begin(values), std::end(values), data));
        CHECK(copied.RawData() != static_cast<const void*>(values));
    }

    const TensorDesc bool_desc{ "flags", { 2 }, DataType::Bool, TensorFormat::Undefined };
    const std::uint8_t valid_bool[] = { 0, 1 };
    const std::uint8_t invalid_bool[] = { 0, 2 };
    Tensor flags;
    const auto bool_status = Tensor::FromBytes(bool_desc, valid_bool, sizeof(valid_bool), flags);
    CHECK(bool_status.ok());
    if (bool_status.ok()) {
        const auto* data = flags.Data<bool>();
        CHECK(data != nullptr);
        if (data) {
            CHECK(!data[0]);
            CHECK(data[1]);
        }
    }
    CHECK(Tensor::FromBytes(bool_desc, invalid_bool, sizeof(invalid_bool), flags).code() == ErrorCode::InvalidArgument);

    Tensor moved(std::move(src));
    CHECK(src.SizeBytes() == 0);
    CHECK(src.RawData() == nullptr);
    CHECK(!src.Validate().ok());
    CHECK(moved.Validate().ok());
    CHECK(moved.SizeBytes() == sizeof(values));
    const auto* moved_data = moved.Data<float>();
    CHECK(moved_data != nullptr);
    if (moved_data)
        CHECK(std::equal(std::begin(values), std::end(values), moved_data));
}

// Tensor 不可拷贝：先分配，再移动到 vector 中。
bool AppendAllocatedTensor(const edgeflow::TensorDesc& desc, std::vector<edgeflow::Tensor>& tensors) {
    edgeflow::Tensor tensor;
    const auto status = edgeflow::Tensor::Allocate(desc, tensor);
    CHECK(status.ok());
    if (!status.ok())
        return false;
    tensors.push_back(std::move(tensor));
    return true;
}

/** 测试 MockBackend 生命周期、加载参数及输入契约。 */
void TestMockBackendLifecycleAndErrorPaths() {
    using namespace edgeflow;
    MockBackend backend;
    std::vector<Tensor> inputs, outputs;
    CHECK(!backend.IsLoaded());
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::RuntimeError);

    ModelConfig config;
    CHECK(backend.Load(config).code() == ErrorCode::InvalidArgument);
    config.model_path = "virtual/mock.rknn";
    config.expected_sha256 = std::string(64, 'a');
    CHECK(backend.Load(config).code() == ErrorCode::Unsupported);
    config.expected_sha256.clear();

    const auto check_option = [&](const char* key, const char* value) {
        ModelConfig bad = config;
        bad.backend_options[key] = value;
        CHECK(backend.Load(bad).code() == ErrorCode::InvalidArgument);
        CHECK(!backend.IsLoaded());
    };
    check_option("unknown", "1");
    check_option("num_outputs", "0");
    check_option("num_outputs", "65");
    check_option("num_outputs", "abc");
    check_option("fail_run", "yes");

    MockBackendSpec file_spec;
    file_spec.check_model_file_exists = true;
    MockBackend file_backend(file_spec);
    ScopedTempFile model("mock_model", { 1, 2, 3 });
    ModelConfig file_config;
    // 使用不存在的同级路径，避免 ENOTDIR 与 NotFound 混淆。
    file_config.model_path = model.path() + ".missing";
    CHECK(file_backend.Load(file_config).code() == ErrorCode::NotFound);
    file_config.model_path = "/tmp";
    CHECK(file_backend.Load(file_config).code() == ErrorCode::InvalidArgument);
    file_config.model_path = model.path();
    CHECK(file_backend.Load(file_config).ok());
    CHECK(file_backend.IsLoaded());

    const auto load = backend.Load(config);
    CHECK(load.ok());
    if (!load.ok())
        return;
    CHECK(backend.IsLoaded());
    CHECK(backend.InputDescs().size() == 1);
    if (backend.InputDescs().size() != 1)
        return;
    const auto expected = backend.InputDescs().front();
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::InvalidArgument);
    if (!AppendAllocatedTensor(expected, inputs))
        return;
    CHECK(backend.Run(inputs, inputs).code() == ErrorCode::InvalidArgument);
    CHECK(inputs.size() == 1);
    if (inputs.size() != 1)
        return;
    CHECK(inputs.front().Validate().ok());
    if (!AppendAllocatedTensor(expected, inputs))
        return;
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::InvalidArgument);
    inputs.clear();

    // 分别只修改 shape、dtype、format；防止一种错误掩盖另一种错误。
    const TensorDesc wrong_descs[] = {
        { expected.name, { 1, 112, 448, 3 }, DataType::UInt8, TensorFormat::NHWC },
        { expected.name, expected.dims, DataType::Float32, TensorFormat::NHWC },
        { expected.name, expected.dims, DataType::UInt8, TensorFormat::NCHW }
    };
    for (const auto& wrong : wrong_descs) {
        if (!AppendAllocatedTensor(wrong, inputs))
            return;
        CHECK(backend.Run(inputs, outputs).code() == ErrorCode::InvalidArgument);
        CHECK(backend.run_count() == 0);
        inputs.clear();
    }
    inputs.emplace_back(); // 未初始化的 Tensor
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::InvalidArgument);
    CHECK(backend.run_count() == 0);
    inputs.clear();

    if (!AppendAllocatedTensor(expected, inputs))
        return;
    CHECK(backend.Run(inputs, outputs).ok());
    CHECK(backend.run_count() == 1);
    backend.Reset();
    backend.Reset(); // Reset 应当幂等
    CHECK(!backend.IsLoaded());
    CHECK(backend.InputDescs().empty());
    CHECK(backend.OutputDescs().empty());
    CHECK(backend.run_count() == 0);
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::RuntimeError);
}

/** 测试失败 Load 不破坏旧状态以及两种故障注入方式。 */
void TestMockBackendFaultInjectionAndStrongGuarantee() {
    using namespace edgeflow;
    MockBackend backend;
    ModelConfig config;
    config.model_path = "virtual/mock.rknn";
    const auto load = backend.Load(config);
    CHECK(load.ok());
    if (!load.ok())
        return;
    CHECK(backend.InputDescs().size() == 1);
    if (backend.InputDescs().size() != 1)
        return;
    std::vector<Tensor> inputs, outputs;
    if (!AppendAllocatedTensor(backend.InputDescs().front(), inputs))
        return;
    const auto run = backend.Run(inputs, outputs);
    CHECK(run.ok());
    if (!run.ok())
        return;
    CHECK(backend.run_count() == 1);
    CHECK(!outputs.empty());

    ModelConfig bad = config;
    bad.model_path.clear();
    CHECK(backend.Load(bad).code() == ErrorCode::InvalidArgument);
    CHECK(backend.IsLoaded());
    CHECK(backend.InputDescs().size() == 1);
    CHECK(backend.OutputDescs().size() == 1);
    CHECK(backend.run_count() == 1);
    CHECK(backend.Run(inputs, outputs).ok());
    CHECK(backend.run_count() == 2);

    ModelConfig fail = config;
    fail.backend_options["fail_run"] = "true";
    const auto fail_load = backend.Load(fail);
    CHECK(fail_load.ok());
    if (!fail_load.ok())
        return;
    outputs.clear();
    if (!AppendAllocatedTensor({ "stale", { 1 }, DataType::Float32, TensorFormat::Undefined }, outputs))
        return;
    CHECK(backend.Run(inputs, outputs).code() == ErrorCode::RuntimeError);
    CHECK(outputs.empty());
    CHECK(backend.run_count() == 0);

    MockBackendSpec load_spec;
    load_spec.forced_load_status = Status::PermissionDenied("injected load error");
    MockBackend load_backend(load_spec);
    const auto injected_load = load_backend.Load(config);
    CHECK(injected_load.code() == ErrorCode::PermissionDenied);
    CHECK(injected_load.message() == "injected load error");
    CHECK(!load_backend.IsLoaded());

    MockBackendSpec run_spec;
    run_spec.forced_run_status = Status::IoError("injected run error");
    MockBackend run_backend(run_spec);
    const auto run_load = run_backend.Load(config);
    CHECK(run_load.ok());
    if (!run_load.ok())
        return;
    if (!AppendAllocatedTensor({ "stale", { 1 }, DataType::Float32, TensorFormat::Undefined }, outputs))
        return;
    const auto injected_run = run_backend.Run(inputs, outputs);
    CHECK(injected_run.code() == ErrorCode::IoError);
    CHECK(injected_run.message() == "injected run error");
    CHECK(outputs.empty());
    CHECK(run_backend.run_count() == 0);
}

/** 通过统一接口模拟三输出检测模型，并检查所有元素的确定性。 */
void TestMockBackendMultiOutputPolymorphism() {
    using namespace edgeflow;
    MockBackendSpec spec;
    spec.input_descs = { { "images", { 1, 640, 640, 3 }, DataType::UInt8, TensorFormat::NHWC } };
    spec.output_descs = {
        { "head_80", { 1, 84, 80, 80 }, DataType::Float32, TensorFormat::NCHW },
        { "head_40", { 1, 84, 40, 40 }, DataType::Float32, TensorFormat::NCHW },
        { "head_20", { 1, 84, 20, 20 }, DataType::Float32, TensorFormat::NCHW }
    };
    std::unique_ptr<IInferenceBackend> backend = std::make_unique<MockBackend>(spec);
    CHECK(backend->Capabilities().backend_name == "mock");
    CHECK(backend->Capabilities().supports_multi_output);
    ModelConfig config;
    config.model_path = "virtual/yolov8.rknn";
    const auto load = backend->Load(config);
    CHECK(load.ok());
    if (!load.ok())
        return;
    CHECK(backend->InputDescs().size() == 1);
    CHECK(backend->OutputDescs().size() == 3);
    std::vector<Tensor> inputs, outputs, again;
    if (!AppendAllocatedTensor(spec.input_descs.front(), inputs))
        return;
    auto* pixels = inputs.front().MutableData<std::uint8_t>();
    CHECK(pixels != nullptr);
    if (!pixels)
        return;
    pixels[0] = 42;
    const auto first = backend->Run(inputs, outputs);
    const auto second = backend->Run(inputs, again);
    CHECK(first.ok());
    CHECK(second.ok());
    if (!first.ok() || !second.ok())
        return;
    CHECK(outputs.size() == 3);
    CHECK(again.size() == 3);
    if (outputs.size() != 3 || again.size() != 3)
        return;
    for (std::size_t i = 0; i < spec.output_descs.size(); ++i) {
        const auto& expected = spec.output_descs[i];
        const auto& a = outputs[i];
        const auto& b = again[i];
        CHECK(a.Validate().ok());
        CHECK(b.Validate().ok());
        CHECK(a.Desc().dims == expected.dims);
        CHECK(b.Desc().dims == expected.dims);
        CHECK(a.Desc().dtype == DataType::Float32);
        CHECK(b.Desc().dtype == DataType::Float32);
        CHECK(a.Desc().fmt == expected.fmt);
        CHECK(b.Desc().fmt == expected.fmt);
        CHECK(a.SizeBytes() == expected.ByteSize());
        CHECK(b.SizeBytes() == expected.ByteSize());
        const auto* pa = a.Data<float>();
        const auto* pb = b.Data<float>();
        CHECK(pa != nullptr);
        CHECK(pb != nullptr);
        if (!pa || !pb || a.SizeBytes() != expected.ByteSize() || b.SizeBytes() != expected.ByteSize())
            continue;
        const auto n = expected.NumElements();
        CHECK(std::all_of(pa, pa + n, [](float x) {
            return std::isfinite(x);
        }));
        CHECK(std::all_of(pb, pb + n, [](float x) {
            return std::isfinite(x);
        }));
        CHECK(std::equal(pa, pa + n, pb));
    }
    // 用默认只有一个输出的 Mock，确保覆盖选项真的改变了数量。
    MockBackend dynamic_backend;
    const auto base_load = dynamic_backend.Load(config);
    CHECK(base_load.ok());
    if (!base_load.ok())
        return;
    CHECK(dynamic_backend.OutputDescs().size() == 1);
    config.backend_options["num_outputs"] = "3";
    const auto dynamic_load = dynamic_backend.Load(config);
    CHECK(dynamic_load.ok());
    if (dynamic_load.ok())
        CHECK(dynamic_backend.OutputDescs().size() == 3);
}

} // namespace

/**
 * 单元测试入口, 运行所有核心离线测试并按结果返回退出码
 */
int main() {
    TestStatus();
    TestSystemProbeModelChecks();
    TestSystemReportToJson();
    TestLoadJpegRgb24();
    TestTopKPredictions();
    TestComputeLatencyStats();
    TestTensorDescAndAllocation();
    TestMockBackendLifecycleAndErrorPaths();
    TestMockBackendFaultInjectionAndStrongGuarantee();
    TestMockBackendMultiOutputPolymorphism();

    if (g_failures == 0) {
        std::cout << "[PASS] all edgeflow_unit_tests passed\n";
        return 0;
    }
    std::cerr << "[FAIL] " << g_failures << " check(s) failed\n";
    return 1;
}
