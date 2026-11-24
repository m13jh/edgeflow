#include "edgeflow/core/system_probe.h"
#include "edgeflow/core/status.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace edgeflow {
namespace {
namespace fs = std::filesystem;

/**
 * ProbeAccess用于探测字符设备
 */
DeviceAccess ProbeAccess(const std::string& path) {
    DeviceAccess result;
    result.path = path;
    result.exists = (::access(path.c_str(), F_OK) == 0);
    if (result.exists) {
        result.readable = (::access(path.c_str(), R_OK) == 0);
        result.writable = (::access(path.c_str(), W_OK) == 0);
    }
    return result;
}

/**
 * ProbeRegularFile用于探测普通文件
 */
DeviceAccess ProbeRegularFile(const std::string& path) {
    DeviceAccess result;
    result.path = path;
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        return result;
    }
    result.exists = true;
    result.readable = (::access(path.c_str(), R_OK) == 0);
    result.writable = (::access(path.c_str(), W_OK) == 0);
    return result;
}

/**
 * 读取文件第一行并去除首尾空白字符
 */
std::optional<std::string> ReadFirstLine(const fs::path& path) {
    std::ifstream in(path);
    std::string line;
    // 1. 文件流打开失败或没有读到内容
    if (!in || !std::getline(in, line)) {
        return std::nullopt;
    }
    // 2. 去除行首尾所有空白字符: 空格, \t, \r, \v, \f
    const auto not_space = [](unsigned char ch) {
        return !std::isspace(ch);
    };
    const auto first = std::find_if(line.begin(), line.end(), not_space);
    const auto last = std::find_if(line.rbegin(), line.rend(), not_space).base();
    return first < last ? std::string(first, last) : std::string{};
}

/**
 * 读取文件第一行并严格解析为 64 位整数
 */
std::optional<std::int64_t> ReadInteger(const fs::path& path) {
    // 1. 读第一行内容
    const auto line = ReadFirstLine(path);
    if (!line) {
        return std::nullopt;
    }
    std::istringstream stream(*line);
    std::int64_t value = 0;
    std::string extra;
    // 2. 字符串解析为 int64_t, 其他会溢出导致失败, 后面有多余内容也为失败
    // "abc" -> false, "123 extra" -> false
    if (!(stream >> value) || (stream >> extra)) {
        return std::nullopt;
    }
    return value;
}

/**
 * 从 /proc/meminfo 解析系统总内存与可用内存 (单位 MiB)
 */
Status ReadMemory(double& total_mib, double& available_mib) {
    // 1. 打开内存文件
    std::ifstream in("/proc/meminfo");
    if (!in) {
        return Status::IoError("无法读取 /proc/meminfo");
    }
    std::optional<std::uint64_t> total_kib;
    std::optional<std::uint64_t> available_kib;
    std::string line;
    // 2. 循环读取每一行
    while (std::getline(in, line)) {
        // 2.1 创建流, 以空格为分隔符读取
        std::istringstream row(line);
        std::string key, unit;
        std::uint64_t value;
        if (row >> key >> value >> unit && unit == "kB") {
            if (key == "MemTotal:") {
                total_kib = value;
            } else if (key == "MemAvailable:") {
                available_kib = value;
            }
        }
        if (total_kib && available_kib) {
            break;
        }
    }

    if (!total_kib || !available_kib) {
        return Status::IoError("无法解析 /proc/meminfo 中的 MemTotal 或 MemAvailable");
    }
    total_mib = static_cast<double>(*total_kib) / 1024.0;
    available_mib = static_cast<double>(*available_kib) / 1024.0;
    return Status::Ok();
}

/**
 * 遍历 /sys/class/thermal 读取各温区类型与温度 (单位 °C) 并按名称排序
 */
std::vector<ThermalZone> ReadThermalZone() {
    // 逐文件读取
    // 1. 打开根目录
    std::vector<ThermalZone> zones;
    const fs::path base("/sys/class/thermal");
    std::error_code ec;
    fs::directory_iterator it(base, fs::directory_options::skip_permission_denied, ec);
    if (ec) {
        return zones;
    }
    const fs::directory_iterator end;
    // 2. 循环遍历每一个文件
    for (; !ec && it != end; it.increment(ec)) {
        const fs::path path = it->path();
        const std::string name = path.filename().string();
        // 2.1 名字前缀不匹配, 跳过
        if (name.compare(0, std::string("thermal_zone").size(), "thermal_zone") != 0) {
            continue;
        }
        // 2.2 设置相关信息
        ThermalZone zone;
        zone.name = name;
        zone.type = ReadFirstLine(path / "type").value_or("");
        if (const auto milli_c = ReadInteger(path / "temp")) {
            zone.temperature_c = static_cast<double>(*milli_c) / 1000.0;
        }
        zones.push_back(zone);
    }
    // 3. 按名称排序
    std::sort(zones.begin(), zones.end(), [](const ThermalZone& a, const ThermalZone& b) {
        return a.name < b.name;
    });
    return zones;
}

/**
 * 按 JSON 规范转义特殊字符并用双引号包裹字符串
 */
std::string JsonQuote(const std::string& value) {
    // 给字符串首尾加上引号
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char ch : value) {
        switch (ch) {
        case '"':
            out += "\\\"";
            break; // 双引号 -> \" ;
        case '\\':
            out += "\\\\";
            break; // 反斜杠 -> \\ ;
        case '\b':
            out += "\\b";
            break; // 退格符 ;
        case '\f':
            out += "\\f";
            break; // 换页符 ;
        case '\n':
            out += "\\n";
            break; // 换行符 ;
        case '\r':
            out += "\\r";
            break; // 回车符 ;
        case '\t':
            out += "\\t";
            break; // 制表符 ;
        default:
            if (ch < 0x20) {
                out += "\\u00";
                out += hex[ch >> 4];
                out += hex[ch & 0x0f];
            } else {
                out += static_cast<char>(ch);
            }
        }
    }
    out += '"';
    return out;
}

/**
 * 将布尔值转换为 JSON 字符串字面量 ("true" / "false")
 */
const char* JsonBool(bool b) {
    return b ? "true" : "false";
}

/**
 * 向输出流写入路径的 exists、readable、writable 三个 JSON 字段
 */
void WriteAccessFields(std::ostringstream& out, const DeviceAccess& access) {
    out << "\"exists\": " << JsonBool(access.exists)
        << ", \"readable\": " << JsonBool(access.readable)
        << ", \"writable\": " << JsonBool(access.writable);
}

/**
 * 向输出流写入可选浮点数, 无值时输出 null
 */
void WriteOptionalNumber(std::ostringstream& out, const std::optional<double>& value) {
    if (value) {
        out << *value;
    } else {
        out << "null";
    }
}

} // namespace

/**
 * 采集内存、关键硬件设备节点、温区、NPU 频率及模型文件状态
 */
Status SystemProbe::Collect(SystemReport& out, const std::string& model_path) {
    if (model_path.empty()) {
        return Status::InvalidArgument("模型路径不能为空");
    }
    SystemReport report;
    const Status memory_status = ReadMemory(report.mem_total_mib, report.mem_available_mib);
    if (!memory_status.ok()) {
        return memory_status;
    }
    constexpr std::array<const char*, 4> paths = {
        "/dev/dri/renderD129",
        "/dev/rga",
        "/dev/mpp_service",
        "/dev/dma_heap/system"
    };
    // 测试字符设备文件
    for (const char* path : paths) {
        report.devices.push_back(ProbeAccess(path));
    }

    report.thermal_zones = ReadThermalZone();
    constexpr const char* npu_freq_path = "/sys/class/devfreq/27700000.npu/cur_freq";
    const auto freq_hz = ReadInteger(npu_freq_path);
    if (freq_hz && *freq_hz >= 0) {
        report.npu_frequency_mhz = static_cast<double>(*freq_hz) / 1'000'000.0;
    }
    // 测试普通文件
    report.model = ProbeRegularFile(model_path);
    out = std::move(report);
    return Status::Ok();
}

/**
 * 将系统探测报告序列化为格式化 JSON 字符串
 */
std::string SystemReport::to_json() const {
    std::ostringstream out;
    out.imbue(std::locale::classic()); // 强制使用标准 "C" 区域设置,确保小数点永远是 .
    out << std::setprecision(15);
    out << "{\n  \"memory\": {\n"
        << "    \"mem_total_mib\": " << mem_total_mib << ",\n"
        << "    \"mem_available_mib\": " << mem_available_mib << "\n"
        << "  },\n  \"devices\": {\n";
    for (std::size_t i = 0; i < devices.size(); ++i) {
        out << "    " << JsonQuote(devices[i].path) << ": {";
        WriteAccessFields(out, devices[i]);
        out << "}" << (i + 1 < devices.size() ? "," : "") << '\n';
    }

    out << "  },\n  \"thermal_zones\": [\n";
    for (std::size_t i = 0; i < thermal_zones.size(); ++i) {
        const auto& zone = thermal_zones[i];
        out << "    {\"name\": " << JsonQuote(zone.name)
            << ", \"type\": " << JsonQuote(zone.type)
            << ", \"temperature_c\": ";
        WriteOptionalNumber(out, zone.temperature_c);
        out << "}" << (i + 1 < thermal_zones.size() ? "," : "") << '\n';
    }

    out << "  ],\n  \"npu\": {\"frequency_mhz\": ";
    WriteOptionalNumber(out, npu_frequency_mhz);
    out << "},\n  \"model\": {\"path\": " << JsonQuote(model.path) << ", ";
    WriteAccessFields(out, model);
    out << "}\n}\n";
    return out.str();
}

} // namespace edgeflow