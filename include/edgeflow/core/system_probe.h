#pragma once

#include "edgeflow/core/status.h"
#include <optional>
#include <string>
#include <vector>

namespace edgeflow {

// 设备状态
struct DeviceAccess {
    std::string path;      // 设备路径
    bool exists = false;   // 是否存在
    bool readable = false; // 可读
    bool writable = false; // 可写
};

// 温度传感器信息
struct ThermalZone {
    std::string name;                    // 名称
    std::string type;                    // 类型
    std::optional<double> temperature_c; // 温度, 读不到温度时为 nullopt
};

// 完整设备探测报告
struct SystemReport {
    std::vector<DeviceAccess> devices;       // 总设备情况
    double mem_total_mib = 0.0;              // 总内存
    double mem_available_mib = 0.0;          // 可用内存
    std::vector<ThermalZone> thermal_zones;  // 设备温度信息
    std::optional<double> npu_frequency_mhz; // npu 频率
    DeviceAccess model;                      // 模型文件
    std::string to_json() const;             // 结果转换为 json
};

// 设备信息采集器
class SystemProbe {
public:
    static constexpr const char* kDefaultModelPath = "/usr/share/model/RK3576/mobilenet_v1.rknn";

    static Status Collect(SystemReport& out, const std::string& model_path = kDefaultModelPath);
};

} // namespace edgeflow