#pragma once

#include "edgeflow/core/status.h"
#include <cstddef>
#include <vector>

namespace edgeflow {

struct LatencyStats {
    std::size_t count = 0;       // 样本数
    double min_ms = 0.0;         // 最小延时
    double max_ms = 0.0;         // 最大延时
    double mean_ms = 0.0;        // 平均延时
    double p50_ms = 0.0;         // 50% 分位延迟
    double p95_ms = 0.0;         // 95% 分位延迟
    double p99_ms = 0.0;         // 99% 分位延迟
    double throughput_fps = 0.0; // 串行推理的估算吞吐量
};

Status ComputeLatencyStats(const std::vector<double>& samples_ms, LatencyStats& out);

} // namespace edgeflow