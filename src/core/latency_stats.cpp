#include "edgeflow/core/latency_stats.h"
#include "edgeflow/core/status.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

namespace edgeflow {
namespace {

/**
 * 基于已升序排列的样本数组, 按最近秩法 (Nearest-rank) 计算指定百分位延迟
 */
double NearestRank(const std::vector<double>& sorted, double percentile) {
    const std::size_t n = sorted.size();
    // std::ceil 向上取整
    const auto rank = static_cast<std::size_t>(std::ceil(n * percentile));
    // 将排名限制在 [1, n], 防止越界
    const std::size_t safe_rank = std::clamp(rank, std::size_t{ 1 }, n);
    return sorted[safe_rank - 1];
}

} // namespace

/**
 * 校验耗时样本并计算最小、最大、平均、P50/P95/P99 分位延迟及估算吞吐率
 */
Status ComputeLatencyStats(const std::vector<double>& samples_ms, LatencyStats& out) {
    // 1. 清空上一次记录
    out = LatencyStats{};
    // 2. 校验参数
    if (samples_ms.empty()) {
        return Status::InvalidArgument("latency samples must not be empty");
    }
    if (
        !std::all_of(samples_ms.begin(), samples_ms.end(), [](double ms) {
            return std::isfinite(ms) && ms >= 0.0;
        })
    ) {
        return Status::InvalidArgument("latency samples must be finite and non-negative");
    }
    // 3. 排序
    std::vector<double> sorted = samples_ms;
    std::sort(sorted.begin(), sorted.end());
    // 4. 统计
    out.p50_ms = NearestRank(sorted, 0.5);
    out.p95_ms = NearestRank(sorted, 0.95);
    out.p99_ms = NearestRank(sorted, 0.99);
    out.min_ms = sorted.front();
    out.max_ms = sorted.back();
    out.count = sorted.size();
    // 平均值
    out.mean_ms = std::accumulate(sorted.begin(), sorted.end(), 0.0) / out.count;
    // 估算串行推理吞吐量, 帧率
    if (out.mean_ms > 0.0) {
        out.throughput_fps = 1000.0 / out.mean_ms;
    }
    return Status::Ok();
}

} // namespace edgeflow