#include "edgeflow/postprocess/classification.h"
#include "edgeflow/core/status.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace edgeflow {

/**
 * 校验分类得分并按分数降序提取前 K 个类别预测结果, 同分时按类别编号升序排列
 */
Status TopKPredictions(
    const std::vector<float>& scores,
    std::size_t k,
    std::vector<ClassPrediction>& out
) {
    // 1. 清空上一次结果
    out.clear();
    // 2. 校验参数
    if (scores.empty()) {
        return Status::InvalidArgument("classification scores must not be empty");
    }
    if (k == 0) {
        return Status::InvalidArgument("top-k must be greater than zero");
    }
    if (
        !std::all_of(scores.begin(), scores.end(), [](float f) {
            return std::isfinite(f);
        })
    ) {
        return Status::InvalidArgument("classification scores contain non-finite values");
    }
    // 3. 构建索引对
    const std::size_t actual_k = std::min(k, scores.size());
    std::vector<std::size_t> order(scores.size());
    std::iota(order.begin(), order.end(), std::size_t{ 0 });
    std::partial_sort(order.begin(), order.begin() + actual_k, order.end(), [&](std::size_t a, std::size_t b) {
        if (scores[a] == scores[b]) {
            return a < b;
        }
        return scores[a] > scores[b];
    });
    // 4. 组装结果
    out.reserve(actual_k);
    for (std::size_t i = 0; i < actual_k; ++i) {
        out.push_back(ClassPrediction{ order[i], scores[order[i]] });
    }
    return Status::Ok();
}

} // namespace edgeflow