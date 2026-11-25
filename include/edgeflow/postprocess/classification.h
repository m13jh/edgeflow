#pragma once

#include "edgeflow/core/status.h"
#include <cstddef>
#include <vector>

namespace edgeflow {

struct ClassPrediction {
    std::size_t class_id = 0; // 类别 id
    float score = 0;          // 分数
};

Status TopKPredictions(
    const std::vector<float>& scores,
    std::size_t k,
    std::vector<ClassPrediction>& out
);

} // namespace edgeflow