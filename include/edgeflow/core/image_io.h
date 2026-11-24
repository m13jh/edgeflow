#pragma once

#include "edgeflow/core/status.h"
#include <cstdint>
#include <string>
#include <vector>

namespace edgeflow {

struct RgbImage {
    std::uint32_t width = 0;        // 图片宽度
    std::uint32_t height = 0;       // 图片高度
    std::uint32_t channels = 3;     // 图片通道数
    std::vector<std::uint8_t> data; // 图片数据，按行存储
};

Status LoadJpegRgb24(const std::string& path, RgbImage& out);

} // namespace edgeflow