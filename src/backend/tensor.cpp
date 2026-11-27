#include "edgeflow/backend/tensor.h"
#include "edgeflow/core/status.h"
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace edgeflow {

static_assert(CHAR_BIT == 8 && sizeof(bool) == 1);
static_assert(sizeof(Float16Bits) == 2);
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);

std::size_t DataTypeSize(DataType dtype) noexcept {
    switch (dtype) {
    case DataType::UInt8:
    case DataType::Int8:
    case DataType::Bool:
        return 1;
    case DataType::UInt16:
    case DataType::Int16:
    case DataType::Float16:
        return 2;
    case DataType::UInt32:
    case DataType::Int32:
    case DataType::Float32:
        return 4;
    case DataType::Int64:
        return 8;
    default:
        return 0;
    }
}

const char* DataTypeName(DataType dtype) noexcept {
    switch (dtype) {
    case DataType::UInt8:
        return "UInt8";
    case DataType::Int8:
        return "Int8";
    case DataType::UInt16:
        return "UInt16";
    case DataType::Int16:
        return "Int16";
    case DataType::UInt32:
        return "UInt32";
    case DataType::Int32:
        return "Int32";
    case DataType::Int64:
        return "Int64";
    case DataType::Float16:
        return "Float16";
    case DataType::Float32:
        return "Float32";
    case DataType::Bool:
        return "Bool";
    default:
        return "Unknown";
    }
}

const char* TensorFormatName(TensorFormat fmt) noexcept {
    switch (fmt) {
    case TensorFormat::NCHW:
        return "NCHW";
    case TensorFormat::NHWC:
        return "NHWC";
    case TensorFormat::NC1HWC2:
        return "NC1HWC2";
    default:
        return "Undefined";
    }
}

/**
 * 计算一个张量的元素个数
 */
std::size_t TensorDesc::NumElements() const noexcept {
    if (dims.empty()) {
        return 0;
    }
    constexpr size_t max = std::numeric_limits<std::size_t>::max();
    std::size_t count = 1;
    for (std::uint32_t dim : dims) {
        if (dim == 0 || count > max / dim) {
            return 0;
        }
        count *= dim;
    }
    return count;
}

/**
 * 计算一个张量的大小
 */
std::size_t TensorDesc::ByteSize() const noexcept {
    const std::size_t count = NumElements();
    const std::size_t typeSize = DataTypeSize(dtype);
    if (count == 0 || typeSize == 0) {
        return 0;
    }
    constexpr std::size_t max = std::numeric_limits<std::size_t>::max();
    if (count > max / typeSize) {
        return 0;
    }
    return count * typeSize;
}

/**
 * 只分配内存, 不需要外部数据
 */
Status Tensor::Allocate(TensorDesc desc, Tensor& out) {
    const std::size_t bytes = desc.ByteSize();
    if (bytes == 0) {
        return Status::InvalidArgument("Invalid tensor shape, dtype or byte size");
    }
    const std::size_t count = desc.NumElements();
    Tensor tmp;
    try {
        switch (desc.dtype) {
        case DataType::UInt8:
            tmp.data_ = AllocateStorage<std::uint8_t>(count);
            break;
        case DataType::Int8:
            tmp.data_ = AllocateStorage<std::int8_t>(count);
            break;
        case DataType::UInt16:
            tmp.data_ = AllocateStorage<std::uint16_t>(count);
            break;
        case DataType::Int16:
            tmp.data_ = AllocateStorage<std::int16_t>(count);
            break;
        case DataType::UInt32:
            tmp.data_ = AllocateStorage<std::uint32_t>(count);
            break;
        case DataType::Int32:
            tmp.data_ = AllocateStorage<std::int32_t>(count);
            break;
        case DataType::Int64:
            tmp.data_ = AllocateStorage<std::int64_t>(count);
            break;
        case DataType::Float16:
            tmp.data_ = AllocateStorage<Float16Bits>(count);
            break;
        case DataType::Float32:
            tmp.data_ = AllocateStorage<float>(count);
            break;
        case DataType::Bool:
            tmp.data_ = AllocateStorage<bool>(count);
            break;
        default:
            return Status::InvalidArgument("Unsupported tensor dtype");
        }
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("Failed to allocate tensor storage");
    }
    tmp.desc_ = std::move(desc);
    tmp.byte_size_ = bytes;
    out = std::move(tmp);
    return Status::Ok();
}

/**
 * 分配内存, 并且复制外部数据
 */
Status Tensor::FromBytes(
    TensorDesc desc,
    const void* src,
    std::size_t byte_size,
    Tensor& out
) {
    const std::size_t expected = desc.ByteSize();
    if (!src || expected == 0 || byte_size != expected) {
        return Status::InvalidArgument("Invalid tensor source or byte size");
    }
    Tensor tmp;
    Status status = Allocate(desc, tmp);
    if (!status.ok()) {
        return status;
    }
    // 区分 bool 数据和其他数据类型拷贝, bool 不能用 memcpy, 错误数据可能会触发未定义
    if (tmp.desc_.dtype == DataType::Bool) {
        // 转为无符号字节解析源数据
        const auto* input = static_cast<const unsigned char*>(src);
        // 获取目标地址
        auto* dst = tmp.MutableData<bool>();
        // 逐字节校验与转换
        for (std::size_t i = 0; i < byte_size; ++i) {
            if (input[i] > 1) {
                return Status::InvalidArgument("Invalid Bool byte (expected 0 or 1)");
            }
            dst[i] = (input[i] != 0);
        }
    } else {
        std::memcpy(tmp.data_.get(), src, byte_size);
    }
    out = std::move(tmp);
    return Status::Ok();
}

/**
 * 内存是否合法
 */
bool Tensor::HasValidHostStorage() const noexcept {
    if (memory_type_ != MemoryType::Host || !data_) {
        return false;
    }
    const std::size_t expected = desc_.ByteSize();
    return expected != 0 && byte_size_ == expected;
}

/**
 * 是否合法, 写死为 Host
 */
Status Tensor::Validate() const {
    if (memory_type_ != MemoryType::Host) {
        return Status::InvalidArgument("Only Host memory is supported");
    }
    if (desc_.ByteSize() == 0) {
        return Status::InvalidArgument("Invalid tensor descriptor");
    }
    if (!HasValidHostStorage()) {
        return Status::InvalidArgument("Tensor storage does not match descriptor");
    }
    return Status::Ok();
}

} // namespace edgeflow