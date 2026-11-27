#pragma once

#include "edgeflow/core/status.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace edgeflow {

/**
 * Tensor 数据类型
 */
enum class DataType {
    Unknown = 0,
    UInt8,
    Int8,
    UInt16,
    Int16,
    UInt32,
    Int32,
    Int64,
    Float16,
    Float32,
    Bool,
};

/**
 * Tensor 格式
 */
enum class TensorFormat {
    Undefined = 0,
    NCHW,
    NHWC,
    NC1HWC2,
};

/**
 * 内存类型
 */
enum class MemoryType {
    Host = 0,
    DmaBuf,
    CudaDevice,
    CudaManaged,
};

/**
 * C++17 没有 float16 类型, 因此使用 uint16_t 表示
 */
struct Float16Bits {
    std::uint16_t bits = 0;
};

std::size_t DataTypeSize(DataType dtype) noexcept;
const char* DataTypeName(DataType dtype) noexcept;
const char* TensorFormatName(TensorFormat fmt) noexcept;

/**
 * 张量描述信息
 */
struct TensorDesc {
    std::string name;
    std::vector<std::uint32_t> dims;
    DataType dtype = DataType::Unknown;
    TensorFormat fmt = TensorFormat::Undefined;

    std::size_t NumElements() const noexcept;
    std::size_t ByteSize() const noexcept;
};

class Tensor {
public:
    Tensor() = default;
    ~Tensor() = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    /**
     * 移动构造
     */
    Tensor(Tensor&& other) noexcept
        : desc_(std::move(other.desc_)),
          memory_type_(std::exchange(other.memory_type_, MemoryType::Host)),
          byte_size_(std::exchange(other.byte_size_, 0)),
          data_(std::exchange(other.data_, Storage{ nullptr, nullptr })) {
        other.desc_ = {};
    }

    /**
     * 移动赋值
     */
    Tensor& operator=(Tensor&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        desc_ = std::move(other.desc_);
        memory_type_ = std::exchange(other.memory_type_, MemoryType::Host);
        byte_size_ = std::exchange(other.byte_size_, 0);
        data_ = std::exchange(other.data_, Storage{ nullptr, nullptr });
        other.desc_ = {};
        return *this;
    }

    Status Validate() const;
    // 只分配内存, 不需要外部数据
    static Status Allocate(TensorDesc desc, Tensor& out);
    // 分配内存, 并且复制外部数据
    static Status FromBytes(TensorDesc desc, const void* src, std::size_t byte_size, Tensor& out);

    const TensorDesc& Desc() const noexcept {
        return desc_;
    }

    MemoryType GetMemoryType() const noexcept {
        return memory_type_;
    }

    std::size_t SizeBytes() const noexcept {
        return data_ ? byte_size_ : 0;
    }

    /**
     * 返回智能指针内部管理的原始指针
     */
    const void* RawData() const noexcept {
        return data_.get();
    }

    template <typename T>
    const T* Data() const noexcept {
        if (!Matches<T>() || !HasValidHostStorage()) {
            return nullptr;
        }
        return static_cast<const T*>(data_.get());
    }

    template <typename T>
    T* MutableData() noexcept {
        if (!Matches<T>() || !HasValidHostStorage()) {
            return nullptr;
        }
        return static_cast<T*>(data_.get());
    }

private:
    /**
     * 返回对应类型枚举
     */
    template <typename T>
    static constexpr DataType TypeOf() noexcept {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
            return DataType::UInt8;
        } else if constexpr (std::is_same_v<T, std::int8_t>) {
            return DataType::Int8;
        } else if constexpr (std::is_same_v<T, std::uint16_t>) {
            return DataType::UInt16;
        } else if constexpr (std::is_same_v<T, std::int16_t>) {
            return DataType::Int16;
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
            return DataType::UInt32;
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
            return DataType::Int32;
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
            return DataType::Int64;
        } else if constexpr (std::is_same_v<T, Float16Bits>) {
            return DataType::Float16;
        } else if constexpr (std::is_same_v<T, float>) {
            return DataType::Float32;
        } else if constexpr (std::is_same_v<T, bool>) {
            return DataType::Bool;
        } else {
            return DataType::Unknown;
        }
    }

    /**
     * 类型不为 Unknown, 且与 desc_ 相匹配
     */
    template <typename T>
    bool Matches() const noexcept {
        return TypeOf<T>() != DataType::Unknown && TypeOf<T>() == desc_.dtype;
    }

    // 智能指针内部持有的是 void*, 销毁对象方式
    using Storage = std::unique_ptr<void, void (*)(void*)>;

    /**
     * 分配不同类型的内存
     */
    template <typename T>
    static Storage AllocateStorage(std::size_t count) {
        return Storage(new T[count]{}, [](void* ptr) noexcept {
            delete[] static_cast<T*>(ptr);
        });
    }

    // 内存是否合法
    bool HasValidHostStorage() const noexcept;

    TensorDesc desc_;
    MemoryType memory_type_ = MemoryType::Host;
    std::size_t byte_size_ = 0;
    Storage data_{ nullptr, nullptr };
};

} // namespace edgeflow