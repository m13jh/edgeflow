#include "edgeflow/core/status.h"

namespace edgeflow {

// 私有构造函数
Status::Status(ErrorCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

// 判断是否成功
bool Status::ok() const noexcept {
    return code_ == ErrorCode::Ok;
}

// 获取错误码
ErrorCode Status::code() const noexcept {
    return code_;
}

// 获取错误信息
const std::string& Status::message() const noexcept {
    return message_;
}

// 成功状态
Status Status::Ok() {
    return Status();
}

// 非法参数
Status Status::InvalidArgument(std::string message) {
    return Status(ErrorCode::InvalidArgument, std::move(message));
}

// 资源不存在
Status Status::NotFound(std::string message) {
    return Status(ErrorCode::NotFound, std::move(message));
}

// 权限不足
Status Status::PermissionDenied(std::string message) {
    return Status(ErrorCode::PermissionDenied, std::move(message));
}

// 文件或设备 I/O 错误
Status Status::IoError(std::string message) {
    return Status(ErrorCode::IoError, std::move(message));
}

// 运行时错误
Status Status::RuntimeError(std::string message) {
    return Status(ErrorCode::RuntimeError, std::move(message));
}

// 不支持的操作
Status Status::Unsupported(std::string message) {
    return Status(ErrorCode::Unsupported, std::move(message));
}

} // namespace edgeflow