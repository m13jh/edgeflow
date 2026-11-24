#include "edgeflow/core/status.h"

namespace edgeflow {

/**
 * 私有构造函数, 按错误码与错误信息构造 Status
 */
Status::Status(ErrorCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

/**
 * 判断当前状态是否为成功
 */
bool Status::ok() const noexcept {
    return code_ == ErrorCode::Ok;
}

/**
 * 获取当前状态的错误码
 */
ErrorCode Status::code() const noexcept {
    return code_;
}

/**
 * 获取当前状态的错误描述信息
 */
const std::string& Status::message() const noexcept {
    return message_;
}

/**
 * 构造成功状态
 */
Status Status::Ok() {
    return Status();
}

/**
 * 构造非法参数错误状态
 */
Status Status::InvalidArgument(std::string message) {
    return Status(ErrorCode::InvalidArgument, std::move(message));
}

/**
 * 构造资源不存在错误状态
 */
Status Status::NotFound(std::string message) {
    return Status(ErrorCode::NotFound, std::move(message));
}

/**
 * 构造权限不足错误状态
 */
Status Status::PermissionDenied(std::string message) {
    return Status(ErrorCode::PermissionDenied, std::move(message));
}

/**
 * 构造文件或设备 I/O 错误状态
 */
Status Status::IoError(std::string message) {
    return Status(ErrorCode::IoError, std::move(message));
}

/**
 * 构造运行时错误状态
 */
Status Status::RuntimeError(std::string message) {
    return Status(ErrorCode::RuntimeError, std::move(message));
}

/**
 * 构造不支持操作错误状态
 */
Status Status::Unsupported(std::string message) {
    return Status(ErrorCode::Unsupported, std::move(message));
}

} // namespace edgeflow