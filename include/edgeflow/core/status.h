#pragma once

#include <string>

namespace edgeflow {

// 统一错误码
enum class ErrorCode {
    Ok = 0,           // 成功
    InvalidArgument,  // 无效参数
    NotFound,         // 资源未找到
    PermissionDenied, // 权限不足
    IoError,          // IO错误
    RuntimeError,     // 运行错误
    Unsupported,      // 不支持
};

class Status {
public:
    // 默认构造：表示操作成功
    Status() noexcept = default;
    // 判断操作是否成功
    bool ok() const noexcept;
    // 获取错误码
    ErrorCode code() const noexcept;
    // 获取错误信息
    const std::string& message() const noexcept;
    // 静态工厂方法
    static Status Ok();
    static Status InvalidArgument(std::string message);
    static Status NotFound(std::string message);
    static Status PermissionDenied(std::string message);
    static Status IoError(std::string message);
    static Status RuntimeError(std::string message);
    static Status Unsupported(std::string message);

private:
    // 只允许通过工厂方法创建指定错误状态
    Status(ErrorCode code, std::string message);
    ErrorCode code_ = ErrorCode::Ok;
    std::string message_;
};

} // namespace edgeflow
