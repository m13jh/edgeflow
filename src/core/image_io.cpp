#include "edgeflow/core/image_io.h"
#include "edgeflow/core/status.h"
#include <cerrno>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace edgeflow {
namespace {

/**
 * std::unique_ptr 文件流释放器, 确保离开作用域时自动关闭打开的文件句柄
 */
struct FileCloser {
    /**
     * 若文件流指针非空则调用 fclose 关闭文件
     */
    void operator()(std::FILE* file) const noexcept {
        if (file != nullptr) {
            std::fclose(file);
        }
    }
};

struct JpegErrorState {
    jpeg_error_mgr base{};             // libjpeg 的错误管理器和回调函数
    jmp_buf jump{};                    // 保存发生 JPEG 错误时要跳转回去的执行环境
    char message[JMSG_STR_PARM_MAX]{}; // 保存具体的 JPEG 错误文本
};

/**
 * libjpeg 致命错误回调, 格式化错误信息并通过 longjmp 跳回调用方以避免默认 exit 终止进程
 */
void JpegErrorExit(j_common_ptr info) {
    auto* error = reinterpret_cast<JpegErrorState*>(info->err);
    (*info->err->format_message)(info, error->message); // 调用函数指针
    longjmp(error->jump, 1);
}

/**
 * libjpeg 消息回调, 捕获首个损坏数据警告文本并累计警告计数
 */
void JpegEmitMessage(j_common_ptr info, int leval) {
    if (leval < 0) {
        auto* error = reinterpret_cast<JpegErrorState*>(info->err);
        if (error->base.num_warnings == 0) {
            (*info->err->format_message)(info, error->message);
        }
        ++error->base.num_warnings;
    }
}

/**
 * RAII 守卫, 确保离开作用域时自动销毁 libjpeg 解压状态机并释放内部内存
 */
class JpegDestroyGuard {
public:
    /**
     * 构造守卫并绑定待管理的 libjpeg 解压结构体引用
     */
    explicit JpegDestroyGuard(jpeg_decompress_struct& info) noexcept
        : info_(info) {}

    /**
     * 析构时调用 jpeg_destroy_decompress 释放 libjpeg 解压资源
     */
    ~JpegDestroyGuard() noexcept {
        jpeg_destroy_decompress(&info_);
    }
    JpegDestroyGuard(const JpegDestroyGuard&) = delete;
    JpegDestroyGuard& operator=(const JpegDestroyGuard&) = delete;

private:
    jpeg_decompress_struct& info_;
};

enum class DecodeResult {
    Ok,
    JpegError,
    InvalidDimensions,
    InvalidComponents,
    TooLarge,
    ShortRead
};

/**
 * 读取 JPEG 头信息并校验尺寸与内存上限, 逐行将像素数据解码为 RGB24 格式
 */
DecodeResult DecodeJpeg(
    std::FILE* file,
    jpeg_decompress_struct* info,
    JpegErrorState* error,
    RgbImage* image
) {
    if (setjmp(error->jump) != 0) {
        return DecodeResult::JpegError;
    }
    // 1. 创建解析器
    jpeg_create_decompress(info);
    jpeg_stdio_src(info, file);
    // 2. 读取图像头信息
    if (jpeg_read_header(info, TRUE) != JPEG_HEADER_OK) {
        return DecodeResult::JpegError;
    }
    info->out_color_space = JCS_RGB;
    // 3. 开始解析
    if (!jpeg_start_decompress(info)) {
        return DecodeResult::JpegError;
    }
    // 4. 判断解析结果
    if (info->output_width == 0 || info->output_height == 0) {
        return DecodeResult::InvalidDimensions;
    }
    if (info->output_components != 3) {
        return DecodeResult::InvalidComponents;
    }
    constexpr std::size_t kMaxDecodedBytes = 256ULL * 1024 * 1024; // 最大图片大小
    const std::size_t width = info->output_width;
    const std::size_t height = info->output_height;
    if (width > std::numeric_limits<std::size_t>::max() / 3) {
        return DecodeResult::TooLarge;
    }
    const std::size_t row_bytes = width * 3;
    if (height > std::numeric_limits<std::size_t>::max() / row_bytes ||
        height * row_bytes > kMaxDecodedBytes) {
        return DecodeResult::TooLarge;
    }
    // 5. 数据赋值
    image->width = info->output_width;
    image->height = info->output_height;
    image->channels = 3;
    image->data.resize(height * row_bytes);
    while (info->output_scanline < info->output_height) {
        const std::size_t offset = static_cast<std::size_t>(info->output_scanline) * row_bytes;
        // JSAMPROW 本质上就是指向一行像素数据的指针, 通常为: unsigned char*
        // JSAMPROW row[1] 表示一个长度为 1 的指针数组
        JSAMPROW row[1] = {
            reinterpret_cast<JSAMPROW>(image->data.data() + offset)
        };
        // 把数据读取到 JSAMPROW row[1] 这个指针中去
        if (jpeg_read_scanlines(info, row, 1) != 1) {
            return DecodeResult::ShortRead;
        }
    }
    if (!jpeg_finish_decompress(info)) {
        return DecodeResult::JpegError;
    }
    if (error->base.num_warnings != 0) {
        return DecodeResult::JpegError;
    }
    return DecodeResult::Ok;
}

} // namespace

/**
 * 打开并解码指定路径的 JPEG 文件, 输出按行紧凑存储的 RGB24 图像数据
 */
Status LoadJpegRgb24(const std::string& path, RgbImage& out) {
    // 1. 判断图片是否为空
    if (path.empty()) {
        return Status::InvalidArgument("JPEG path must not be empty");
    }
    try {
        // 2. 打开文件
        std::unique_ptr<std::FILE, FileCloser> file(std::fopen(path.c_str(), "rb"));
        if (!file) {
            const int open_errno = errno;
            if (open_errno == ENOENT) {
                return Status::NotFound("JPEG file not found: " + path);
            }
            return Status::IoError("cannot open JPEG '" + path + "': " + std::strerror(open_errno));
        }
        jpeg_decompress_struct info{};
        JpegErrorState error{};
        info.err = jpeg_std_error(&error.base);
        error.base.error_exit = JpegErrorExit;
        error.base.emit_message = JpegEmitMessage;
        // RAII 管理
        JpegDestroyGuard cleanup(info);
        // 3. 解析图片
        RgbImage decoded;
        const DecodeResult result = DecodeJpeg(file.get(), &info, &error, &decoded);
        switch (result) {
        case DecodeResult::Ok:
            out = std::move(decoded);
            return Status::Ok();
        case DecodeResult::TooLarge:
            return Status::InvalidArgument("JPEG decoded image exceeds 256 MiB: " + path);
        case DecodeResult::InvalidDimensions:
            return Status::IoError("JPEG has invalid image dimensions: " + path);
        case DecodeResult::InvalidComponents:
            return Status::IoError("JPEG cannot produce RGB24 output: " + path);
        case DecodeResult::ShortRead:
            return Status::IoError("JPEG scanline decoding failed: " + path);
        case DecodeResult::JpegError:
            return Status::IoError("JPEG decode failed '" + path + "': " + (error.message[0] ? error.message : "invalid JPEG data"));
        }
        return Status::RuntimeError("unexpected JPEG decode result");
    } catch (const std::bad_alloc&) {
        return Status::RuntimeError("out of memory decoding JPEG: " + path);
    } catch (const std::length_error& e) {
        return Status::RuntimeError("invalid JPEG buffer size '" + path + "': " + e.what());
    }
}

} // namespace edgeflow