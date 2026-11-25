# EdgeFlow

面向 Rockchip RK3576 等嵌入式 Linux 边缘设备的轻量级 C++17 AI 推理运行时与诊断工具（当前版本：`v0.1.0`）。

## 核心特性（v0.1.0）

- **板端健康诊断 (`doctor`)**：探测内存、NPU/RGA/MPP/DMA-BUF 设备节点读写权限、温区温度、NPU 频率与模型可读性，输出标准 JSON。
- **RKNN RAII 封装**：安全管理 `rknn_context` 生命周期（禁拷贝、支持移动、异常/错误自动释放），SDK 头文件严格 `PRIVATE` 隔离。
- **图像分类与延迟统计 (`infer`)**：支持 JPEG RGB24 解码、Top-K 分类后处理，以及带预热（Warmup）的多轮推理延迟分位数统计（Min/Mean/P50/P95/P99/Max/FPS）。
- **跨平台可测性**：支持 `-DEDGEFLOW_WITH_RKNN=OFF` 脱离硬件 SDK 在任意主机（如 x86_64 CI）编译并运行核心单测。

## 目录结构

```text
edgeflow/
├── apps/                 # CLI 入口 (edgeflow)
├── configs/              # 板端环境基线快照 (device-report.json)
├── docs/                 # 基线与设计文档
├── include/edgeflow/     # 公共头文件 (core / backend / postprocess)
├── src/                  # 模块实现 (core / backend/rknn / postprocess)
├── tests/                # 自动化测试 (unit / hardware / integration)
└── third_party/rknn/     # 锁定的 RKNN C API 头文件 (2.0.0b0)
```

## 环境依赖

- Linux（AArch64 板端或 x86_64 主机）
- CMake >= 3.20，Ninja，支持 C++17 的编译器（GCC 13+ / Clang）
- `libjpeg` 开发库（Ubuntu：`libjpeg-dev`）
- 板端启用 RKNN 时需系统预装 `/lib/librknnrt.so`（`2.0.0b0`）

## 快速构建与测试

### 1. 板端构建（默认启用 RKNN）

```bash
# Debug 构建与测试
cmake --preset debug
cmake --build --preset debug
ctest --preset debug

# Release 构建与测试（测性能请使用 Release）
cmake --preset release
cmake --build --preset release
ctest --preset release
```

### 2. 无硬件环境构建（关闭 RKNN）

```bash
cmake -S . -B build/norknn -G Ninja -DCMAKE_BUILD_TYPE=Debug -DEDGEFLOW_WITH_RKNN=OFF
cmake --build build/norknn
ctest --test-dir build/norknn
```

## CLI 使用示例

```bash
# 查看版本与编译开关
./build/release/edgeflow version

# 输出板端设备与环境 JSON 诊断报告
./build/release/edgeflow doctor

# 单次图像分类推理
./build/release/edgeflow infer --input /usr/share/model/dog_224x224.jpg

# 预热 10 次 + 连续推理 100 次并输出延迟分布
./build/release/edgeflow infer \
  --model /usr/share/model/RK3576/mobilenet_v1.rknn \
  --input /usr/share/model/dog_224x224.jpg \
  --warmup 10 \
  --runs 100 \
  --topk 5
```

## 退出码约定

| 退出码 | 含义 |
|---:|---|
| `0` | 执行成功（`doctor` 关键设备与模型检查通过） |
| `1` | 设备不可访问、模型缺失/损坏或推理运行失败 |
| `2` | 命令行参数错误 |
