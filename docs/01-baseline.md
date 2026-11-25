# 01 - RK3576 硬件与 MobileNetV1 推理基线（v0.1.0）

本文档记录 EdgeFlow `v0.1.0` 在 KickPi RK3576 开发板上的软硬件环境快照、精度对齐结果与初始推理性能基线。

## 1. 测试环境

| 项目 | 实测信息 |
|---|---|
| 开发板 / 架构 | KickPi RK3576 (`aarch64`, 8 核 CPU) |
| 系统与内核 | Ubuntu 24.04.3 LTS, Linux `6.1.75` |
| 物理内存 | `3897.9 MiB` |
| RKNN Runtime | `/lib/librknnrt.so` `2.0.0b0 (35a6907d79@2024-03-24T10:31:14)` |
| Runtime SHA-256 | `3f465203344bf5a89f674dae7491b9092340b289bd4e781e78304ac2e123718b` |
| RKNPU 内核驱动 | `v0.9.7`（节点 `/dev/dri/renderD129` 普通用户读写正常） |
| 编译器与配置 | `g++ 13.3.0`, CMake `Release` 预设 (`-O3`), 单上下文默认核策略 |
| 板端常温状态 | 各温区约 `37.0 ~ 37.9 °C`，空闲 NPU 频率 `300 MHz` |

完整设备盘点见 `configs/device-report.json`。

## 2. 模型与张量属性

- **模型路径**：`/usr/share/model/RK3576/mobilenet_v1.rknn`
- **模型 SHA-256**：`bc66943ea85ec0dd8a04da22c4276bfc8a4c6fe24f5ea8be7a1e5c3c22c8259d`

| 方向 | 名称 | 维度 (`dims`) | 布局 | 类型 | 量化参数 | 字节数 |
|---|---|---|---|---|---|---:|
| Input 0 | `input` | `[1, 224, 224, 3]` | `NHWC` | `INT8` | `AFFINE_ASYMMETRIC` (`zp=0, scale=0.0078125`) | `150528` |
| Output 0 | `MobilenetV1/Predictions/Reshape_1` | `[1, 1001]` | `UNDEFINED` | `FLOAT16` | `AFFINE_ASYMMETRIC` (`zp=0, scale=1`) | `2002` |

## 3. 精度对齐验证

对比官方测试工具 `rknn_common_test` 与 `edgeflow infer`，两张标准测试图的 Top-5 类别编号与置信度 **100% 逐位一致**：

| 测试样本 | Top-1 | Top-2 | Top-3 | Top-4 | Top-5 | 与 `rknn_common_test` 对比 |
|---|---|---|---|---|---|---|
| `dog_224x224.jpg` | `156` (`0.935059`) | `155` (`0.057037`) | `205` (`0.003881`) | `284` (`0.003119`) | `285` (`0.000172`) | 完全一致 |
| `cat_224x224.jpg` | `283` (`0.415283`) | `282` (`0.175781`) | `286` (`0.158813`) | `278` (`0.060303`) | `279` (`0.043427`) | 完全一致 |

## 4. 推理延迟基线

复现命令：

```bash
./build/release/edgeflow infer \
  --model /usr/share/model/RK3576/mobilenet_v1.rknn \
  --input /usr/share/model/dog_224x224.jpg \
  --warmup 10 \
  --runs 100 \
  --topk 5
```

实测结果（模型加载耗时约 `29.0 ms`）：

| 测试模式 | Warmup | Runs | Min (ms) | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) | 估算 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 冷启动单次 | 0 | 1 | - | 7.489 | - | - | - | - | 133.5 |
| 稳态连续推理 | 10 | 100 | 4.571 | 5.202 | 5.095 | 6.042 | 6.091 | 6.141 | 192.2 |

> 说明：冷启动首帧因 NPU 处于空闲低频（`300 MHz`）且缓存未热，耗时约 `7.5 ms`；经过 10 次预热触发 `devfreq` 升频后，稳态 P50 降至 `5.095 ms`，P99 控制在 `6.091 ms` 以内。

## 5. 当前边界与后续计划

1. 当前延迟统计范围仅包含 `RknnContext::RunRgbUint8`（输入设置 + NPU 执行 + 输出读取），不含外部 JPEG 解码耗时。
2. 当前要求输入图片分辨率与模型输入严格一致（`224x224`），任意尺寸缩放与 RGA 硬件加速将在 P2 阶段接入并对比。
3. P1 阶段将补齐模型 SHA-256 校验、10,000 次长稳内存/FD 监控以及目标检测模型兼容性验证。
