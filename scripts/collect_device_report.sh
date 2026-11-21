#!/usr/bin/env bash
# 环境检测, 并写入 ../configs/device-report.json
set -euo pipefail

OUT_PATH="${1:-$(cd "$(dirname "$0")/.." && pwd)/configs/device-report.json}"
mkdir -p "$(dirname "$OUT_PATH")"

python3 - "$OUT_PATH" <<'PY'
import glob
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

out_path = sys.argv[1]

def run_cmd(cmd):
    """ 执行 linux 命令 """
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def sha256_file(path):
    """ 计算文件 sha256 """
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def dev_info(path):
    """ 检查 Linux 设备文件的存在性和访问权限 """
    exists = os.path.exists(path)
    return {
        "exists": exists,
        "readable": os.access(path, os.R_OK) if exists else False,
        "writable": os.access(path, os.W_OK) if exists else False,
    }

def read_first_line(path):
    """ 读取文件第一行 """
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.readline().strip()
    except Exception:
        return None

# Linux 通常使用 /etc/os-release 描述发行版信息
# 获取 CPU, 内存, 系统和内核信息
os_release = {}
if os.path.isfile("/etc/os-release"):
    with open("/etc/os-release", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                os_release[k] = v.strip('"')

# /proc/meminfo 是 Linux 内核提供的内存统计接口
# 检测内存容量
mem_total_kb = None
if os.path.isfile("/proc/meminfo"):
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
                break

rknn_lib = "/lib/librknnrt.so"
rknn_ver = None
if os.path.isfile(rknn_lib):
    out = run_cmd(["strings", rknn_lib])
    if out:
        for line in out.splitlines():
            if "librknnrt version:" in line:
                rknn_ver = line.split("librknnrt version:", 1)[1].strip()
                break

# 获取工具的版本
tools = {}
for name in ["gcc", "g++", "cmake", "ninja", "make", "pkg-config", "git", "gdb", "strace", "clang-format", "clang-tidy", "v4l2-ctl", "gst-launch-1.0"]:
    path = shutil.which(name)
    if path:
        ver = run_cmd([name, "--version"])
        tools[name] = ver.splitlines()[0] if ver else path
    else:
        tools[name] = None

thermal_zones = {}
for tz in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
    t = read_first_line(os.path.join(tz, "type"))
    if t:
        thermal_zones[os.path.basename(tz)] = t

report = {
    "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "system": {
        "os": os_release.get("PRETTY_NAME"),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count(),
        "mem_total_mib": round(mem_total_kb / 1024, 1) if mem_total_kb else None,
        # 检查内核配置
        "proc_config_gz": os.path.isfile("/proc/config.gz"),
        # 检查编译目录
        "kernel_build_tree": os.path.isdir(f"/lib/modules/{platform.release()}/build"),
    },
    "devices": {
        "/dev/dri/renderD129": dev_info("/dev/dri/renderD129"),
        "/dev/rga": dev_info("/dev/rga"),
        "/dev/mpp_service": dev_info("/dev/mpp_service"),
        "/dev/dma_heap/system": dev_info("/dev/dma_heap/system"),
        "gpiochips": sorted(glob.glob("/dev/gpiochip*")),
        "video_node_count": len(glob.glob("/dev/video*")),
        "media_node_count": len(glob.glob("/dev/media*")),
    },
    "telemetry": {
        "thermal_zones": thermal_zones,
        "devfreq": sorted(os.listdir("/sys/class/devfreq")) if os.path.isdir("/sys/class/devfreq") else [],
    },
    "sdks": {
        "rknn": {
            "library": rknn_lib if os.path.isfile(rknn_lib) else None,
            "version": rknn_ver,
            "sha256": sha256_file(rknn_lib),
            "header_local": os.path.isfile(os.path.join(os.path.dirname(out_path), "..", "third_party/rknn/include/rknn_api.h")),
        },
        "rga": {
            "library": "/usr/lib/aarch64-linux-gnu/librga.so.2.1.0" if os.path.isfile("/usr/lib/aarch64-linux-gnu/librga.so.2.1.0") else None,
            "header": "/usr/include/rga/im2d.h" if os.path.isfile("/usr/include/rga/im2d.h") else None,
        },
        "mpp": {
            "library": "/lib/aarch64-linux-gnu/librockchip_mpp.so.1" if os.path.isfile("/lib/aarch64-linux-gnu/librockchip_mpp.so.1") else None,
            "header": "/usr/include/rockchip/rk_mpi.h" if os.path.isfile("/usr/include/rockchip/rk_mpi.h") else None,
        },
    },
    "models": {
        "mobilenet_v1_rk3576": {
            "path": "/usr/share/model/RK3576/mobilenet_v1.rknn",
            "exists": os.path.isfile("/usr/share/model/RK3576/mobilenet_v1.rknn"),
            "sha256": sha256_file("/usr/share/model/RK3576/mobilenet_v1.rknn"),
        }
    },
    "toolchain": tools,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
    f.write("\n")

print(f"Wrote device report to {out_path}")
PY
