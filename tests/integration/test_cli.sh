#!/bin/bash

set -eu
# 1. 检查是否传入 edgeflow 可执行文件路径
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <edgeflow_binary>" >&2
    exit 1
fi

EDGEFLOW="$1"

if [ ! -x "$EDGEFLOW" ]; then
    echo "[FAIL] edgeflow binary is not executable: $EDGEFLOW" >&2
    exit 1
fi

failures=0

# 创建临时目录, 确保用于测试的模型文件不存在
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

# 2. 通用退出码检查函数
check_exit() {
    expected="$1"
    shift

    # 执行 edgeflow 并获取实际退出码
    if output=$("$EDGEFLOW" "$@" 2>&1); then
        actual=0
    else
        actual=$?
    fi

    # 比较实际退出码和预期退出码
    if [ "$actual" -eq "$expected" ]; then
        echo "[PASS] edgeflow $* -> exit $actual"
    else
        echo "[FAIL] edgeflow $*" >&2
        echo "       expected: $expected" >&2
        echo "       actual:   $actual" >&2
        echo "$output" >&2

        failures=$((failures + 1))
    fi
}

# 3. 执行 CLI 集成测试

# 帮助命令
check_exit 0 --help

# 未知子命令
check_exit 2 unknown

# 缺少必选参数
check_exit 2 infer

# 非法推理次数
check_exit 2 infer --input a.jpg --runs 0

# 模型文件不存在
check_exit 1 doctor --model "$tmp_dir/not_exist.rknn"

# 4. 汇总测试结果
if [ "$failures" -eq 0 ]; then
    echo "[PASS] all CLI integration tests passed"
    exit 0
fi

echo "[FAIL] $failures CLI integration test(s) failed" >&2
exit 1