#!/bin/bash
# restore_env.sh — 在新申请的容器上零网络恢复课程环境（约 30 秒）。
#
# 背景：平台镜像回收后，容器层（/root、/opt、/）会被重置，只有挂载盘
# /workspace 保留。本脚本把预先备份在 /workspace/env_persist 的易失资产
# 恢复到标准位置，使 Jupyter Kernel、Ultralytics 设置与 MIOpen/COMGR
# JIT 编译缓存（热启动）全部立即可用，无需重装任何包。
#
# 用法（新容器首次进入时执行一次）：
#   bash /workspace/yolo26-rocm-day1-labs/scripts/restore_env.sh [--no-preflight]
#
# 前提（旧容器上已由部署脚本完成，只需做一次）：
#   - .venv 解释器软链指向 /workspace/env_persist/uv-python（自包含）
#   - /workspace/env_persist/backup/{kernels,ultralytics_config,miopen,comgr} 已备份
set -euo pipefail

PERSIST="${ENV_PERSIST_DIR:-/workspace/env_persist}"
REPO="${COURSE_REPO:-/workspace/yolo26-rocm-day1-labs}"
RUN_PREFLIGHT=1
[ "${1:-}" = "--no-preflight" ] && RUN_PREFLIGHT=0

fail() { echo "[FAIL] $1" >&2; exit 1; }

echo "== 恢复课程环境（来源：$PERSIST） =="

# 0. venv 自包含性检查（解释器必须位于持久盘）
[ -x "$REPO/.venv/bin/python" ] || fail ".venv 缺失：$REPO/.venv（请检查 /workspace 挂载）"
"$REPO/.venv/bin/python" - <<'EOF' || fail ".venv 解释器不可用（软链目标丢失？）"
import sys, torch
assert torch.version.hip is not None, "非 ROCm 构建"
print(f"[OK] venv 自包含：{sys.executable} · torch {torch.__version__} · HIP {torch.version.hip}")
EOF

# 1. Jupyter Kernel 注册（容器层的 ~/.local/share/jupyter 已被重置）
if [ -d "$PERSIST/backup/kernels" ]; then
    mkdir -p /root/.local/share/jupyter/kernels
    cp -a "$PERSIST/backup/kernels/." /root/.local/share/jupyter/kernels/
    echo "[OK] Kernel 已注册：yolo26-rocm7141"
else
    echo "[WARN] 未找到内核备份，改用即时注册"
    "$REPO/.venv/bin/python" -m ipykernel install --user \
        --name yolo26-rocm7141 --display-name "YOLO26 - AMD ROCm 7.14.1"
fi

# 2. Ultralytics 设置（datasets_dir=/workspace/datasets、sync=false 等）
if [ -d "$PERSIST/backup/ultralytics_config" ]; then
    mkdir -p /root/.config
    cp -a "$PERSIST/backup/ultralytics_config" /root/.config/Ultralytics
    echo "[OK] Ultralytics 设置已恢复（datasets_dir=/workspace/datasets）"
else
    "$REPO/.venv/bin/yolo" settings datasets_dir=/workspace/datasets sync=false >/dev/null 2>&1 \
        && echo "[OK] Ultralytics 设置已重建" || echo "[WARN] Ultralytics 设置重建失败（非致命）"
fi

# 3. MIOpen 查找库 + COMGR JIT 编译缓存（恢复热启动：省去首次训练 ~1 分钟编译）
#    用合并模式（-n 不覆盖已有文件）：幂等、可重复执行、与新生成的缓存共存。
for cache in miopen comgr; do
    if [ -d "$PERSIST/backup/$cache" ]; then
        mkdir -p "/root/.cache/$cache"
        cp -a -n "$PERSIST/backup/$cache/." "/root/.cache/$cache/"
        echo "[OK] $cache 缓存已合并（$(du -sh /root/.cache/$cache 2>/dev/null | cut -f1)，JIT 热启动）"
    fi
done

# 4. 课程资产完整性提示（权重/数据本就在持久盘）
for f in "$REPO/yolo26n.pt" /workspace/datasets/coco8/images; do
    [ -e "$f" ] && echo "[OK] 资产在位：$f" || echo "[WARN] 资产缺失：$f（见 03_CLASSROOM_CAPACITY_PLAN.md §6 预置链）"
done

# 5. 一键自检
if [ "$RUN_PREFLIGHT" = "1" ]; then
    echo
    echo "== 课前自检（preflight_classroom.py） =="
    "$REPO/.venv/bin/python" "$REPO/scripts/preflight_classroom.py" \
        --course-home "${COURSE_HOME:-$REPO/yolo_course}" \
        --json "$REPO/validation/linux_w7900/logs/preflight_restore.json" || \
        { echo "[FAIL] preflight 未全绿——按上方 next 提示处理后重试"; exit 1; }
fi

echo
echo "== 恢复完成：直接打开 Notebook（Kernel 选 YOLO26 - AMD ROCm 7.14.1）即可运行 =="
