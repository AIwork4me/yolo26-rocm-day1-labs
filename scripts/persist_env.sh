#!/bin/bash
# persist_env.sh — 一次性部署：把课程环境的易失资产安置到持久盘 /workspace。
#
# 背景：容器层（/、/root、/opt）在镜像回收后重置，仅挂载盘 /workspace 保留。
# 本脚本在【旧容器/刚部署好的容器】上执行一次，之后任何新容器只需运行
# scripts/restore_env.sh（约 30 秒、零网络）即可直接运行 Notebook。
#
# 用法：
#   bash scripts/persist_env.sh
set -euo pipefail

PERSIST="${ENV_PERSIST_DIR:-/workspace/env_persist}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
UV_PYTHON_SRC="${UV_PYTHON_SRC:-/root/.local/share/uv/python}"

fail() { echo "[FAIL] $1" >&2; exit 1; }
[ -d "$REPO/.venv" ] || fail "未找到 .venv：$REPO/.venv（先按部署文档 §2B 建好环境）"

mkdir -p "$PERSIST/backup"

echo "== 1/4 Python 解释器持久化（venv 自包含的关键） =="
if [ ! -d "$UV_PYTHON_SRC" ]; then
    echo "[SKIP] $UV_PYTHON_SRC 不存在（venv 可能已指向其他解释器）"
    PY_REAL=$("$REPO/.venv/bin/python" -c "import os,sys; print(os.path.realpath(sys.executable))")
    echo "       当前解释器实际路径：$PY_REAL"
else
    cp -a "$UV_PYTHON_SRC" "$PERSIST/uv-python"
    # uv 的"版本别名目录"是指向 /root 的绝对软链，cp -a 会原样保留 → 改为相对链接
    for link in "$PERSIST"/uv-python/cpython-*; do
        [ -L "$link" ] || continue
        target=$(readlink "$link")
        case "$target" in
            /*) ln -sfn "$(basename "$target")" "$link"
                echo "[OK] 绝对软链已改为相对：$link -> $(basename "$target")" ;;
        esac
    done
    # 修正 venv：bin/python 软链 + pyvenv.cfg 的 home
    VERSION_DIR=$(basename "$(find "$PERSIST/uv-python" -maxdepth 1 -type d -name 'cpython-3*' | head -1)")
    ln -sfn "$PERSIST/uv-python/$VERSION_DIR/bin/python3.12" "$REPO/.venv/bin/python"
    sed -i "s|^home = .*|home = $PERSIST/uv-python/$VERSION_DIR/bin|" "$REPO/.venv/pyvenv.cfg"
    # 校验无残留绝对链接
    if find "$PERSIST/uv-python" -type l -exec sh -c 'readlink "$1" | grep -q "^/root"' _ {} \; | grep -q .; then
        fail "uv-python 副本中仍存在指向 /root 的软链"
    fi
    "$REPO/.venv/bin/python" -c "import torch; assert torch.version.hip" 2>/dev/null \
        && echo "[OK] venv 已自包含（解释器位于 $PERSIST）" \
        || fail "venv 修复后 torch 不可用——请检查上方输出"
fi

echo "== 2/4 备份 Jupyter Kernel 注册 =="
mkdir -p "$PERSIST/backup/kernels"
cp -a /root/.local/share/jupyter/kernels/. "$PERSIST/backup/kernels/" 2>/dev/null \
    && echo "[OK] kernels（yolo26-rocm7141）" || echo "[WARN] 未找到内核注册（先运行 ipykernel install）"

echo "== 3/4 备份 Ultralytics 设置 =="
cp -a /root/.config/Ultralytics "$PERSIST/backup/ultralytics_config" 2>/dev/null \
    && echo "[OK] Ultralytics settings（datasets_dir=/workspace/datasets）" \
    || echo "[WARN] 未找到 /root/.config/Ultralytics（非致命）"

echo "== 4/4 备份 JIT 编译缓存（MIOpen 查找库 + COMGR 编译缓存） =="
for cache in miopen comgr; do
    if [ -d "/root/.cache/$cache" ]; then
        rm -rf "$PERSIST/backup/$cache"
        cp -a "/root/.cache/$cache" "$PERSIST/backup/$cache"
        echo "[OK] $cache（$(du -sh "$PERSIST/backup/$cache" | cut -f1)）"
    fi
done

echo
echo "== 完成。持久盘占用：$(du -sh "$PERSIST" | cut -f1) =="
echo "新容器恢复命令：bash $REPO/scripts/restore_env.sh"
