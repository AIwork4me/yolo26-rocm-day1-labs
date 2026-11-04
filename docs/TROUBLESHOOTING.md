# 故障排查手册 — YOLO26 ROCm Day 1 Labs(Windows 原生 / Linux Radeon Cloud)

## 环境

| 症状 | 根因 | 处理 |
|---|---|---|
| `torch.version.hip is None` | Kernel 用了 CPU/CUDA 版 PyTorch | 切到 `yolo26-rocm7141` Kernel;绝不在本环境 `pip install torch` |
| `pip check` 报 torch 相关冲突 | 某依赖试图替换三件套 | `pip install --constraints constraints-rocm.txt ...`;被破坏则按 docs/ROCM_WINDOWS_SETUP.md §2 重装 |
| `torch.cuda.is_available()` False | 驱动过旧 / GPU 被独占 / 非 ROCm 构建 | 更新 Adrenalin;查占用;确认 `torch.version.hip != None` |
| GPU 型号 ≠ gfx1100 | 硬件不匹配 | 换用对应 `device-gfxXXXX` extra 重装;不要谎报通过 |
| `uv lock` 报 `symlink 'nul' 函数不正确` | uv + AMD 索引在 Windows 的兼容缺陷 | 用 constraints-rocm.txt + pip freeze 清单(本仓库已采用) |

## 训练(Lab C / E)

| 症状 | 根因 | 处理 |
|---|---|---|
| **首次训练/Conv/BN 卡住数分钟** | MIOpen 首次对每类 kernel 做 hipRTC 编译 + find 调优(写用户缓存) | 等待;第二次起复用缓存。CPU 占用持续增长即正常,不是死机 |
| `miopenStatusUnknownError` + hipRTC `'type_traits' file not found` | 旧 wheel 的 MIOpen 头依赖宿主 STL(ROCm/rocm-libraries#13437 已修) | 使用 ≥ rocm7.14.1 的修复版 wheel;不要降级 |
| 训练 NaN | AMP 在个别栈上的数值问题 | 课程默认 `amp=False`;若你改过,恢复默认并记录 |
| Windows 下 DataLoader 卡死/报 spawn 错误 | 多进程 worker 问题 | 课程统一 `workers=0`;不要为提速单方面改 A/B 任一组 |
| OOM | batch 过大 | 课堂方案可降 batch,但 **A/B 两组必须同改并记录**,否则实验无效 |
| 训练慢 | 首次 kernel 编译、缓存冷、GPU 被共享 | 查 MIOpen 缓存是否生成;确认 GPU 空闲;`imgsz=416` 组天然更快 |

## 数据(Lab B)

| 症状 | 根因 | 处理 |
|---|---|---|
| COCO128 下载超时 | GitHub 直链慢/被限 | 课前把 `coco128.zip` 放到 `yolo_course/data/raw/`(官方离线方案) |
| `EXACT IMAGE LEAKAGE` | 划分或拷贝导致跨 split 重复 | 这是 QA 正确报警;检查是否手工复制过图片 |
| 标签 `box extends outside image` | 标注坐标越界 | 用 Lab B 的备份机制修正;不要绕过 QA |
| 划分结果与同学不同 | 换了 seed 或源数据变化 | 课程固定 `seed=42`;自定义需声明 |

## 评测(Lab D / E)

| 症状 | 根因 | 处理 |
|---|---|---|
| 固定阈值 P/R 与官方 mAP 差异大 | 两者定义不同(教学匹配 ≠ COCO AP) | 正常现象,写进分析;不要"修正"任一方 |
| `best.pt` 缺失 | Lab C 未完成或 course_config 指向旧路径 | 顺序执行;勿手工伪造权重 |
| A/B 除 imgsz 外还有差异 | 复制参数时引入改动 | `changed == {'imgsz','project','name'}` 断言会拦下;回溯 baseline_train_settings |
| Test 成绩反复查看 | 违反 Test 封存原则 | `RUN_FINAL_TEST` 只在冻结方案后置 True 一次 |

## Jupyter / 路径

| 症状 | 根因 | 处理 |
|---|---|---|
| `FileNotFoundError: 请先将 ZIP 完整解压` | cwd 不在项目根/notebooks | 从项目根或 notebooks/ 启动;或设 `YOLO_COURSE_HOME` |
| 输出写到了意外位置 | `YOLO_COURSE_HOME` 已设置 | `print(home)` 确认;隔离测试用独立 home |
| Windows 提示 `\/dev/kfd 不存在` | 该节点是 Linux 概念 | Notebook 仅探测记录,Windows 为 False 属预期 |
| 图不显示(nbclient/无人值守) | matplotlib 后端 | 运行器已设 `MPLBACKEND=module://matplotlib_inline.backend_inline` |

## Radeon Cloud 实操（云端 GPU 实例）

| 症状 | 根因 | 处理 |
|---|---|---|
| `pip install` 在云端只有 ~50kB/s，装 jupyterlab(17MB) 跑了 6 分钟（**本测试环境观察，非平台普遍行为**） | 云端到 PyPI 官方源被限速（本环境观察） | 仅**普通 Python 依赖**可换阿里云镜像：`pip install -i https://mirrors.aliyun.com/pypi/simple/ ...`；AMD ROCm 的 PyTorch wheel 仍按固定指引走 `repo.amd.com`，勿混用；课程自带 `constraints-rocm.txt` 锁定 torch/torchvision/torchaudio 三件套，别让依赖漂移 |
| 解压课程 ZIP 报 `unzip: command not found` | 云端镜像没装 unzip | 用 `python3 -c "import zipfile; zipfile.ZipFile('yolo26.zip').extractall('.')"` 解压 |
| 练习 notebook 不填 `TODO` 也能直接运行（Lab 00/01/03/04/05），但 `filled` 字段会记录「未填」 | 当前是**软练习门禁**：cell 默认可跑、不阻断，完成情况通过 `filled` 写入 `reports/*.json`，讲师据此看是否真动手（对齐 `build_notebooks.py`，旧的阻塞式行为已弃用） | 学生无需任何额外操作；`YOLO_PRACTICE_FILL` 是**无人值守复验通道（学生不需要）**，它接受「一个 JSON 文件的路径」而非内联内容，用于讲师/自动化注入已填写样例：`python scripts/run_all_labs.py --practice-fill /path/to/practice.json` |
| 实例被删后 `/workspace` 外文件丢失；重 Launch 后 SSH 连不上（**本测试环境观察，非平台普遍保证**） | 本测试实例约 72h 自动删除，`/workspace` 是持久卷但实例 IP/Port 会刷新 | 重要产物（权重/报告）放 `/workspace`；本地 `~/.ssh/config` 的 `HostName`/`Port` 在重 Launch 后需同步更新 |
