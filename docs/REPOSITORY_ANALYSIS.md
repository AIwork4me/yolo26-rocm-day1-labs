# YOLO26 ROCm Day 1 Labs：仓库分析

## 仓库概览

本仓库面向大学生提供一天入门级 AMD GPU 目标检测实训，采用中英双语 Notebook，覆盖 ROCm 环境检查、推理与阈值、数据集 QA、GPU 训练、mAP 评价与受控 A/B 实验。

## 仓库结构

| 路径 | 作用 |
| --- | --- |
| `notebooks/` | 6 份中英双语课程 Notebook，由 `build_notebooks.py` 生成 |
| `course/lab_utils.py`、`course/annotation_import.py` | 数据、QA、评测和路径共享逻辑；X-AnyLabeling 导入器 |
| `scripts/` | GPU 冒烟检查、全课程运行、课堂课前自检（preflight）、环境持久化/恢复（persist_env/restore_env）、平台验证及 MIOpen shim 安装 |
| `tests/` | 标签、IoU、数据泄漏、A/B 纯度、ROCm 门禁、Notebook 结构/生成器一致性等离线回归 |
| `.github/workflows/ci.yml` | 双平台（ubuntu/windows）、无 torch 的离线 CI |
| `docs/` | Windows/Linux 部署及故障排查材料 |
| `validation/` | GPU 验证报告、执行矩阵、执行后的 Notebook 与日志 |

## 六个实验

1. 00 环境预检：检查 ROCm/HIP、GPU 冒烟测试并冻结权重。
2. 01 Lab A：推理与置信度阈值，观察边界框、IoU 和预测结果。
3. 02 Lab B：YOLO 标注、数据 QA 与防泄漏划分。
4. 03 Lab C：COCO8 冒烟及 12 Epoch 微调，生成本地权重和训练曲线。
5. 04 Lab D：比较官方 mAP 与教学固定阈值匹配，分析失败案例。
6. 05 Lab E：只改变 imgsz 的受控 A/B 对照实验。

课程另提供 X-AnyLabeling 导出数据导入路径（含场景分组防泄漏与坏数据门禁）。README 特别指出，学生自带数据的完整训练链路仍需真机走通，离线检查不能称为 GPU 验证通过；仓库内的受控 Fixture 测试结果已明确标注，不冒充真实学生导出验收。

## 教学交互与环境工程特性（2026-11 发布版）

- **软门禁练习设计**：六份 Notebook 各含一个学生主动操作练习格（预测/读曲线/错误归因/假设先行）。默认占位值可直接运行（Run-All 全链绿），输出真实结果并温和邀请学生替换 TODO 再跑；填写状态以 `filled` 字段落盘供讲师审计。安全门禁（ROCm/数据泄漏/权重指纹/A-B 纯度）保持硬失败，不因练习软化。
- **Kernel 自选**：每份 Notebook 顶部有内核选择提示，且元数据钉住 `yolo26-rocm7141`，Jupyter 打开即预选课程内核。
- **环境持久化**：`scripts/persist_env.sh`（部署一次）把解释器与 JIT 编译缓存安置到持久盘 `/workspace/env_persist`；镜像回收后新容器只需 `scripts/restore_env.sh`（约 30 秒、零网络）即可直接运行。存储地图见 `docs/RADEON_CLOUD_LINUX_SETUP.md` §7。
- **课前自检**：`scripts/preflight_classroom.py` 12 项只读检查（版本/HIP/GPU/NMS/权重 SHA256/数据/内核/解释器持久化/磁盘/冷热状态）。
- **CI**：`.github/workflows/ci.yml` 在 ubuntu 与 windows 双平台运行无 torch 离线回归套件，守卫测试确保 CI 永不安装 torch、必跑 pytest。

## 平台和验证范围

仓库 README 报告两轮真机验证，均锁定 **torch 2.12.0+rocm7.14.1** 三件套（wheel 来自 repo.amd.com）：

- **Windows 11 原生**：Radeon RX 7900 XTX（gfx1100），运行时 Python 3.12.16；
- **Linux（AMD Radeon Cloud）**：Radeon Pro W7900D（gfx1100、48 GB）、Ubuntu 24.04、uv 隔离环境（Python 3.12.13）。

公开验证材料报告 6/6 Notebook 通过，矩阵为 41/41 code cells，包含两次完整 12-Epoch 训练与受控 A/B 实验（Linux 轮经四次独立复现，指标确定性一致）。具体运行轮次和平台以 `validation/` 中的证据文件与 SHA256SUMS 为准。

以上为仓库公开记录与验证工件。不可据此推定其他 GPU、系统或依赖组合已验证；README 摘要与验证报告的运行轮次和计数口径可能不同，引用数字时应标明对应证据。

Windows 部署文档记录 ROCm 7.14.1 wheel 的 MIOpen/hipRTC 编译问题，并提供 scripts/install_miopen_shims.py 解决路径。相关说明见 docs/ROCM_WINDOWS_SETUP.md 和 docs/TROUBLESHOOTING.md。

## 数据与实验诚信

- 教学数据为 COCO128 的 person/car/chair 三类重划分，按图片隔离 train/val/test；不能描述为工业数据或 COCO 榜单成绩。
- Lab D 固定阈值 TP/FP/FN 匹配用于教学，不等同于 COCO mAP 算法。
- 不得混分训练集与测试集、用测试集调参、伪造训练产物或预置性能提升结论。
- 数据集与训练权重（如 *.pt）不应提交；第三方模型、数据和图片资产须遵守各自许可。

## 贡献与复现

贡献指南要求从 main 创建 feat/<topic> 或 fix/<topic> 分支。提交 PR 前运行（Linux 形式；Windows 见 CONTRIBUTING.md）：

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
```

涉及 Notebook 或 build_notebooks.py 时，运行生成器：

```powershell
.\.venv\Scripts\python.exe build_notebooks.py
```

Notebook 由 build_notebooks.py 生成，应修改生成器而非手改 .ipynb，并保持 Markdown 中英对照。PR 应填写现象、根因和验证证据并关联 Issue；不得移除 ROCm 门禁、数据泄漏检查或 A/B 纯度断言。

## 本文档验证

内容依据公开 README.md、CONTRIBUTING.md、docs/ 与 validation/ 目录整理，并经维护者核对版本口径与发布特性（2026-11-03）。本文档本身未执行仓库测试、Notebook 或 ROCm GPU 训练；所引用的验证结论以 `validation/` 工件为准。
