# GGGS 基线管道清理验证（2026-11-09）

目标机器：`root@UKD-4090`，代码：`/workspace/Geometry-Grounded-Gaussian-Splatting`。
清理基于 `0c9a4ac894faedd197e51f457a51326311953cb6`，分支为
`codex/gggs-baseline-cleanup-20261109`。本轮没有加入 Wrapping、改 CUDA 渲染公式，
也没有启动完整 245 相机 / 20k 基线训练。

## 代码范围

`train.py` 从 1124 行整理为 476 行；主训练函数从 689 行降至 190 行。
移除临时 pseudo/R2C 路径、未调用实验代码和失效参数；损失、日志/保存分到小模块。
保留现有 RGBD 初始化、VCD/VCP、FastGS 日程、相机校准与输入内容的续训校验。
默认有效外部 normal 权重仍为 0，NCC 仍为 .02。具体阅读入口和命令见
[训练说明](training_baseline.md)。

## 环境与数据

- Python 3.12.11、Torch 2.9.1+cu128、torchvision 0.24.1+cu128、NumPy 2.1.2；
  后三者未因补依赖而升级。
- Open3D 固定为用户确认的 **0.19.0**；补齐缺失 Python/系统运行库，编译仓库的
  `warp_patch_ncc`。环境差异和完整日志保存在验证目录。
- 当前机器的 Open3D 0.20 最小常量平面 TSDF 输出为空；同参数在 0.19 输出
  12288 点和 24130 三角面。float32 / uint16 两种深度均复现。此结论只描述这次
  环境对照，不泛化到其他机器，也没有为绕过失败而修改融合算法或阈值。
- 原始数据 245 份相机/RGB/depth/normal 对应完整。短训只抽取 frame 0/8/16/24，
  独立保存 COLMAP 元数据，图像与 prior 用只读软链接引用原数据。

## 验证结果

1. **30 项 CPU 测试通过**：损失值/解析梯度、有效域/分母、权重日程、光度兼容、
   depth 路径与缓存身份、同数据 checkpoint、实际生命周期和累积梯度。
2. 修改的 Python 文件通过 Ruff/语法检查，启动脚本通过 `bash -n`。
3. 4090 光度数值检查：兼容后端对照显式 valid-SSIM 卷积与 L1，损失绝对差
   `8.94e-8`、梯度最大绝对差 `1.22e-9`，均有限。
4. **旧版和候选各 12 iteration 的真实 CUDA 对照完成**：相同随机种子、初始化
   点云逐字节相同，均 12 次真实视角渲染、11 次主优化器更新、0 次 SH 更新。
   增密在第 4/8 步发生，点数均为 `166335 → 208137 → 248550`。depth prior、
   depth-normal 自一致性、PatchMatch、VCD/VCP、评估与保存均经过实际执行。
5. 固定四视角比较最终 PLY：RGB 平均绝对差 `3.71e-8`，camera-Z 深度平均绝对差
   `7.07e-9`（最大 `2.35e-4`），normal 分量平均绝对差 `2.24e-8`。
6. **从候选 12 步 checkpoint 续到 17 步完成**，额外 5 次真实视角、4 次主更新、
   1 次 SH 更新；显式开启 normal prior=.05、同时关闭 NCC/geo，验证新参数确实生效。
   这只是接口测试，不更改正式基线默认权重。
7. 最终远程工作目录再次通过全部 30 项单测；启动脚本使用
   `global_depth/depth_npy` 路径完成 5 步 smoke、4 次实际主更新和最终 checkpoint。
   启动脚本命名为 `scripts/run_245_baseline.sh`，避免仓库原有 `train*.sh` 忽略规则。

旧版 reference 也使用相同的 `photometric.py` 兼容层，因当前官方 fused-ssim 不含
历史代码调用的额外 `photometric_loss` API。对照不包含那个不可用的自定义 kernel。

**不声称 checkpoint 位级等价：** 严格 tensor allclose 检查未通过，少数近零梯度
对应参数存在更大的绝对差。旧版自身以同 seed 再跑也出现这种波动；其 xyz 最大差
`2.13e-4`、opacity raw 最大差 `.0489`，而旧/新对应为 `1.01e-5` / `.00429`。
两套 Adam 原本使用 `eps=1e-15`；本轮未改 epsilon 或放宽阈值来伪装严格通过。
行为契约、更新/增密统计、实际渲染差共同支持本次重构，仍不等于完整场景质量验收。

## 可追溯产物

远程目录：`/workspace/gggs_cleanup_validation_20261109/`。

- `pre_cleanup_tracked.tar`：修改前的相关 tracked 源码。
- `reference/` / `candidate/`：验证用源码。
- `run_parity.sh`、`reference_compatibility.txt`：对照命令与兼容边界。
- `unit_tests.log`、`photometric_gpu_check.json`。
- `training_parity.json`：保留严格 checkpoint allclose 为 false 的原始检查。
- `cuda_repeatability.json`、`render_parity.json`：旧版复跑与实际渲染差。
- `reference_o319/`、`candidate_o319/`、`resume_controls/`：短训模型与配置。
- `environment_audit.json` 与 pip/conda/dpkg 前后差异、Open3D 最小复现。

这些短训采用 320 像素宽和提前触发的阶段，目的是覆盖执行分支；不用于评价最终
mesh 质量，不与完整 245 帧基线混用。
