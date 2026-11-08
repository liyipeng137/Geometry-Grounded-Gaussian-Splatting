# GGGS 真实相机基线训练

这次整理以 `dev/gggs-fast-scene` 的 `0c9a4ac` 为起点，只保留真实相机训练。
目标是把当前 GGGS 跑成后续对照基线；没有加入 Gaussian Wrapping 或改变渲染公式。

## 阅读入口

- `train.py::main`：解析、检查参数，设置随机种子。
- `train.py::training`：初始化、单相机训练循环与明确的阶段顺序。
- `train.py::update_topology`：现有 VCD/VCP、opacity reset、3D filter 日程。
- `utils/training_losses.py`：启用时机、损失原始值和加权贡献。
- `utils/training_io.py`：实际配置、JSONL、评估、checkpoint 与同数据续训校验。
- `utils/photometric.py`：兼容官方与自定义 fused-ssim 包。

## 每轮执行顺序

1. 更新学习率，按旧日程解锁 SH/SG。
2. 从真实训练相机中无放回随机选一个；全部遍历完再开始下一轮。
3. 渲染 RGB；当几何损失需要时同时计算 median depth / normal。
4. 计算 RGB、depth/normal、mask 和相邻真实相机的 PatchMatch 损失。
5. `backward()`。
6. 记录损失，按配置评估并导出 Gaussian PLY。
7. 累积增密统计，按原日程增密、剪枝、opacity reset 和更新 3D filter。
8. 调用模型的 `optimizer_step()`；按配置保存 checkpoint。

**保留的旧行为：** PLY 在该步 optimizer 前保存，checkpoint 在 optimizer 后保存；
最后一个 iteration 不执行 optimizer step。主循环不额外 `zero_grad()`，避免破坏
FastGS 的梯度累积。默认前 15k 主参数每步更新，15k～20k 每 32 步更新，20k 后
每 64 步更新；SH 前期每 16 步更新。实际更新次数记录在 `training_stats.json`。
`--dense_optimizer_updates` 仍是显式可选配方，不在本次清理中默认开启。

## 有效损失默认值

| 项目 | 默认权重 | 启用与含义 |
|---|---:|---|
| RGB | 1 | `(1−lambda_dssim)×L1 + lambda_dssim×(1−SSIM)`，`lambda_dssim=.2` |
| Depth prior | .4 | 从 3000 步开启，默认绝对 camera-Z L1；支持原有 relative Charbonnier |
| Depth–normal 自一致性 | .05 | 从 7000 步开启；无效像素置零后按全图平均 |
| 外部 normal prior | **0** | 默认关闭，与清理前实际行为一致；显式设置非零时从 `normal_prior_from_iter` 开启 |
| 多视角 NCC | **.02** | 从 7000 步开启 |
| 多视角几何 | .02 | 从 7000 步开启 |
| Mask | 0 | 现有背景约束；不把 depth-valid mask 解释为物体轮廓 |

旧参数文件将 normal/NCC 写成 `.1/.6`，但循环强制使用 `0/.02`。现在默认值改成
原来**实际生效**的 `0/.02`，命令行参数会真正生效。旧命令若显式传 `.1/.6`，
需要删除或改回 `0/.02` 才能重现原有效配方。normal prior 仍使用无向 cosine，
没有在清理中改为有向法线监督。相对 depth 不拟合新的 scale/shift。

## 数据与初始化

```text
245_for_train/
├── images/
├── sparse/0/
├── normals/
├── global_depth/depth_npy/
├── global_depth/valid_mask/
└── tsdf_mesh.ply
```

`--depth_prior_dir global_depth/depth_npy` 现在同时控制训练监督与 RGBD 初始化。
支持绝对目录，文件按完整 COLMAP `image.name` 匹配，并将扩展名换成 `.npy`。
仍保留已有世界尺度、完整 K 和半整数像素中心。

初始化继续使用训练相机的 RGBD→TSDF 融合点云（约 20 万采样点、voxel `.02`、
trunc `.06`）；**没有读取 `tsdf_mesh.ply` 的 faces，也没有切换为 mesh warm start**。
指定 depth 来源进入初始化缓存指纹。改变路径或融合代码会重新生成缓存。

当前 depth 加载器以有限正深度作为有效性，本次没有新增独立 valid_mask 加权。
给定样例抽查的 depth-valid mask 全有效且与正深度一致；它不能传给 `--mask_dir`
冒充前景分割。若后续数据有非全有效 mask，应另行定义监督与初始化共同的掩码契约。

## UKD-4090 上的基线命令

环境使用已有 `py312`（Python 3.12 / Torch 2.9.1+cu128），Open3D 固定为
`0.19.0`。本次环境中 Open3D 0.20 的最小平面 TSDF 融合返回空体积，因此不使用它。

```bash
cd /workspace/Geometry-Grounded-Gaussian-Splatting
bash scripts/run_245_baseline.sh /workspace/outputs/gggs_245_base
```

脚本使用 `/usr/local/miniconda3/envs/py312/bin/python`、全部 245 个真实相机、
默认 20k iterations，显式保持 normal=0 / NCC=.02。图像分辨率沿用原 CLI 默认；
`frame_gpu_budget=4` 只限制图像/prior 的 GPU 驻留数，不减少训练相机。
可以在输出目录后追加正常训练参数，例如 `--iterations 200 --max_image_width 480`。
这种短训仅用于运行验证，不是完整基线结果。

脚本不安装包、不下载数据、不修改输入目录。输出必须使用新目录，避免覆盖既有运行。
没有加 `--eval`，所以全部相机参与训练；训练视角抽查的 PSNR 不代表独立 mesh 验收。

## 输出与续训

- `training_config.json`：实际 loss 权重、数据路径、优化日程、光度后端及完整参数。
- `loss.jsonl`：按 `--log_interval` 记录子损失、加权贡献、总损失、相机和点数。
- `training_stats.json`：实际 main/SH 更新数、增密/剪枝历史、结束状态。
- `evaluation_*.json`：指定迭代的图像评估。
- `point_cloud/iteration_*/`：现有 Gaussian PLY 导出格式。
- `chkpnt*.pth` / `.state.json`：模型、优化器及嵌入的数据校验证明；最后一轮自动保存。

续训用 `--start_checkpoint ...`，输出另选新目录；仍核对相机划分、K/pose、输入
RGB/depth 内容和初始化来源。仅接受 `gggs_real_baseline` 配方，旧 R2C checkpoint
明确拒绝。现有恢复机制不保存相机抽样栈、RNG 或未应用的累积梯度，因此是同数据
继续优化，不保证与未中断训练逐步等价。

## 移除的实验内容与依赖兼容

已从训练入口移除 pseudo manifest、替换/叠加伪视角、伪深度与增密、临时梯度诊断、
R2C 配方绑定、未调用的 Pearson 损失/贡献剪枝和未启动的 GUI 热循环。
对应旧 CLI 参数会报未知参数，避免静默忽略。

官方 `fused_ssim` 包只有 `fused_ssim`，历史训练代码却调用了额外的
`photometric_loss`。兼容层优先使用存在的自定义入口，否则恢复显式 L1+
`fused_ssim(padding="valid")`；uint8 RGB 转为 `[0,1]`。外观模型仍使用校正后
图像计算 L1、原 render 计算 SSIM。数学定义沿用历史实现，但不同 CUDA 归约的
浮点结果和速度不能宣称逐位相同；实际后端写入运行配置。

## 验证

在已有 Torch 环境中运行：

```bash
python -m pytest -q tests
python -m compileall -q train.py arguments scene utils
```

CPU 单测覆盖损失值/梯度、权重与阶段门控、depth 来源及缓存身份；真实 CUDA、
初始化、checkpoint 和提面依赖仍需在目标机器上验证。验证记录见同次清理报告。
