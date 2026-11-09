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
| 外部 normal prior | **.1 → .15 → .2 → .1** | 默认从第 1 步开启，采用 `train_demo.py` 的有向 cosine 和下述日程 |
| 多视角 NCC | **.2 → .1 → .05** | 从 7001 步按 demo 日程启用，15k 后保持 .05 |
| 多视角几何 | .02 | 从 7000 步开启 |
| Mask | 0 | 现有背景约束；不把 depth-valid mask 解释为物体轮廓 |

normal prior 现完整采用用户提供的生产 `GGGS/train_demo.py` 逻辑，默认 20k 日程为：

| iteration | 当前权重 |
|---|---:|
| 1–3000 | .1 |
| 3001–5000 | .15 |
| 5001–11000 | .2 |
| 11001–15000 | 按 geometry 阶段后半程线性降至 .1 |
| 15001–20000 | 保持 .1，避免源 demo 的越界负权重 |

geometry 进度为 `(iteration−7001)/(15000−7001)`，前半程保持 .2。
少于 15k 的短训沿用 demo：geometry 终点截为总 iterations，init 仍使用 3000/5000
的绝对阈值。`lambda_normal_prior` 是初始/尾段基准权重（默认 .1），其余阶段使用
1.5 倍、2 倍和线性回落；传 0 可以关闭。`normal_prior_from_iter` 默认 1，仍可显式延迟。

损失为 `1−cos(render_normal, prior) + 1−cos(depth_normal, prior)`，不再取 `abs(cos)`。
有效域为 prior mask、`alpha > .5` 和 depth-derived normal 有效像素的交集；不将
prior 乘以 alpha。法线加载、坐标转换与归一化维持当前实现。

NCC 使用同一 geometry 进度，默认日程为：

| iteration | NCC 权重 |
|---|---:|
| 1–7000 | 0 |
| geometry 前半程（7001–11000） | 从 .2 线性降至约 .1 |
| geometry 后半程（11001–15000） | 从约 .1 线性降至 .05 |
| 15001–20000 | 保持 .05 |

`lambda_multi_view_ncc` 是起始基准权重（默认 .2），后两个端点为其一半和四分之一；
传 0 可关闭 NCC 损失。短训仍将 geometry 终点截到总 iterations，并保留已有
`regularization_from_iter` 启用门控。geo 的权重和日程不随本次 NCC 迁移而改变。
没有加入 demo 的 depth-plane loss、背景或完整阶段式训练器。
`loss.jsonl` 同时记录当前 normal/NCC 权重及加权贡献。

**每个训练相机必须同时加载 normal prior 和 depth prior。** 缺任意一种即在
optimizer 初始化和训练循环之前报错，并指出帧名和缺失类型；即使对应 loss 权重
设为 0，也必须满足数据要求。检查加载标记而非 GPU 常驻属性，以兼容 CPU/GPU
帧缓存；不使用 demo 的“无 normal prior 时 NCC=.6”替代配方。

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

初始化先融合全部训练相机的 RGBD（voxel `.02`、trunc `.06`），再提取最终 TSDF
mesh，以共享边定义三角面连通分量，按面数降序只保留最大的 3 个；不足 3 个则全部
保留，同面数按原组件顺序决定。删除其余三角面后清除无引用顶点，再从保留顶点
无放回抽取最多 20 万个 XYZ/RGB 初始点，空表面直接报错。

这替代了原先“每融合一帧就抽点、最后拼接中间快照”的方式，避免中间浮点残留。
因此抽样分布也会改变，不是仅在旧 20 万点上删除一部分。原始 `tsdf_mesh.ply`
仍不作为输入，也不会被修改。指定 depth 来源、过滤参数及融合实现均进入缓存
身份；旧初始化缓存会失效，续训仍须通过原有初始化身份校验。

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
默认 20k iterations，启用上述 normal prior 与 NCC 日程。图像分辨率沿用原 CLI 默认；
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
