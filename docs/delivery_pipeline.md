# GGGS 交付入口

在仓库根目录、已激活的 GGGS 训练环境中运行：

```bash
python pipeline.py --dataset /workspace/dataset/245_for_train
```

入口依次检查输入、准备或复用虚拟机位与固定 D/N teacher、准备初始化点云，再调用现有
`train.py` 联合训练。它不调用 GenRecon 生成场景，也不自动对齐输入 mesh。
mesh、相机和 priors 必须已经处于同一 COLMAP 坐标系与尺度。

默认配置完整列在仓库根目录 [config.yaml](../config.yaml)：

| 项目 | 默认值 |
|---|---|
| 初始化 | Gen mesh 全表面面积采样，200,000 点 |
| 虚拟机位 | 最多 160 个，候选上限 1,200 |
| 轨迹平移比例 | `trajectory_offset_fraction: 0.16` |
| 表面证据采样 | 120,000 点，与初始化点云独立采样 |
| 机位评分 / teacher 分辨率 | 长边 256 / 512，保持宽高比 |
| 训练 | 20,000 iterations，appearance 模式 3 |
| 主优化器 | joint 模式每步更新；SH 保留独立更新节奏 |
| Gaussian 全局上限 | 2,500,000 |
| CPU 线程 / 真实帧 GPU 缓存 | 8 / 4 帧 |
| 虚拟 CPU / GPU 缓存 | 160 / 2 帧 |
| 初始化 RGB 缓存上限 | 2 GiB |
| 日志间隔 | 200 iterations |

160 是选择上限，不承诺所有场景都能找到 160 个合格机位。轨迹平移使用
`min(比例 × anchor teacher 中位深度, 比例 × 场景尺度)`，再通过路径和覆盖检查。
需要复现旧 8% 设置时显式配置 `.08`。

joint 的拓扑和 opacity 日程统一由 `geometry` 控制：`topology_interval` 决定拓扑检查间隔，
`topology_until` 决定停止增删点的阶段，`last_opacity_reset` 指定一次 opacity reset 的迭代。
`training` 不再接受旧的 `densification_interval`、`densify_until_iter`、`opacity_reset_interval`、
`opacity_reset_min_remaining`；复制旧 YAML 时应移除它们，并在 `geometry` 中设置所需行为。
`runtime.cpu_threads` 在重库导入前设置 OMP 线程，并用于几何预处理和 PyTorch CPU 线程设置。

## 输入

默认布局：

```text
DATASET/
├── images/                 # 真实 RGB，文件名与 COLMAP image.name 一致
├── depth/                  # 对应同名 .npy，float camera-Z
├── normals/                # 对应同名 .png，现有真实 normal prior 格式
├── sparse/0/               # cameras/images 的 COLMAP binary 或 text 模型
└── genrecon_mesh.ply        # 完整三角 mesh，已与 COLMAP 对齐
```

支持已去畸变的 `PINHOLE`、`SIMPLE_PINHOLE` 相机。所有注册帧都必须具备 RGB、depth、normal，
且它们的宽高必须与 COLMAP 一致。此交付入口使用全部注册真实帧；需要独立训练/验证划分时，
使用已有的底层训练入口及相应 teacher 准备流程。

`inputs` 中的路径相对于 `--dataset`，也可填写绝对路径。完整配置不用复制；`--config` 接受
局部 YAML 覆盖，未填写项继承根目录配置，未知字段会报错。例如：

```yaml
inputs:
  images: photos
  depth: global_depth/depth_npy
  normals: stable_normals
  colmap: refined_colmap/0
  mesh: /workspace/meshes/scene_colmap.ply
```

```bash
python pipeline.py --dataset /workspace/dataset/scene \
  --config /workspace/configs/scene_inputs.yaml
```

真实 depth/normal 继续用于原训练监督；虚拟相机只使用 mesh 渲染的几何，不生成虚拟 RGB。
teacher 的深度采用 camera-Z，法线采用朝向相机的相机空间实际面法线。

## 初始化选择与点数

默认 `gen_mesh` 按完整 mesh 的三角面积采样，不按顶点密度、不筛掉不可见表面。
颜色来自实际训练照片：按候选顺序做 mesh first-hit 遮挡检查，取第一个可见真实 RGB；
所有候选都不可见的点使用中性 `128`。不使用 Gen 贴图，不新增法线/opacity/filter 初始化配方。

切换 TSDF 并调整点数，例如保存为 `/workspace/configs/tsdf_init.yaml`：

```yaml
initialization:
  method: tsdf
  num_points: 300000
```

```bash
python pipeline.py --dataset /workspace/dataset/scene \
  --config /workspace/configs/tsdf_init.yaml
```

`gen_mesh` 生成指定数量的表面点；TSDF 沿用 RGBD 融合及最大三个连通分量过滤，点数是抽样上限，
最终数量以 `summary.json` 的 `initialization.actual_points` 为准。`num_points` 必须为正且不能超过
`geometry.max_gaussians`。切换初值后仍需提供 Gen mesh，虚拟机位与 D/N teacher 仍以它为源。

## 缓存与运行目录

默认写入数据集内的 `gggs/`，不会修改输入 RGB、priors 或 mesh：

```text
DATASET/gggs/
├── cache/
│   ├── cameras/<key>/
│   ├── teacher/<key>/
│   └── initialization/<key>/
└── runs/<timestamp>_<id>/
```

三个阶段分别计算 key；`delivery_cache.json` 记录完成状态和文件 SHA。完整且身份一致的缓存
默认复用；不完整、被修改或身份不匹配的已有目录会保留并报错。`cache.reuse: false` 会建立带
新后缀的独立缓存，不删除旧缓存；该选项不能与 `--resume` 同时使用。

| 缓存 | 影响其身份的主要输入 |
|---|---|
| cameras | mesh、COLMAP、真实 depth、输入路径、选择参数、seed、预览开关 |
| teacher | camera 缓存身份、teacher 分辨率、真实 RGB/normal 内容 |
| Gen 初始化 | mesh、COLMAP、真实 RGB、初始化方法/点数、seed |
| TSDF 初始化 | COLMAP、真实 RGB/depth、初始化方法/点数、seed |

改变初始化点数不会重新选择机位。teacher 沿用 schema 3 的保守来源绑定：即使虚拟 D/N 来自
固定 mesh，真实 RGB 或 normal 内容变化仍会使 teacher 缓存失效。不能将其解释为这些照片
或 priors 不影响缓存身份。运行还会校验实际源文件和 teacher 数组，不能只凭目录名判断可复用。

`--output` 只指定本次 run 的新目录；缓存仍在 `DATASET/gggs/cache`：

```bash
python pipeline.py --dataset /workspace/dataset/scene \
  --output /workspace/outputs/scene_delivery_01
```

已有非空 run 目录会报错，不作为一次新运行覆盖写入。使用原 A/B 入口生成的
`DATASET/virtual_geometry` 属于另一套分步产物；本入口按自身分阶段 key 管理缓存。

## 默认交付产物

成功的默认 20k run 包括：

```text
RUN/
├── point_cloud/iteration_20000/
│   ├── point_cloud.ply       # 原生 GGGS Gaussian 模型，含 filter_3D
│   └── appearance.pth        # 默认模式 3 的配对外观参数
├── cfg_args
├── resolved_config.yaml     # 实际合并后的完整配置
├── joint_config.json
├── training_config.json
├── summary.json             # 状态、阶段耗时、缓存命中、实际初始点数、最终模型路径
├── training_stats.json
├── virtual_geometry_stats.json
├── loss.jsonl
├── virtual_loss.jsonl
└── run.log
```

另保留现有训练的数据/相机来源记录。mode 3 的 `appearance.pth` 应与同目录 PLY 一起保留；
mode 4 则使用其配对 `ppisp.pth`。本入口不自动提取最终 mesh。

默认不生成机位预览、图像评估、中间 PLY、viewer PLY 或最终训练 checkpoint；最终原生 PLY
仍会保存。图像评估输出关闭不会关闭 joint 训练内部的几何审计、补点或预算检查。

需要诊断输出时用 YAML 显式开启，例如：

```yaml
outputs:
  camera_preview: true
  evaluation_iterations: [7000, 15000, 20000]
  save_iterations: [7000, 15000]
  viewer_ply: true
  checkpoint_interval: 5000
  save_final_checkpoint: true
  log_interval: 200
```

`camera_preview` 会生成 camera cache 中的原 mesh 渲染 PNG、`scene_preview.npz`，并自动调用已有
预览导出脚本生成交互式 `preview.html` 和 `contact_sheet*.jpg`。
`viewer_ply` 在保存模型时额外生成适配普通 3DGS viewer 的文件。最终迭代始终追加到 PLY 保存列表。

## 续训

默认不写 checkpoint；需要恢复能力，应在首次运行前设置非零 `checkpoint_interval` 或开启
`save_final_checkpoint`。PLY 和 appearance 文件不包含优化器、采样器及拓扑续训状态。
启用周期 checkpoint 后仅保留最新完成的一份及其配对状态文件；新文件保存成功后才清理旧文件，
不会保留每个历史保存点。

从已保存的、匹配当前数据与配置的 checkpoint 恢复，输出仍使用新目录：
例如任务中断时最新保留的是 `chkpnt10000.pth`：

```bash
python pipeline.py --dataset /workspace/dataset/scene \
  --config /workspace/configs/scene_with_checkpoints.yaml \
  --resume /workspace/outputs/scene_delivery_01/chkpnt10000.pth \
  --output /workspace/outputs/scene_delivery_01_resumed
```

保持同一数据、初始化、固定 teacher 和联合训练配置，并使用 `cache.reuse: true`。
原有 checkpoint 身份校验仍生效，恢复点必须早于本次配置的总 iterations。
如果调整总预算，还需让 `geometry` 各阶段满足配置中的顺序与预算约束。

## 已做的性能整理与边界

- 机位选择、teacher 和 Gen 初始化共享完整 CPU mesh、Open3D raycast 场及面积 CDF；
  atlas 与初始化仍各自用独立 RNG 采样，不能互相替代。
- 初始化颜色把同一候选 rank 的不同相机射线合并为一次 raycast；真实 RGB 使用有字节上限的缓存。
- teacher 在全池曝光预算确定后，每帧数组只压缩写入一次；新生成数组可直接交给训练缓存，避免立即重复解压。
- PLY 字段批量构建；生产默认关闭大量预览/中间文件，日志集中到配置间隔，减少逐项标量同步。

这些是实现层面的优化，不构成已测得固定提速倍数的结论。虚拟分支仍使用现有 renderer；
专用的纯 D/N CUDA 渲染路径和异步 CPU→GPU 预取没有在本轮实现。真实场景的端到端耗时及
最终 mesh 质量应分别验证；训练完成或 teacher 误差下降都不能代替实际 mesh 验收。
