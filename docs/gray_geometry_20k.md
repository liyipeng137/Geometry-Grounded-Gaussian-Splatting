# 真实视角几何重建

`pipeline.py` 从注册照片和对齐到 SfM 坐标系的 camera-Z 深度，训练无颜色高斯，并用默认的 `mesh_extract_mini.py` 提取最终平滑网格。默认训练 20,000 步，随机种子为 0，真实帧 GPU 缓存为 4。

## 运行

激活项目的 GGGS 环境后，在仓库根目录运行

```bash
python pipeline.py --dataset /workspace/dataset/scene
```

新环境先用项目已有的脚本构建支持零颜色通道的 CUDA rasterizer。主入口会给训练和提面子进程加载这个仓库内的构建，也支持环境中已安装的同版本扩展。

```bash
bash scripts/build_geometry_rasterizer.sh build/geometry_rasterizer
```

输入沿用现有数据集布局。

```text
DATASET/
├── images/
├── sparse/0/             # 已去畸变的 COLMAP PINHOLE 或 SIMPLE_PINHOLE，相机、位姿及 points3D
├── depth/                # 与 image.name 对应的原分辨率 .npy，float camera-Z
└── depth_normal_prior/   # 可选；缺失的帧在训练前由深度生成相机空间法线 .png，保留相对子目录
```

默认初始化由 TSDF 采样点和 COLMAP 稀疏点组成。先融合训练 RGB-D，保留三角面数最多的 TSDF 连通块，从其顶点中采样最多 250,000 个点。对每个 TSDF 采样点，计算它到最近训练相机中心的欧氏距离，取所有距离中的最大值作为裁切半径。所有到最近训练相机的距离小于等于该半径的稀疏点都会按原始 XYZ/RGB 追加，包括与 TSDF 重合的点。250,000 是 TSDF 部分的上限，合并后的初始点数可以更多。

`pipeline.py` 和直接调用的 `train_real_geometry.py` 都默认启用 `--merge-sparse-init`。自动合并读取 COLMAP 的 `points3D.bin` 或 `points3D.txt`。如果文件中的点数为零，初值就是全部 TSDF 采样点。若只有相机和位姿文件，或需要复现仅用 TSDF 的基线，指定 `--no-merge-sparse-init`。

```bash
python pipeline.py --dataset /workspace/dataset/scene --no-merge-sparse-init
python train_real_geometry.py -s /workspace/dataset/scene \
  -m /workspace/results/scene_tsdf_only --no-merge-sparse-init
```

已有同坐标系的带色 XYZ/RGB 初始点云可以直接复用。显式 `--init-ply`（训练器使用 `--init_ply`）优先于自动初始化，文件中的点就是本次的完整初值。为一次运行指定输出时使用新目录。

```bash
python pipeline.py --dataset /workspace/dataset/scene \
  --init-ply /workspace/initialization/point_cloud.ply \
  --output /workspace/results/scene_01
```

默认输出为 `DATASET/gggs/runs/real_geometry_<UTC时间>_<ID>/`。其中 `input.ply` 是送入高斯模型的初始点云，`recon_bounded_post_smooth.ply` 是最终 mesh，`point_cloud/iteration_20000/point_cloud.ply` 是最终原生高斯。`geometry_config.json` 记录请求的初始化开关，`camera_split.json` 的 `initialization` 记录实际使用的初值。`pipeline_summary.json` 汇总初始化记录与产物路径。保留 0、7k、15k、20k 的原生高斯和最终的中性 viewer PLY。`--iterations` 可显式指定验证用步数。

需要查看 20k 运行的高斯渲染时，可在仓库根目录使用同一份 rasterizer。

```bash
run=/workspace/results/scene_01
PYTHONPATH="$PWD/build/geometry_rasterizer/lib:${PYTHONPATH:-}" \
  python render_gray_geometry.py -m "$run" --output "$run/render_bundle" --steps 7000 15000 20000
```

## 法线与致密策略

每个注册帧的法线位于 `DATASET/depth_normal_prior/<相对帧名>.png`。例如 `images/cam1/view.jpg` 对应 `depth_normal_prior/cam1/view.png`。初始化前为缺失 PNG 的帧生成法线，已存在的 PNG 原样复用，不做内容校验。深度或内参变化后需删除对应 PNG 让其重新生成。

生成使用原图内参和 COLMAP 半像素约定。将每帧有效深度反投影，在一像素邻域内计算水平与垂直差分。邻点深度跳变超过两者较近深度的 5% 时排除该邻点，单侧可用时采用单侧差分。取垂直与水平切向量的叉积，归一化并朝向相机，按 `round((1 - normal) * 127.5)` 编码为 RGB PNG。边框、无效深度以及缺少有效切向量的位置写为中性灰并由 GGGS 的法线掩码排除。训练监督和拓扑审计读取同一份 PNG。本次生成与复用的帧数记录在 run 的 `camera_split.json` 的 `normal_prior` 中。

每次真实几何训练都在新输出目录中直接生成 `input.ply`。合并时另存本次的 `tsdf_init.ply`，记录保留与裁掉的稀疏点数。

致密策略固定为 `accelerated`。默认 20k 训练中的增长日程如下。

| 首次检查至最后确认 | 检查间隔 | 增长决策次数 |
|---|---:|---:|
| 800～3,000 | 200 步 | 6 |
| 3,100～14,800 | 100 步 | 59 |
| 15,000～18,000 | 200 步 | 8 |

两次持续检查通过后执行长轴 split 或双真实视角 seed，共 73 次增长决策。剪枝沿用原来的 35 次确认，最后一次在 15,000 步。18,001～20,000 步固定点集。法线项的目标来自输入深度。实际日程和增删记录可查看 `geometry_config.json`、`geometry_stats.json`。

## 真实视角深度损失

默认 `--real-depth-loss-mode absolute` 在有效 prior 像素上监督渲染 camera-Z 的绝对 L1。对已经按帧做过正尺度对齐的单目深度，可以显式启用已有的 AffineGS 风格相对监督。

```bash
python pipeline.py --dataset /workspace/dataset/scene --real-depth-loss-mode affinegs
```

`affinegs` 将单帧 prior 转为逆深度，用当前渲染范围及引导滤波估计局部尺度和偏移。它在渲染和 prior 都有效的区域计算重新对齐后的深度 L1，并匹配逆深度梯度的幅值和方向。拟合出的目标停止梯度，损失计算最长边最多 960 像素。默认深度总权重为 0.4，梯度项对应的权重为 0.1。TSDF、法线生成、覆盖损失和拓扑审计继续读取原始对齐深度。运行采用的模式记录在 `geometry_config.json` 的 `args.real_depth_loss_mode` 和 `pipeline_summary.json` 中。

## 虚拟视角接入位置

主入口预留 `--virtual-geometry-config PATH`。当前只执行真实视角；显式传入该参数会在开始准备数据前退出，并提示此能力待验证。后续从 `pipeline.run` 接入已有机位清单和固定 depth/normal teacher，再向几何训练器传递虚拟监督。历史联合训练可通过 `pipeline_virtual.py` 单独复现，机位和 teacher 的准备代码位于 `utils/delivery_pipeline.py`。
