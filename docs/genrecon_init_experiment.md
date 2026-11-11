# GenRecon 全局点云初始化实验

正式训练现可直接传 `--init_mesh`，由 `train.py` 自动完成采样与训练照片着色，
无需先单独运行准备脚本：

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 bash scripts/run_245_virtual_geometry.sh \
  /workspace/outputs/gggs_245_mesh_init_NEW \
  --init_mesh /workspace/dataset/245_for_train/genrecon_mesh.ply
```

初始化缓存位于新 run 的 `initialization/`。固定20万点，采样 seed 来自
`--training_seed`；颜色仅取实际训练划分及 `--images` 对应的照片。
此参数不重建虚拟 teacher。默认 TSDF、PLY复用、续训与离线渲染的完整说明见
[Mesh 初始化入口](mesh_initialization.md)。

## 已完成实验的两步流程

下面保留2026-11-11实验实际使用的“准备 PLY → `--init_ply`”流程，
不将历史运行改写为使用新入口。示例中的历史输出目录已有产物；复跑时使用新目录。

本实验从 baseline02（`3981080`）的完整 20k 联合流程出发，只替换初始 XYZ/RGB：
在同 COLMAP 坐标系的 GenRecon mesh 全表面按三角面积均匀抽取 200,000 点，
再从真实照片投影颜色。不会筛除所有训练相机都不可见的表面，也不增加
Gen 专用 filter、早期保护、法线朝向初始化或增密/剪枝策略。

`scripts/prepare_genrecon_init.py` 使用固定随机种子 0；按入射角绝对值/距离平方
排列真实机位，逐候选检查原 mesh 的 first-hit 遮挡，使用首个可见机位的双线性 RGB。
候选全部不可见的点保留中性灰 128。COLMAP 半整数坐标投影后减 0.5 再索引照片。
输出 `point_cloud.ply`、`metadata.json` 和 `sampling_provenance.npz`，记录来源、
输入 SHA、采样面与着色机位；非空输出目录禁止复用。

```bash
python scripts/prepare_genrecon_init.py \
  --dataset /workspace/dataset/245_for_train \
  --mesh /workspace/dataset/245_for_train/genrecon_mesh.ply \
  --output /workspace/outputs/gggs_245_genrecon_init_20261111/initialization \
  --seed 0 --threads 8

OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 bash scripts/run_245_virtual_geometry.sh \
  /workspace/outputs/gggs_245_genrecon_init_20261111/training \
  --init_ply /workspace/outputs/gggs_245_genrecon_init_20261111/initialization/point_cloud.ply \
  --test_iterations 3000 3500 5000 7000 10000 14000 15000 20000 \
  --checkpoint_iterations 3000 3500 5000 7000 10000 15000
```

`--init_ply` 仅覆盖 COLMAP 初始化点云；不传时保留原 RGBD/TSDF 路径。
真实图像、depth/normal、相机划分、160 个冻结虚拟机位和 teacher、loss 日程、
mode 3、250 万点上限及 15k 停止拓扑变化均沿用 baseline02。
这是从零训练，不能传入 baseline02 的 `--start_checkpoint`。

外部初始化记录为 `external_ply` / `external_colored_ply_v1`，按 PLY 内容验证；
checkpoint 保留该来源，续训时不能换成另一批初始化点云。
训练结果独立保存，baseline02 目录保持只读。最终 mesh 由用户手动提取。
