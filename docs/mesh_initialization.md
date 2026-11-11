# Mesh 初始化入口

新训练不指定初始化选项时，仍使用当前训练相机的 RGBD/TSDF 点云。
`--init_mesh PATH` 在 `train.py` 内自动准备初值；`--init_ply PATH` 则直接读取
已经准备好的 XYZ/RGB 点云。两项互斥。

**输入 mesh 必须已经与 COLMAP 相机处于同一坐标系和尺度。** 入口不执行配准、
坐标变换或自动缩放。当前接受三角形 PLY，使用去畸变的
`PINHOLE` / `SIMPLE_PINHOLE` 相机及仓库现有 Open3D 0.19 raycast 环境。

## 首次训练时做什么

1. 从原 mesh 的全部三角面按面积均匀采样 **200,000 点**，保留不可见表面。
   采样使用 `--training_seed` 的局部随机数生成器，不额外改变训练的全局随机状态。
2. 只使用实际训练划分中的照片着色。机位按入射角与距离排序，用原 mesh 的
   first-hit 遮挡和命中面检查选择可见照片，再双线性采样 RGB。
   没有可见训练照片的点保留为灰色 `128`。
3. 在本次输出目录的 `initialization/` 中保存 `point_cloud.ply`、`metadata.json`、
   `sampling_provenance.npz` 和 `state.json`。记录 mesh、照片、相机、seed、采样面及
   着色来源，随后由原 Gaussian 初始化流程读取点云。

首版采样数量固定为 20 万。着色遵守 `--images` 指定的图像目录，以及 `--eval`
或 `--train_exclude_images` 得到的训练划分；验证/排除帧不参与初始化颜色。
沿用现有规则，`--eval` 与显式 `--train_exclude_images` 不能同时使用。

mesh 入口只替换初始 XYZ/RGB。它不保留 mesh 的连接关系，也不把其法线直接设为
Gaussian 朝向；scale、opacity、3D filter、增密/剪枝和损失仍按所选训练模式执行。
真实训练帧的 RGB、depth prior、normal prior 输入要求保持不变。

## 使用例子

以下命令在仓库根目录、已激活的训练环境中执行。每个新训练使用独立输出目录。

### 默认 TSDF 初值

```bash
python train.py -s /workspace/dataset/245_for_train \
  -m /workspace/outputs/gggs_tsdf_NEW \
  --frame_gpu_budget 4 --iterations 20000 --training_seed 0
```

### Mesh 初值，使用真实相机训练

```bash
python train.py -s /workspace/dataset/245_for_train \
  -m /workspace/outputs/gggs_mesh_NEW \
  --init_mesh /workspace/dataset/245_for_train/genrecon_mesh.ply \
  --frame_gpu_budget 4 --iterations 20000 --training_seed 0
```

### Mesh 初值，加上已准备的虚拟 D/N

```bash
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 bash scripts/run_245_virtual_geometry.sh \
  /workspace/outputs/gggs_mesh_virtual_NEW \
  --init_mesh /workspace/dataset/245_for_train/genrecon_mesh.ply
```

该脚本显式传入 `--virtual_geometry_config configs/virtual_geometry_245.json`，
使用已有的160视角 teacher。**初始化来源与虚拟监督独立选择：** 单独传
`--init_mesh` 不会开启虚拟 D/N，也不会重新生成或覆盖 teacher。

切换训练划分时，虚拟 teacher 也必须与实际训练相机集合一致；现有全245帧
teacher 不能直接用于加了 `--eval` 或排除列表的训练。相关流程见
[虚拟几何训练说明](virtual_geometry_training.md)。

### 跨 run 复用已生成的点云

```bash
python train.py -s /workspace/dataset/245_for_train \
  -m /workspace/outputs/gggs_mesh_ply_reuse_NEW \
  --init_ply /workspace/outputs/gggs_mesh_NEW/initialization/point_cloud.ply \
  --frame_gpu_budget 4 --iterations 20000 --training_seed 0
```

这会读取同一份点云，不重新采样或着色。原始 mesh 采样记录仍保存在来源 run 的
`initialization/` 中；新 run 按该 PLY 的内容 SHA 记录初值。

### 从 checkpoint 续训

```bash
bash scripts/run_245_virtual_geometry.sh \
  /workspace/outputs/gggs_mesh_virtual_resume_NEW \
  --start_checkpoint /workspace/outputs/gggs_mesh_virtual_NEW/chkpnt7000.pth
```

续训沿用 checkpoint 记录的初值，不必再次提供 `--init_mesh`。程序从旧 run 的
`input.ply` 或原初始化缓存中定位同一份点云并校验 SHA，不重新采样、着色或切回
默认 TSDF。原 mesh 可以不再存在，但同内容的初始化点云缓存必须仍可读取。

显式提供 `--init_ply` 时，其内容 SHA 必须与 checkpoint 一致；显式提供
`--init_mesh` 时，须对应 checkpoint 记录的 mesh 来源路径。续训始终使用已冻结的
点云，不会因为同路径 mesh 被修改而重新生成初值。真实数据、划分及联合模式的
teacher/config 仍须满足原有续训校验；改变初值应使用新的训练 run。

## 缓存与离线渲染

自动准备要求 `initialization/` 为空；不会覆盖非空目录。跨 run 复用请使用
`--init_ply`，checkpoint 续训则走上述恢复路径。

mesh 产生的点云仍按现有 `external_ply` / `external_colored_ply_v1` 格式记录，
另外保存 `requested_source="mesh"`、mesh 路径/SHA 及元数据路径。续训保留这份来源
记录，实际读取的缓存位置不改变原初始化身份。

`render.py` / `Scene(load_iteration=...)` 直接加载已经训练的 Gaussian PLY，跳过
初始化准备；不会因为记录中有 `init_mesh` 而再次读取或采样原 mesh。

```bash
python render.py -m /workspace/outputs/gggs_mesh_virtual_NEW --iteration 20000
```

离线渲染仍需要对应相机/图像数据和已保存的模型，但不要求原 mesh 文件继续存在。
