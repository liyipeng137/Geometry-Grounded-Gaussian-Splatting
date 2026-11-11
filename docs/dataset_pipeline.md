# 按数据集运行：A 准备，B 训练

A 手动准备虚拟机位、预览和固定 depth/normal；B 运行训练。
A 不启动训练，B 不自动生成或修复这些虚拟几何产物。两步复用现有筛选、teacher和联合训练
算法，不改变其损失或日程。

下面的命令在仓库根目录、已激活的训练环境中执行。

## 完整两条命令

```bash
python scripts/prepare_virtual_geometry.py --dataset /workspace/dataset/245_for_train
bash scripts/run_dataset.sh /workspace/dataset/245_for_train
```

A 默认读取 `DATASET/genrecon_mesh.ply`，该 mesh 必须已经与 COLMAP 相机同坐标、同尺度。
它不调用 GenRecon 生成场景，也不做自动对齐。默认输出到 `DATASET/virtual_geometry/`：

```text
virtual_geometry/
├── cameras/
│   ├── manifest.json
│   ├── real_cameras.json
│   ├── virtual_cameras.json
│   ├── preview.html
│   ├── contact_sheet*.jpg
│   └── previews/
├── teacher/
│   ├── manifest.json
│   └── views/
├── train_config.json
└── pipeline.json
```

`cameras/preview.html` 和联系表用于查看机位、来源分组及mesh渲染预览。
`teacher/views/` 保存后续联合训练读取的固定几何缓存。
只有完整准备成功后，才发布可训练的 `train_config.json` 和 `pipeline.json` 的
`complete` 状态；中间产物不表示可以开始联合训练。
对同一数据集、mesh和seed再次运行A时，完整且通过验证的产物会直接复用。
不完整或不匹配的已有输出会保留并报错，不会自动覆盖重建。

B 默认沿用生产参数：appearance模式3、20k iterations、真实帧GPU缓存预算4、
`depth/` 与 `normals/`、`training_seed=0`、每100步日志；7k/15k/20k评估和保存，
7k/15k及最后一步保存checkpoint。额外的 `train.py` 参数放在数据集位置参数之后。

## B 如何决定训练模式

| 输入状态或选项 | 实际行为 |
|---|---|
| 默认 `DATASET/virtual_geometry/` 不存在 | 运行既有纯真实基线 |
| 默认目录有完整且有效的A产物 | 自动读取其 `train_config.json`，运行现有联合训练 |
| 默认目录存在，但训练所需产物不完整、失败或已与输入不匹配 | 报错；不静默退回基线，也不自动重做A |
| 显式 `--virtual_geometry_config PATH` | 优先使用指定配置，按该配置检查teacher |
| 显式 `--disable_virtual_geometry` | 强制纯真实训练，跳过自动发现 |

`--disable_virtual_geometry` 与显式 `--virtual_geometry_config` 互斥。
直接调用 `python train.py -s DATASET ...` 也使用相同发现规则，不局限于wrapper。

如果只需要纯真实训练，可以直接运行B，无需运行A；已有默认缓存时可明确关闭：

```bash
bash scripts/run_dataset.sh /workspace/dataset/245_for_train --disable_virtual_geometry
```

## 控制输出位置

B 不要求提供 `-m`，省略时由 `train.py` 在仓库的 `output/<run-id>/` 下创建训练输出。
需要固定地址时使用新的输出目录：

```bash
bash scripts/run_dataset.sh /workspace/dataset/245_for_train \
  -m /workspace/outputs/gggs_245_NEW
```

wrapper会切到仓库根目录执行。数据集位置参数支持调用时的相对路径；额外参数中的
路径建议使用绝对路径。已有训练结果不会作为新run的输出继续覆盖。

## 自定义mesh或A输出

A 提供 `--mesh`、`--output`、`--threads` 和 `--seed`。省略 `--mesh`、`--output`
时分别使用数据集内的 `genrecon_mesh.ply` 和 `virtual_geometry/`。
`--threads`、`--seed` 默认分别为8和0；A的seed用于预处理，B的训练seed由训练参数决定。
例如把新一版预处理保存在独立目录：

```bash
python scripts/prepare_virtual_geometry.py \
  --dataset /workspace/dataset/245_for_train \
  --mesh /workspace/meshes/scene_colmap.ply \
  --output /workspace/outputs/virtual_geometry_245_NEW \
  --threads 8 --seed 0

bash scripts/run_dataset.sh /workspace/dataset/245_for_train \
  --virtual_geometry_config /workspace/outputs/virtual_geometry_245_NEW/train_config.json \
  -m /workspace/outputs/gggs_245_custom_teacher_NEW
```

**自定义A输出不会写入数据集指针。** B不传显式配置时只检查默认
`DATASET/virtual_geometry/`，不会自动寻找最近一次或其他位置的预处理结果。
修改输入或重建缓存时，使用新的输出目录保留旧产物及其来源记录。

## 缓存绑定与续训

缓存绑定准备时的数据集路径和输入内容。移动数据集，或修改其mesh、相机、照片、
priors等已绑定输入后，需要重新准备匹配的A产物；旧缓存不会自动改绑。

自动发现不改变checkpoint的配方约束。旧纯真实checkpoint在数据集已有A缓存时，
续训须明确加 `--disable_virtual_geometry`；旧joint checkpoint则应显式传入原来的
`--virtual_geometry_config`。几何看起来相同，也不保证新目录中的manifest/config身份
相同。不能通过自动发现把旧checkpoint静默切换为另一种模式或另一份teacher。

## 初始化与虚拟监督独立

训练初值仍默认使用RGBD/TSDF。可以额外指定 `--init_mesh`：

```bash
bash scripts/run_dataset.sh /workspace/dataset/245_for_train \
  --init_mesh /workspace/dataset/245_for_train/genrecon_mesh.ply \
  -m /workspace/outputs/gggs_245_mesh_init_NEW
```

此参数只控制初始化，不替代A，也不重建teacher。训练是否使用虚拟D/N，仍由上表
中的发现规则或显式选项决定；需要mesh初值的纯真实训练时再加
`--disable_virtual_geometry`。PLY复用与续训见[初始化说明](mesh_initialization.md)，
联合loss、拓扑与日程见[虚拟几何训练说明](virtual_geometry_training.md)。

`scripts/run_245_virtual_geometry.sh` 保持原样，固定旧245数据集和旧teacher，
用于历史实验复现。新数据集使用本页的A/B入口；历史20k记录和产物不被覆盖。
