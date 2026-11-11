# GenRecon 虚拟几何正式训练流程

本流程直接接入 `train.py`。每一步仍训练一张真实照片，按阶段额外加入一张虚拟
depth/normal。真实 RGB、原 D/N、NCC/geo、自一致性及默认 appearance=3 保留。
虚拟机位使用预先准备的固定pool，不增加生成纹理或虚拟 RGB。

新数据集的默认使用方式见[A 准备、B 训练](dataset_pipeline.md)。本页以下245数据集
命令固定复用历史160机位teacher，用于复现已经完成的实验。

实现与短训证据见 [2026-11-10验证记录](virtual_geometry_validation_20261110.md)。

## 运行

远程仓库 `/workspace/Geometry-Grounded-Gaussian-Splatting`，环境 `py312`。
已准备的完整 teacher 为：

`/workspace/outputs/gggs_virtual_geometry_245_20261110/teacher_v1/manifest.json`

```bash
cd /workspace/Geometry-Grounded-Gaussian-Splatting
source /usr/local/miniconda3/bin/activate py312
bash scripts/run_245_virtual_geometry.sh /workspace/outputs/gggs_245_virtual_joint
```

输出目录必须是新的。等价入口是 `python train.py ... --virtual_geometry_config
configs/virtual_geometry_245.json`。通用入口省略配置时会检查数据集默认缓存；可用
`--disable_virtual_geometry` 明确选择纯真实训练。初始化默认使用
train-only RGBD/TSDF；也可独立指定 `--init_mesh` 或 `--init_ply`，见
[初始化入口](mesh_initialization.md)。切换初值不改变真实训练划分或重建虚拟teacher。
联合模式依赖受控的3D filter，不接受`--disable_filter3D`，避免该标志被周期刷新静默覆盖。

首次换机器需要编译本分支的 rasterizer；旧 wheel 不包含新增几何评分接口：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.9
python -m pip install -e submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
```

需要重新生成 teacher 时，使用新目录并更新配置中的路径；不要覆盖已绑定checkpoint的缓存：

```bash
python scripts/prepare_virtual_geometry_teacher.py \
  --approved-manifest /workspace/outputs/gggs_virtual_cameras_245_20261110/attempt_01/manifest.json \
  --dataset /workspace/dataset/245_for_train \
  --output /workspace/outputs/gggs_virtual_geometry_245_20261110/teacher_NEW \
  --long-edge 512 --threads 8
```

## 阅读代码的顺序

1. `utils/virtual_geometry_config.py`：所有工程初值、阶段边界与虚拟频率。
2. `train.py::training()`：初始化，真实反传，可选虚拟反传，参数更新，拓扑事务，保存。
3. `utils/virtual_geometry_losses.py`：D、两种N、coverage和固定域缺失统计。
4. `utils/geometry_topology.py`：独立几何评分、增长预算、补点、剪枝与filter。
5. `utils/joint_runtime.py` / `utils/training_io.py`：日志、身份绑定及完整续训状态。
6. `scene/virtual_cameras.py` / `utils/virtual_geometry_data.py`：精确相机与只读teacher缓存。

## 一次20k的阶段

iteration只由真实帧推进；一张虚拟帧不是额外optimizer step。

| 区间 | 虚拟监督 | 几何生命周期 |
|---|---|---|
| 1–3000 | 无 | 真实约束建立结构；3000为最后一次opacity reset |
| 3001–3500 | 无 | 恢复覆盖，禁止因刚reset而低opacity软删 |
| 3501–5000 | 每4步一次，权重渐入 | 建立几何审计和持续缺口记录 |
| 5001–14000 | 每2步一次 | 两路增密、显式mesh补点、保护与剪枝 |
| 14001–15000 | 每2步一次 | 不再新增mesh seed，完成已有原语调整 |
| 15001–20000 | 每4步一次 | 不增删点，不调用旧18k光度剪枝，继续收敛 |

默认20k真实渲染，额外6625次虚拟训练渲染。几何审计另计。
Joint启用`dense_optimizer_updates=True`：主参数每步更新，SH每16步更新；包括末步。
因此完整20k是20,000次主更新、1,250次SH更新，较原FastGS尾段增加实际预算。
不能把同为20k iteration称作同计算预算，也不把全流程收益归因于单个loss。

## 几何监督的职责

`L_virtual = ramp * (.20 L_D + .05 L_Ng + .05 L_Nd + .05 L_C)`。

- `L_D`：相对Charbonnier，保留camera-Z绝对坐标，不拟合逐图scale/shift。
- `L_Ng`：归一化后的Gaussian聚合法线与mesh面法线的有向cosine。
- `L_Nd`：depth导出的法线与mesh面法线，要求有效的完整深度stencil。
- `L_C`：`relu(.8-alpha)^2`，促使已有贡献建立覆盖，不声称它独自决定表面位置。

Teacher有效域、q与归一化Z由完整固定机位池决定；可靠性只乘分子。
unknown权重不会因除以自身权重而被抵消。mask不会因为学生缺面而缩小分母。
原mesh无命中处不自动当“必须为空”。不把生成表面当独立GT。

只有有效D50具有depth梯度。无贡献、低alpha、高alpha但无根分开统计；joint最多扩展
两次根搜索区间，仍失败则屏蔽无效D梯度，不把该错误当普通空洞反复补点。
alpha约`.55`是现有D50门槛，`.8`是覆盖目标；两者含义不同。

虚拟损失不给SH或appearance直接梯度；真实照片继续更新这些参数。虚拟相机不加入
真实相机集合、appearance ID、PatchMatch邻居、scene extent或全体filter相机池。

## VCD/VCP与几何通道

真实通道仍使用照片残差、真实屏幕梯度和VCD。几何通道单独审计D/N：新增native评分
按`weight * alpha * T`及teacher射线深度带关联Gaussian，分别记录误差、正确支持、
带内机会和带外贡献。原整数metricCount不被冒充为正确几何的支持证明。

几何候选无需通过真实RGB高误差gate。高误差意味着需要修改，不意味着永久保护。
稳定正确几何支持可以保护真实图像看不到的补全面；VCP只对符合删除条件的候选排序，
不单凭外观误差删除已建立的生成表面。

完全缺支撑处使用稳定的世界表面候选跨view复查，经多次/多机位确认后入补点队列。
先检查真实自由空间、已有椭球支撑和重复样本；避免随机像素难以再次命中同一位置、
日志里一直有缺口而始终插不出点。

## 新点与预算

新点位于固定mesh表面，法向对应最小尺度轴，保留非零厚度；初始filtered opacity约
`.25`。有真实可见照片时逆mode3 gain/bias取得canonical DC，否则用中性色，higher SH为0。
不新增虚拟曝光参数。

批内按固定世界索引和双方实际spacing距离去重；两个见证需要近期缺面记录且射线
夹角至少3°，同位置转向的不同ID不算两种方向。失效超过最近两轮的缺口记录释放队列容量。

无真实frustum支撑的Gen点及后代使用局部虚拟footprint filter并限制相对局部间距的大小；
不沿用全场最大距离回退，以免细薄新点刚插入就被filter压低有效opacity。

统一硬上限250万点。每批净增长最多`min(20000,.05*N,剩余额度)`，默认真实/几何/seed
预算比例50/25/25，空余可借用。直接seed每批最多5000且不超过当前点数1%，累计15万。
clone与二分split均按净增1计费，先算最终点数再申请新张量。

Seed有短期grace和见证机位审计额度；子代不能通过重复分裂刷新保护。
真实自由空间硬冲突不受grace保护。未被复查不等于无支持，不仅凭wall-step过期删点。
几何来源本身不提供永久豁免。

## 参数更新与拓扑事务

两路backward完成后只做一次主更新。约每100步到期的拓扑事件等待SH累计梯度被消费的
16步边界，再一次性执行保留/删除/clone/split/seed。空事务不替换Parameter。

同一个索引映射同步九类Gaussian参数、Adam moments、filter、radii、统计和来源ID；
旧点保留优化历史，新点moments置0，组级Adam step继续沿用。新点不会在同一次事务内
再次被选为父点。新流程的PLY/checkpoint在参数更新和事务之后保存。

这修复了把新loss简单接在旧before-step拓扑函数前会丢失当前梯度的问题。
另外修复了恢复mode3时先建optimizer、再替换embedding导致优化器绑定旧Parameter的问题；
恢复的外观对象现在先设置，再创建并加载optimizer。

## 保存、续训和日志

联合checkpoint使用独立recipe，不能把真实baseline checkpoint静默当joint续训。
正式cache与配置identity绑定；真实划分必须与teacher来源一致，不能使用`--eval`偷偷
把已经参与teacher的图像当作独立留出集。

恢复包含真实剩余抽样栈、虚拟sampler、全局RNG、待应用的SH梯度、filter、逐点来源和
保护期、审计游标、增长队列和预算计数。允许换输出目录和延长训练终点，但修改teacher
或联合配方需要新run；不承诺不同GPU上的位级确定性。

```bash
bash scripts/run_245_virtual_geometry.sh /workspace/outputs/gggs_joint_resume \
  --start_checkpoint /workspace/outputs/gggs_245_virtual_joint/chkpnt15000.pth
```

- `training_config.json`：实际配方、teacher identity、optimizer/save顺序。
- `loss.jsonl`：真实各项loss与权重。
- `virtual_loss.jsonl`：每次虚拟曝光、四项loss及固定域缺面指标。
- `virtual_geometry_stats.json`：累计虚拟样本、拓扑事件/补点/预算与保护状态。
- `training_stats.json`：本段真实样本、实际主/SH更新次数和joint摘要。
- `chkpnt*.pth` / `.state.json`：嵌入状态及可读身份/哈希；sidecar不是恢复权威。

配置支持明确的短日程用于工程验证，默认日程不会因为`--iterations`很小而偷偷缩放。
短训只验证生命周期与接口，不用于评价完整场景质量。完整mesh验收需统一提面后检查
实际表面、薄结构、恢复孔洞与新增孔洞，而不是只看teacher-fit或聚合法线。
