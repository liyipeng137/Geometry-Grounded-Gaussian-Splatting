# PPISP 外观模式 4

`--use_decoupled_appearance 4` 使用用户 fork 的 PPISP，作为现有模式 3（PGSR）的对照。
默认外观模式仍为 3。PPISP 子模块固定于
`a247368047651db2e03e6f97a36e4f5bebbe5c5f`，本次未修改 fork 源码。

## 安装与运行

在已有 PyTorch/CUDA 环境中：

```bash
git submodule update --init submodules/ppisp
python -m pip install -e submodules/ppisp --no-build-isolation
```

UKD-4090 已安装到 `/usr/local/miniconda3/envs/py312`，使用 CUDA 12.8 / RTX 4090。
模式 0–3 不导入 PPISP，不要求安装它。

两次独立运行使用相同数据、种子和几何配方，输出目录必须各自为空：

```bash
cd /workspace/Geometry-Grounded-Gaussian-Splatting
source /usr/local/miniconda3/bin/activate py312

python train.py -s /workspace/dataset/245_for_train \
  -m output/245_appearance3 --use_decoupled_appearance 3 --training_seed 0

python train.py -s /workspace/dataset/245_for_train \
  -m output/245_appearance4 --use_decoupled_appearance 4 --training_seed 0
```

当前数据的浮点深度在 `depth/`，与默认路径一致。默认 20k iterations、全部真实
相机、top3 RGBD 初始化及 normal/NCC 日程保持不变；比较时检查两次初始化 PLY SHA
和训练配置。显存不足时可对两组同时加 `--frame_gpu_budget 4`。

## 模式 3 与模式 4 的区别

| 项目 | 模式 3 | 模式 4 |
|---|---|---|
| 外观模型 | 每帧 gain/bias 两标量 | 每帧曝光/颜色，同相机组共享暗角/CRF |
| RGB L1 | 校正后的 RGB | PPISP 输出 |
| RGB SSIM | 原 render | PPISP 输出 |
| VCD/VCP 残差 | 校正后的 RGB | PPISP 输出 |
| 外观正则 | 原实现 | 官方 PPISP 正则，包含其配置权重，仅加一次 |
| 外观更新 | Gaussian 主优化器的 FastGS 日程 | 官方独立优化器，每个非末尾 iteration 更新 |

因此这是两套实际外观配方的比较，不是只有模型容量不同的严格单变量消融。
原几何参数、depth/normal/NCC/geo/mask 的定义和 Gaussian 更新日程没有改写。
PPISP 不处理深度或 alpha，不参与 Gaussian 的 clone/split/prune 参数切片。

控制器及 controller distillation 均固定关闭。全程保留 RGB 对几何的梯度，不在
80% 处冻结场景。官方 ISP 优化器、500 步 warmup 与默认 30k 衰减时间常数保持原值；
完整配置记录到 `training_config.json`。20k 主流程末步沿用现有跳过更新行为，
PPISP 共更新 19999 次，Gaussian 主优化器仍按 FastGS 日程更新。

## 相机、帧与像素

相机组按 COLMAP `colmap_id` 建立；训练帧按完整 `source_image_name` 记录身份。
当前245帧只有一个内参组，因此是 1 个 PPISP camera、245 个 frame。若实际设备或
成像链不同，应在输入标定中区分组；不要把每个 pose 当作独立的 ISP 相机。

输入 RGB 从 CHW 转为 HWC，使用整幅缩放图像的 `(x+.5,y+.5)` 像素中心。
首版不新增裁剪或稀疏像素训练接口。PPISP 后的 RGB 同时供光度损失与 VCD/VCP 使用。

## 保存与恢复

- 每个 `point_cloud/iteration_N/point_cloud.ply` 仍是 native Gaussian 文件，颜色是
  canonical 场景颜色。
- 同目录的 `ppisp.pth` 与 PLY 在同一时刻保存，包含 ISP 权重、帧/相机映射和配置。
- `point_cloud_canonical_3dgsviewer.ply` 仅烘焙 GGGS 的 `filter_3D`；**未烘焙 PPISP**。
  PPISP 含非线性、逐帧及像素位置相关的变换，不能等价折叠进一份普通 Gaussian PLY。
- 训练 checkpoint 的既有外观状态槽保存 PPISP 权重、优化器和 scheduler，旧模式
  0–3 的 tuple 布局不变。模式 4 恢复时必须保持模式及训练帧/相机身份一致。
- 不能将 mode3 checkpoint 直接冒充 mode4 checkpoint。公平对照从同一初始化重训。

PLY/ISP sidecar 在该步 optimizer 前保存；checkpoint 在 optimizer 后保存。不要将
同 iteration 的 checkpoint ISP 权重随意配到 PLY 上。`Scene(load_iteration=...)`
会读取配对 sidecar，缺少 sidecar 会明确失败。现有恢复不保存 Gaussian 未应用的
累积梯度/RNG，仍不能声称整个训练可逐位断点复现。

## 渲染、评估与 mesh

```bash
python render.py -m output/245_appearance4 --iteration 20000
```

保留原始 `renders/`，模式 4 额外写 `renders_ppisp/` 和 `appearance_rendering.json`。
已训练帧按完整文件名使用对应 ISP 参数；未知帧用 `frame_idx=-1`，只应用已知相机
的暗角/CRF，曝光/颜色为中性值。关闭 controller 的首版没有自动新视角曝光预测。
测试相机的 `uid` 即使与训练相机相同，也不会借用训练帧的曝光。

训练内评估保留原 raw L1/PSNR；模式3/4的校正后训练视角指标另列为
`appearance_diagnostics`。未知帧的 mode4 指标标为 `ppisp_camera_only`。
训练帧拟合指标不等于独立新视角质量，也不能用 mode4 processed PSNR 与 mode3
raw PSNR 直接比较。

mesh 提取仍消费原生 Gaussian 几何，不把 ISP 作用到 depth/occupancy。
现有提取器可以继续使用；原生 mesh 顶点颜色是 canonical，贴图/ISP 呈现应另行处理。
比较几何时对两组使用相同的提面命令，优先检查未平滑 mesh、薄结构、孔洞及固定
真实视角的 depth/normal，而不是只比较 SuperSplat 显示或 RGB 分数。

## 本次验证

63 项接口/数值测试及 34 个子用例通过。官方 PPISP CUDA↔Torch 的4项对照测试通过。
两模式同初始化完成18步真实数据短训；模式4经过增密、checkpoint续训及离线渲染，
验证1个相机组/3个训练帧与1个留出帧的映射，ISP各步更新和正则梯度正常。
这些检查验证接入，不代表 PPISP 已提高完整场景 mesh 质量。

原始记录：`/workspace/gggs_ppisp_validation_20261109/`。

## 完整 20k 对照（2026-11-09）

按用户选择，复用已有模式3 `output/d8dde4d7-3`，只新跑模式4：
`output/245_ppisp_mode4_20261109`。除模式和输出路径外，保存的 model/optimization/
pipeline 参数一致；直接复用经指纹校验的初始化缓存，初始 PLY SHA 相同。
输入数据、相机标定和划分指纹一致。未更改或覆盖模式3已有产物。

模式4完整完成20k，进程耗时802.8秒；主优化器15156步、SH1093步，PPISP19999步。
实际映射为1个相机组、245个训练帧。历史模式3没有独立测量总耗时，不据此声称提速。

固定12个真实训练视角（含frame89），均使用宽480的相同像素网格，得到：

| 指标 | 模式3 | 模式4 |
|---|---:|---:|
| Gaussian 数量 | 2,110,311 | 2,109,250 |
| raw/canonical RGB pooled PSNR | 24.393 dB | 22.307 dB |
| 各自外观校正后 RGB pooled PSNR | 29.222 dB | 29.454 dB |
| 校正后 RGB L1 | .017220 | .016170 |
| 固定 prior 域 depth MAE（场景单位） | .031817 | .032179 |
| 固定 prior 域 relative depth MAE | .026828 | .027064 |
| 不透明命中率 | 99.9885% | 99.9862% |
| 固定 normal 域平均夹角（缺面记180°） | 6.2820° | 6.2793° |

5/12视角的depth误差变大；frame89由.10999增至.11814。总体照片拟合略好，几何
prior一致性没有整体改善，因此当前不据此将模式4升级为默认。raw颜色属于不同
canonical表示，其PSNR降低不能直接解释为几何退化。

这不是独立GT或新视角评估：这些prior/照片参与了训练，normal指标来自Gaussian
聚合法线，并非实际mesh三角面。未进行两组统一提面后的mesh质量验收。原始数据
指纹覆盖RGB/depth；normal按当前文件计算，并另外记录其SHA，未冒充历史normal证明。

比较脚本、输入证明、JSON与三张可视化位于验证目录的 `full_comparison/`，脚本
`compare_appearance_runs.py` 不构造Scene、不修改任何训练结果，也不再拟合外观参数。
模式3用最终checkpoint中的每帧曝光，模式4用最终PLY配对的 `ppisp.pth`。
