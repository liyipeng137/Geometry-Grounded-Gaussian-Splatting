# 虚拟几何联合流程验证（2026-11-10）

范围：实现、数值接口、真实数据短训及续训。完整20k与最终mesh质量对照尚未运行。
测试机器为UKD-4090，RTX4090，PyTorch2.9.1+cu128，Open3D0.19。

## 冻结输入

沿用已审核的160个虚拟pose，不扩大轨迹平移；源为
`/workspace/outputs/gggs_virtual_cameras_245_20261110/attempt_01/manifest.json`。
teacher为`/workspace/outputs/gggs_virtual_geometry_245_20261110/teacher_v1/manifest.json`。

- 160个pose逐值相同；K和网格从评分分辨率严格放大2倍，生成384×512浮点D/N。
- 31,457,280个总像素；有效D为30,514,123（97.00%），有效N为27,372,812（87.02%）。
- D有效域unknown/support/mixed/conflict约59.04%/33.45%/6.04%/1.47%。
- 全池Z、区域曝光q、像素计数重算一致；正常法线模长最大误差5.96e-8。
- 743个真实数据/mesh/已认可pose来源与缓存SHA校验通过。

这些是坐标、身份和缓存一致性证据，不是生成mesh的独立几何真值证明。

## 工程测试

远程完整测试：**172 passed，34 subtests passed**。Ruff、Python语法和启动脚本
`bash -n`检查通过。

- Native CUDA：两层Gaussian的`alpha*T`与深度带归属、偏心主点K、空射线、有限根搜索
  扩展、有效D梯度及joint-only无效D保护、coverage梯度。
- Teacher/loss：固定域归一化、unknown降权、洞边N stencil、近处错误表面的D梯度、
  缓存身份/篡改检查、只读view与独立sampler。
- Topology：参数/Adam/metadata事务、空事务、待SH梯度拒绝、预算、filtered opacity、
  实际seed插入、geometry绕过RGB gate、VCP排序、reset宽限、遮挡unknown不误删、
  按真实距离去重、见证方向/新鲜度、过期队列清理。
- Runtime：真实/虚拟抽样、Python/NumPy/Torch RNG、非边界pending SH梯度、身份拒绝，
  以及恢复mode3参数与optimizer绑定的生产方法回归。

旧compact-box对恰好对角conic的剔除行为没有在本次顺手修改。Native fixture使用
旋转的各向异性Gaussian，并显式断言非空alpha、有效depth与非零梯度，避免空场景
使数值一致性测试假通过。

## 最终版本的真实数据短训

目录：`/workspace/gggs_virtual_joint_validation_20261110/final_smoke_96/`。
使用完整245真实帧、全部160虚拟缓存；真实图像缩到320像素宽，虚拟仍为512长边。
日程显式压缩以覆盖ramp、reset、审计、补点、事务、固定拓扑和保存；不是正式质量配方。

| 记录 | 结果 |
|---|---:|
| 真实帧训练/主更新 | 96 / 96 |
| SH实际更新 | 6 |
| 虚拟训练曝光 | 38 |
| 几何审计 / 拓扑事务 | 7 / 3 |
| 直接mesh补点 | 108 |
| 起始 / 最终Gaussian | 200,000 / 194,073 |
| 清理过期缺口候选 | 9,013 |

四项虚拟loss均实际执行并出现非零值。完成真实视角评估、PLY/viewer导出、71步与
96步checkpoint。该短日程没有在真实场景触发geometry-parent split通道；该通道由
专用事务fixture及独立native归属测试验证，不把这次短训说成其质量验收。

## 非SH边界续训

从71步恢复到96步，目录
`/workspace/gggs_virtual_joint_validation_20261110/final_resume_71_to96/`。

- 额外25个真实样本、25次主更新、2次SH更新，累计虚拟样本38。
- 真实相机序列与虚拟相机/iteration序列完全一致。
- point/parent/source/保护期/见证/检查等整数与布尔身份字段完全一致。
- 修复了旧restore在optimizer构建后替换appearance Parameter的绑定错误。

**不声称GPU数值逐位或严格allclose复现。** 最终checkpoint的xyz最大绝对差约
0.00396场景单位，raw opacity最大差约0.04436；appearance差约4.66e-10。
原始不通过的allclose结果完整保留，未通过改epsilon或阈值掩盖。

另对71步checkpoint执行只读即时恢复检查：模型、Adam、pending梯度、filter、RNG等
48个张量和scalar/group字段逐值相同；opacity饱和修补修改数为0，appearance对象绑定
正确。这排除了该次即时恢复丢状态，但不能单凭此把后续所有差异归因于CUDA。
短训active SH为0，因此真实checkpoint的pending SH为存在但全零张量；非零pending SH
的更新连续性由CPU生产runtime/Adam测试覆盖。

对6个固定虚拟机位作只读原生RGB/D/N/alpha比较：多数机位差异很小，但一个机位
有42个D50有效性翻转，depth均差约0.00113、最大差约2.93场景单位。因此这轮证明
状态/抽样能正确续接，不证明所有几何像素严格复现；短训本身也不是稳定最终表面。

## 原始证据与下一步边界

验证根目录：`/workspace/gggs_virtual_joint_validation_20261110/`。

- `all_tests_release.log`、`renderer_tests.log`、`teacher_loss_tests.log`。
- `teacher_preparation.log`、`smoke_config.json`。
- `final_smoke_96.log`、`final_resume_71_to96.log`。
- `final_resume_comparison.json`、`final_resume_render_comparison.json`。
- `restore71_immediate_audit.json`。
- `backup/`：变更前源码和rasterizer二进制，原Torch/CUDA环境未升级。

源数据、已审核pose与原baseline结果未覆盖。下一步可按正式20k配置训练，再用统一
提面与固定真实视角评价实际mesh；本次未把短训PSNR或teacher-fit作为质量提升证据。
