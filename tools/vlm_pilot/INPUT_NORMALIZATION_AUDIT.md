# 骨架输入均值/方差配置审查（更新2026-10-22）

## 2026-10-22服务器回传：六协议训练集统计已收到，不是LP实验

原始证据：[控制台附件](C:/Users/97537/.codex/attachments/d7a88672-8604-41db-aa66-e513fd8c1e1e/已粘贴的文本.txt)，185行、14218字节，SHA256=`75627e10b77ea1a2ab2fb20adfb7b728e8395556acc7577e6e34e5173668327f`。附件保留六配置完整命令：`--view both --person-mode compare --draws-per-sample 1 --seed 0 --output-dir output_dir/normalization_audit/all_datasets`。没有复制原日志或新建归档目录。

六个协议均有最终统计摘要，scanned分别为40091、37646、63026、54468、18840、5299，累计219370条协议内训练条目；不同协议可能共享样本，不能当219370条独立、去重数据。各组person0全位置计数严格等于`N*120*25`，all全位置严格等于其2倍。所有组`skipped_all_zero=0`、`person0_empty=0`、`native_frame_rule_differs=0`。附件未见异常堆栈，但没有单独的退出码和最后Reports行，不额外补猜进程状态。

**证据范围：** 命令虽为view=both，当前print_report仅显示`model_input`和配置选中的self_shift；raw_valid及另一种self_shift的数值在服务器JSON/CSV中，本次未上传。以下为控制台约7位有效数字，不伪装成完整精度JSON；未收到服务器配置/脚本SHA、NPZ身份或生成文件，不把本地同名文件当作服务器逐字核验。每样本一次固定种子随机变换是全训练集扫描的Monte Carlo估计，不是对所有crop/rotation精确积分。NTU60 XSub当前FM配置不旋转，其他五组当前主配置旋转，因此跨协议差值也包含预处理差异。

### 当前PT person0输入的实测值

以下统一选**model_input / person0 / model_positions**，与当前PT保留person0且没有过滤空位置的输入口径对应。PKUv1为self_shift=True，其余False；var为方差，未开平方。

| 协议 | 训练条目 | self_shift | measured mean XYZ | measured var XYZ |
|---|---:|---|---|---|
| NTU60 XSub | 40091 | False | `[-0.003405887, -0.220585, -0.04651074]` | `[0.04780319, 0.159242, 0.0464151]` |
| NTU60 XView | 37646 | False | `[-0.001669546, -0.2141895, -0.04495448]` | `[0.05182047, 0.1561639, 0.05459429]` |
| NTU120 XSub | 63026 | False | `[-0.003250466, -0.2170586, -0.03844483]` | `[0.04839811, 0.1578478, 0.05186343]` |
| NTU120 XSet | 54468 | False | `[-0.002060315, -0.2201573, -0.02933916]` | `[0.04821338, 0.1603031, 0.05600233]` |
| PKUv1 XSub | 18840 | True | 约`[0,0,0]`（绝对值<1e-18） | `[0.0372804, 0.1905364, 0.03001078]` |
| PKUv2 XSub | 5299 | False | `[-0.08982385, -0.1312416, 2.665961]` | `[0.4580779, 0.453894, 0.2666239]` |

应用**当前主配置原有常数**后的统计：前五组用A方差，PKUv1有效mean被self_shift覆盖为0；PKUv2用H。

| 协议 | normalized mean XYZ | normalized std XYZ |
|---|---|---|
| NTU60 XSub | `[0.01668058, -0.3076392, -0.1483983]` | `[1.523333, 1.406471, 1.459155]` |
| NTU60 XView | `[0.02877824, -0.2850979, -0.1378579]` | `[1.586051, 1.392812, 1.582506]` |
| NTU120 XSub | `[0.01776344, -0.2952102, -0.09376903]` | `[1.532783, 1.400301, 1.542419]` |
| NTU120 XSet | `[0.02605562, -0.3061317, -0.03209762]` | `[1.529855, 1.411149, 1.602783]` |
| PKUv1 XSub | 约`[0,0,0]` | `[1.345261, 1.538477, 1.173303]` |
| PKUv2 XSub | `[-0.09049254, -0.1304298, 0.932402]` | `[1.394383, 1.389476, 0.3698464]` |

### one_person与空槽确实影响统计

空人物帧比例按`1-active_person_frames.count/model_positions.count`计算，单位是位置占比，不是“单人动作样本比例”；person1占用率为从all有效计数减去person0有效计数后除以一个槽的总位置数。

| 协议 | 两槽合计空人物帧位置 (%) | person0空帧位置 (%) | person1非零帧位置 (%) |
|---|---:|---:|---:|
| NTU60 XSub | 40.9593 | 0.0589 | 18.1403 |
| NTU60 XView | 41.1163 | 0.0738 | 17.8412 |
| NTU120 XSub | 0.2650 | 0.0691 | 99.5392 |
| NTU120 XSet | 0.2426 | 0.0757 | 99.5905 |
| PKUv1 XSub | 45.9394 | 0.0976 | 8.2188 |
| PKUv2 XSub | 49.2579 | 0.2817 | 1.7659 |

NTU120的高第二槽占用率不能认定为几乎全是真实双人动作：[当前本地align_frames实现](D:/program/MacDiff/data/ntu120/seq_transformation.py:118)会把单人序列复制到person1，已有文本流水线也处理duplicate_person1。此与附件现象相容，但没有实际服务器NPZ，不能由这份控制台重建复制样本比例。

NTU60 XSub的all/model_positions均值为`[-0.003298144,-0.137521,-0.02636741]`，var为`[0.09355884,0.1093626,0.06344149]`，而person0见首表。保留全部两槽时，均值的Y/Z更接近A，但各轴方差仍不接近A；不能仅凭均值相似就认定A来源。person0含空位置与排空位置的var分别为`[.04780319,.159242,.0464151]`和`[.04783136,.1593072,.04644118]`，说明当前person0口径内是否排空帧只产生很小差别，主要不同来自换到全部槽位及其数据分布。

PKUv2最直观：H用于**all/model_positions**时normalized mean=`[-.0004402994,-.0001566254,-.004618677]`，std=`[1.00183,1.000317,.9914286]`，很接近0/1；用于当前PT person0时Z mean=.932402、std=.3698464。H数值上更符合双槽含空位置统计，不能直接视为当前person0的单位标准化，也不等于已证明原H的计算来源。

### 对训练解释的影响与配置差异

1. **旧A对当前NTU60 XSub person0没有实现零均值/单位方差。** 各轴方差/A分别约`[2.320543,1.978161,2.129133]`。固定扩散系数及单位高斯噪声下，中心化信号方差意义的SNR是“假设标准化后方差为1”时的这些倍数；它不是已有run之间的LP收益，也不是观测到训练异常。FM同样受干净骨架/噪声相对尺度与速度目标幅度影响。固定仿射预处理不要求一定达到0/1，本审计不能证明原模型归一化错误或换统计必然改善。
2. **不能据此把原生随机波动、文本差异或T28下降归因于归一化。** 多数NTU60原版/文本组均使用A；共同常数不匹配不直接解释组间因果。当前没有新归一化PT→LP结果，已有T13/B同时改变batch与LR，也不是此新统计的验证。
3. **配置一致性优先核查PKUv2 PT→LP。** 附件报告PKUv2 LP和FT、PKUv1 FT、NTU120 XSub/XSet FT与各自PT变换不同；本地逐项确认PKUv2 PT=H/self_shiftFalse，LP=A/self_shiftFalse，PKUv1 PT/LP=A/self_shiftTrue而FT=A/self_shiftFalse。FT可更新encoder，不能把所有FT差异与冻结LP的输入错配等同。将本次PKUv2 PT person0统计代入LP的A，推算Z normalized mean约18.2228、std约3.4972；这是把同一审计分布套用另一组参数的计算，不是实际LP裁剪管线实测或LP掉点定因。
4. **PKU有非连续时间padding边界。** PKUv1有66/18840（.3503%）、PKUv2有24/5299（.4529%）样本的非零时间帧不构成连续前缀。两组valid-frame计数规则差异仍为0，说明“数量一致”不等于“有效帧都在前N帧”。原生crop按前valid_frame_num范围取数据，存在保留内部空帧/截掉后部有效帧的可能；此次遵循原函数，没有修数据或改变模型，不能凭数量归因LP问题。

所有打印行的mean/var→normalized mean/std关系已用NumPy复核，误差落在控制台舍入范围；self_shiftFalse五组的含空位置统计还满足零填充混合矩关系。该核对是日志内部一致性验证，不是重新访问NPZ独立计算。

**当前决策：** 仅分析记录，不修改归一化、人物开关、数据处理或训练命令，不新增LP成绩编号。若后续选新统计实验，当前one_person PT优先审阅首表person0/model_positions一组；旧checkpoint的LP继续沿用其PT归一化，不单独重算/替换LP参数。需要完整精度或审查raw_valid时，优先核对本次服务器输出目录中的JSON/CSV，不由控制台反推伪造完整文件，也不默认重跑。

## 2026-10-22新增：训练集统计与现有配置对比脚本

用户要求区分one_person与非one_person重新核算。已实现[compute_skeleton_normalization.py](D:/program/MacDiff/tools/compute_skeleton_normalization.py)，默认同时计算person0和全部人物；只读NPZ的`x_train`，不读取`x_test`或标签。不修改数据、PT/LP配置、checkpoint或已有实验的归一化。脚本编写时数据不在本地；现在已收到顶部服务器控制台统计，以下命令保留为原运行方式，不是默认重跑建议。

在服务器仓库根目录运行NTU60 XSub（CPU，不占GPU；依赖仓库已有的NumPy/PyYAML/Torch）：

```bash
python tools/compute_skeleton_normalization.py --configs config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml --view both --person-mode compare --draws-per-sample 1 --seed 0 --output-dir output_dir/normalization_audit/ntu60_xsub
```

一次计算六个数据协议，并分别对比各目录已有PT/LP/FT配置：

```bash
python tools/compute_skeleton_normalization.py --configs config/ntu60_xsub_joint/pretrain_madiff_flow_matching.yaml config/ntu60_xview_joint/pretrain_madiff.yaml config/ntu120_xsub_joint/pretrain_madiff.yaml config/ntu120_xset_joint/pretrain_madiff.yaml config/pkuv1_xsub_joint/pretrain_madiff.yaml config/pkuv2_xsub_joint/pretrain_madiff.yaml --view both --person-mode compare --draws-per-sample 1 --seed 0 --output-dir output_dir/normalization_audit/all_datasets
```

数据路径默认取各配置的`train_feeder_args.data_path`，按仓库根目录解析，与训练命令一致。单个配置可加`--data-path /实际路径/NTU60_XSub.npz`覆盖；多配置时不允许共用这一覆盖参数。还可用训练集专用`.npy`输入：NPY走mmap，NPZ以小块流式解压`x_train.npy`，不会因`np.load(npz,mmap_mode=...)`实际加载整个数组。默认每块16个样本。Fortran存储的NPZ会明确拒绝，需先单独提取x_train.npy再用NPY模式；不静默转换成大型临时文件。

### 统计口径

脚本默认同时输出以下组合，JSON/CSV均保留明确标签：

| 维度 | 取值与含义 |
|---|---|
| 人物 | `person0`仅固定第0槽；`all`全部槽，绝不把空person0替换为person1 |
| 输入视图 | `raw_valid`为当前NPZ已有预处理后的坐标、去掉全零时间帧，尚未做feeder裁剪；`model_input`复用仓库原生裁剪/resize/配置中的flip和旋转 |
| 空槽处理 | `model_positions`保留该视图全部位置，展示模型实际收到的分布；`active_person_frames`排除所有关节XYZ均为0的整个人物帧，展示有效骨架分布 |
| 中心化 | 同时统计self_shift=False/True；True按每个人独立在T、V上减均值，匹配模型公式，不跨人物中心化 |

单个坐标为0不会被删除，位于有效人物帧内的全零关节也保留，因为可能是合法根关节，不能仅凭0推断缺失。有效人物帧mask在中心化**之前**确定；中心化让原本空的位置变成非零时，也不会误判为有效。全零样本无法执行原生crop，会跳过并明确计数；空person0但person1非空的样本保留，person0全位置统计仍计入0。

统计基于干净骨架：不加入joint/frame/motion/impulse/source_rot增强，也不加diffusion/FM噪声。`model_input`含训练配置本来作用于干净骨架的随机crop/flip/rotation，默认每样本1次、seed0，因此是固定采样的训练输入分布估计。需要更充分估计随机变换时可提高`--draws-per-sample`，原始视图仍只统计每样本一次；它不是对所有可能crop的精确积分。`--max-samples N`只读前N个样本作preview，不作为完整训练集统计；默认0扫描全部训练集。进度每2000样本打印。

两个视图都按每个XYZ坐标向量等权汇总：原始视图长序列/有效帧多的样本权重大；模型全位置视图每样本固定120帧，样本权重一致。self_shift/矩计算使用float64提高数值稳定性，裁剪和旋转复用CPU Torch，不能声称与模型FP32逐位一致。`model_input`保留原生feeder的sum-based有效帧规则；同时记录它与any-nonzero规则不一致、非连续时间padding等异常计数，避免悄悄修正数据管线。

### 输出与解读

- 每个主配置一份JSON，记录数据文件形状/dtype/大小/mtime、NPZ训练成员CRC、配置SHA、样本计数、全部统计组合及对比结果。这些元数据不等于数据内容的完整SHA。
- `comparison.csv`汇总每轴mean、**var**、std，以及应用现有常数后的`normalized_mean`、`normalized_std`、`variance_ratio`。中文表格软件可直接打开。
- `normalized_mean`接近0且`normalized_std`接近1，表示现有常数对该统计口径接近标准化；偏离不等于已证明LP会下降。
- 对比默认包含主配置及同目录`linprobe_madiff.yaml`/`finetune_madiff.yaml`；`--compare-configs`可指定其他参考配置。所有参考常数都应用于本次审计的PT输入视图，不另执行LP/FT的crop管线，不能当LP实际分布测量。
- `self_shift=True`时对比使用的有效input_mean为0；建议候选mean也固定0，保存的configured_mean仍保留便于追溯。空数据或某轴方差为0时，`candidate_usable=False`，不生成可复制的零方差候选。
- 发现PT与LP/FT的有效mean、var或self_shift不一致会显式列出。当前PKUv2 PT使用H、LP使用A，脚本会报告该差异，不替用户改配置。

**如何选用：** 当前FM/native的PT为one_person=True时，优先看`model_input/person0`及当前self_shift的行，结合空人物帧比例审查；双人PT看`all`对应行。`model_positions`与`active_person_frames`分别回答实际输入分布、有效骨架分布的问题，不能未经说明混为一组。无论LP是否读取两个槽，评估同一冻结encoder仍应沿用其PT的那组归一化，不因LP有两个人物就重新换成all统计。建议候选只供新实验审阅，脚本不会自动替换。

仅看NPZ坐标、不执行模型输入变换时用`--view raw`，可在无Torch环境运行；YAML读取仍需PyYAML，也支持JSON形式的配置。没有自动扫描测试集或将train/test合并的选项。

本地验证：`python -m unittest tests.test_skeleton_normalization -v`共11项，9通过、2因缺PyYAML/Torch跳过。真实压缩/未压缩NPZ流式读取、训练集专用NPY mmap、布局、数值稳定性、person0/空槽/合法0/self_shift、train-only完整CLI与CSV/JSON输出均已执行；Python3.8语法通过。顶部已收到六协议服务器实跑统计，支持该环境实际执行了对应路径，但未收到服务器单测输出，不能把本地2个跳过项改记为通过。真实NPZ仍不在本地，本地没有独立重算或新归一化训练结果。

## 以下为历史配置审查快照

> 2026-10-07补充：以下31 YAML/B生效0/NTU60 XSub全A是2026-10-04的历史扫描快照。当前有35份YAML，新增no-share、no-share_st02及原版B的PT/LP四份；`pretrain_madiff_norm_b.yaml`和`linprobe_madiff_norm_b.yaml`已启用B，原版B也已取得LP85.183164%。本次没有重写旧快照的逐项计数。最新实验与保存缺项见 [实验结果总表](D:/program/MacDiff/EXPERIMENT_RESULTS.md)。

范围：`config/` 全部31份YAML；按 input_mean 与紧邻 input_var 成对统计，包含注释。共8组不同值、50次成对出现，其中31次生效、19次注释。模型构造器默认 mean=[0,0,0]、var=[1,1,1] 不计入配置分组；这些YAML均显式指定了一组生效值。未修改训练配置或模型。

这里的生效仅指 YAML 未注释；self_shift=True 时模型先按样本中心化并强制 input_mean=[0,0,0]，input_var 仍使用配置值。

## 不同数值组

| 组 | input_mean | input_var | 生效次数 | 注释次数 | 出现位置含义 |
|---|---|---|---:|---:|---|
| A | [-0.0058, -0.1333, -0.0246] | [0.0206, 0.0805, 0.0218] | 28 | 1 | 旧组；多数配置共用 |
| B | [-0.0024, -0.2132, -0.0446] | [0.0525, 0.1527, 0.0513] | 0 | 6 | pretrain/LP 的 #new；多个协议共用候选值 |
| C | [-0.0034, -0.1322, -0.0271] | [0.088325, 0.106987, 0.065367] | 0 | 1 | NTU60 XSub finetune 的 #new 候选值 |
| D | [-0.0027, -0.1353, -0.0244] | [0.0211, 0.0826, 0.0241] | 0 | 2 | NTU60 XView 注释候选值 |
| E | [-0.0021, -0.2198, -0.0404] | [0.0298, 0.1424, 0.0358] | 1 | 0 | NTU120 XSub finetune 生效值 |
| F | [-0.0016, -0.2247, -0.0321] | [0.0306, 0.1451, 0.0398] | 1 | 0 | NTU120 XSet finetune 生效值 |
| G | [-0.0686, -0.0241, 1.513] | [0.2834, 0.2447, 2.0166] | 0 | 5 | 配置注释标为 pkuv1 2man |
| H | [-0.0459, -0.068, 1.3642] | [0.2356, 0.2351, 1.9492] | 1 | 4 | 配置注释标为 pkuv2 2man |

## 每份配置的分组与状态

| 配置 | 生效组 | 注释候选组 | 显式 self_shift | 显式 one_person |
|---|---|---|---|---|
| [config/ntu120_xset_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/finetune_madiff.yaml) | F | 无 | 未设置 | 未设置 |
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/finetune_madiff.yaml) | E | 无 | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml) | A | C | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml) | A | B | 未设置 | 未设置 |
| [config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml) | A | 无 | False | True |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml) | A | 无 | False | False |
| [config/ntu60_xsub_joint/pretrain_madiff_text.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml) | A | 无 | 未设置 | True |
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml) | A | D | 未设置 | 未设置 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml) | A | D | 未设置 | 未设置 |
| [config/ntu60_xview_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/pretrain_madiff.yaml) | A | 无 | 未设置 | 未设置 |
| [config/pkuv1_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/finetune_madiff.yaml) | A | 无 | 未设置 | 未设置 |
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml) | A | G,H | True | 未设置 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml) | A | G,H | 未设置 | 未设置 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml) | H | A,G | 未设置 | 未设置 |

## 所有出现位置

### A：旧组；多数配置共用

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml:43) | 生效 | 43 / 44 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml:42) | 生效 | 42 / 43 |
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml:54) | 生效 | 54 / 55 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml:51) | 生效 | 51 / 52 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml:42) | 生效 | 42 / 43 |
| [config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_ose_peer.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2.yaml:39) | 生效 | 39 / 40 |
| [config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_stage2_dense_ose.yaml:38) | 生效 | 38 / 39 |
| [config/ntu60_xsub_joint/pretrain_madiff_text.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_ema_bidirectional_shared.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_sample_target_blend_shared.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_rms_st01.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_fixed_rms.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_sentence_sample_target_blend_shared.yaml:41) | 生效 | 41 / 42 |
| [config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff_text_st01.yaml:40) | 生效 | 40 / 41 |
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml:47) | 生效 | 47 / 48 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml:47) | 生效 | 47 / 48 |
| [config/ntu60_xview_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/pretrain_madiff.yaml:45) | 生效 | 45 / 46 |
| [config/pkuv1_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/finetune_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:48) | 生效 | 48 / 49 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:46) | 生效 | 46 / 47 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:50) | 生效 | 50 / 51 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:48) | 生效 | 48 / 49 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:46) | 注释 | 46 / 47 |

### B：pretrain/LP 的 #new；多个协议共用候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/linprobe_madiff.yaml:47) | 注释 | 47 / 48 |
| [config/ntu120_xset_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/pretrain_madiff.yaml:46) | 注释 | 46 / 47 |
| [config/ntu120_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/linprobe_madiff.yaml:47) | 注释 | 47 / 48 |
| [config/ntu120_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/pretrain_madiff.yaml:45) | 注释 | 45 / 46 |
| [config/ntu60_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/linprobe_madiff.yaml:48) | 注释 | 48 / 49 |
| [config/ntu60_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/pretrain_madiff.yaml:45) | 注释 | 45 / 46 |

### C：NTU60 XSub finetune 的 #new 候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu60_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xsub_joint/finetune_madiff.yaml:51) | 注释 | 51 / 52 |

### D：NTU60 XView 注释候选值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu60_xview_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/finetune_madiff.yaml:50) | 注释 | 50 / 51 |
| [config/ntu60_xview_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/ntu60_xview_joint/linprobe_madiff.yaml:49) | 注释 | 49 / 50 |

### E：NTU120 XSub finetune 生效值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xsub_joint/finetune_madiff.yaml:45) | 生效 | 45 / 46 |

### F：NTU120 XSet finetune 生效值

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/ntu120_xset_joint/finetune_madiff.yaml](D:/program/MacDiff/config/ntu120_xset_joint/finetune_madiff.yaml:45) | 生效 | 45 / 46 |

### G：配置注释标为 pkuv1 2man

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:51) | 注释 | 51 / 52 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:49) | 注释 | 49 / 50 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:53) | 注释 | 53 / 54 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:51) | 注释 | 51 / 52 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:49) | 注释 | 49 / 50 |

### H：配置注释标为 pkuv2 2man

| 配置 | 状态 | mean/var 行号 |
|---|---|---|
| [config/pkuv1_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/linprobe_madiff.yaml:54) | 注释 | 54 / 55 |
| [config/pkuv1_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv1_xsub_joint/pretrain_madiff.yaml:52) | 注释 | 52 / 53 |
| [config/pkuv2_xsub_joint/finetune_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/finetune_madiff.yaml:56) | 注释 | 56 / 57 |
| [config/pkuv2_xsub_joint/linprobe_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/linprobe_madiff.yaml:54) | 注释 | 54 / 55 |
| [config/pkuv2_xsub_joint/pretrain_madiff.yaml](D:/program/MacDiff/config/pkuv2_xsub_joint/pretrain_madiff.yaml:52) | 生效 | 52 / 53 |

## 数值比较与运行差异

A与D接近，D/A方差约为[1.0243,1.0261,1.1055]。B与E/F的Y均值、Y方差接近，但X/Z方差不同；它们不是同组。E与F也仅接近而不相同。C的均值接近A，但方差明显更大，特别是X轴，不能与B混为一组。G/H的Z均值约1.4–1.5、Z方差约2，与NTU各组尺度明显不同。数值接近不能证明统计来自同一数据集或相同预处理。

当前所有NTU60 XSub配置（包括文本、OSE、Stage2）均生效A，pretrain与LP一致。NTU60 XView也全用A。NTU120 XSub/XSet的pretrain/LP用A，finetune分别用E/F。PKUv1全部用A；PKUv2 pretrain用H、LP/finetune用A。后两种pretrain与finetune/LP不同的情况必须连同self_shift和是否冻结encoder解释，不能仅凭不同数值判为错误。

B在NTU60 XSub、NTU120 XSub、NTU120 XSet的pretrain/LP共6处出现且全部注释；C仅在NTU60 XSub finetune一处注释。Git初始提交0aafec2已经包含B/C；未找到计算脚本、统计输出或来源记录。不能确证B源自NTU120，也不能确证C就是按NTU60 XSub当前训练预处理计算的。

## A换B对当前训练的影响

代码按 (raw-mean)/sqrt(var) 标准化。下面公式使用当前NTU60设置self_shift=False，并固定同一原始坐标。

| 轴 | A标准差 | B标准差 | B归一化值相对A的比例 | B归一化值的额外偏移 | 中心化方差/SNR相对A |
|---|---:|---:|---:|---:|---:|
| X | 0.143527 | 0.229129 | 0.626403 | -0.014839 | 0.392381 |
| Y | 0.283725 | 0.390768 | 0.726070 | +0.204469 | 0.527177 |
| Z | 0.147648 | 0.226495 | 0.651883 | +0.088302 | 0.424951 |

即 z_B=[0.626403,0.726070,0.651883]*z_A+[-0.014839,0.204469,0.088302]，为逐通道仿射变换。各轴坐标变化幅度缩小27.4%–37.4%；固定时间步下，以中心化信号方差定义的有效SNR降到39.2%–52.7%。均值偏移会影响二阶能量，因此这里的SNR比例不用于未经中心化的原点总能量。

C与B并不等价。A换C的坐标变化幅度比例是[0.482938,0.867426,0.577496]，额外偏移[-0.008075,-0.003363,+0.009778]；其方向间缩放差异更大。

当前骨架扩散 q_sample=a_t*z0+sigma_t*epsilon；epsilon仍是单位高斯。A换B改变骨架encoder输入、原生骨架重建与T→S任务的有效噪声强度，位置/时间embedding与内容幅度的相对关系也会变化。S→T仍预测文本噪声，文本目标的尺度不会直接按骨架方差比例改变，但encoder条件及T→S训练remap/共享decoder产生的间接路径会受影响。

原生与T→S均回归epsilon，loss的目标单位未改变，不能机械把loss乘以old_var/new_var。较小的干净信号可能让模型更容易从noisy输入估计噪声；更低的噪声MSE不能单独证明语义条件使用更强或LP更好。仿射变换不增加样本信息，但优化、正则及扩散训练不是对该变换自动不变，性能可能升也可能降，幅度需实测。

input_mean/input_var是普通Python属性，没有注册为parameter/buffer。LP先用自己的YAML构造模型，再load checkpoint[model]；更改LP统计值会实际生效，旧checkpoint不会自动覆盖它。当前LP冻结encoder，head中的BatchNorm不能保证补偿前端归一化造成的特征变化；仅在旧checkpoint上换LP统计，不能验证新归一化预训练的效果。

建议保留当前512修复实验的A基线。若比较B，使用独立输出目录，从头预训练并让该实验的LP同样使用B，其他训练权重、share、batch、crop、噪声日程与seed固定；当前缓存内容不因这两个骨架输入参数而改变，不需要重生成文本cache。重算数据统计时应明确train split、当前裁剪、person0/两人、空人物、时间padding及旋转口径，仅凭历史#new标签不能认定数值正确。

## 代码证据

- [model/transformer_macdiff.py](D:/program/MacDiff/model/transformer_macdiff.py:363)：var开平方得到std；self_shift覆盖mean。
- [model/transformer_macdiff.py](D:/program/MacDiff/model/transformer_macdiff.py:475)：骨架归一化。
- [guided_diffusion/gaussian_diffusion.py](D:/program/MacDiff/guided_diffusion/gaussian_diffusion.py:201)：q_sample。
- [model/transformer_macdiff_text.py](D:/program/MacDiff/model/transformer_macdiff_text.py:306)：T→S噪声回归；316行S→T文本噪声回归。
- [model/transformer_downstream.py](D:/program/MacDiff/model/transformer_downstream.py:322)：LP归一化参数不是state_dict buffer。
- [main_linprobe.py](D:/program/MacDiff/main_linprobe.py:210)：先按YAML构造模型后加载权重；243行冻结encoder。
